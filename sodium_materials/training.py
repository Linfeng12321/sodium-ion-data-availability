"""Training and checkpoint utilities. Importing this module never starts training."""

import csv
import json
import math
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .model import MaterialSystemModel, make_optimizer
from .preprocessing import ROLES
from .metrics import evaluate


@dataclass
class EarlyStopping:
    patience: int
    minimum_improvement: float
    best: float = float("inf")
    reference: float = float("inf")
    waiting: int = 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Validation loss must be finite")
        improved_checkpoint = loss < self.best
        if improved_checkpoint:
            self.best = loss
        if loss < self.reference - self.minimum_improvement:
            self.reference, self.waiting = loss, 0
        else:
            self.waiting += 1
        return improved_checkpoint, self.waiting >= self.patience


def masked_loss(capacity_output, class_logits, capacity_target, rmi_target, config):
    capacity_mask, rmi_mask = (
        torch.isfinite(capacity_target),
        torch.isfinite(rmi_target),
    )
    pieces = []
    if capacity_mask.any():
        pieces.append(
            config["capacity_weight"]
            * F.mse_loss(capacity_output[capacity_mask], capacity_target[capacity_mask])
        )
    if rmi_mask.any():
        labels = rmi_target[rmi_mask]
        if torch.any((labels != labels.floor()) | (labels < 0) | (labels > 3)):
            raise ValueError("Invalid classification labels")
        pieces.append(
            config["rmi_weight"]
            * F.cross_entropy(
                class_logits[rmi_mask], labels.long(), label_smoothing=0.0
            )
        )
    return sum(pieces) if pieces else None


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def tensors(encoded, cap, rmi, scaler):
    x = {r: torch.tensor(encoded[r], dtype=torch.float32) for r in ROLES}
    y = torch.tensor((cap - scaler["mean"]) / scaler["scale"], dtype=torch.float32)
    labels = torch.tensor(rmi, dtype=torch.float32)
    return x, y, labels


def infer(model, x, batch_size):
    model.eval()
    capacity, logits = [], []
    with torch.no_grad():
        for i in range(0, len(x[ROLES[0]]), batch_size):
            cap, rmi = model({r: x[r][i : i + batch_size] for r in ROLES})
            capacity.append(cap)
            logits.append(rmi)
    if not capacity:
        raise ValueError("Inference requires at least one record")
    capacity, logits = torch.cat(capacity), torch.cat(logits)
    if not torch.isfinite(capacity).all() or not torch.isfinite(logits).all():
        raise ValueError("Model produced nonfinite values")
    return capacity, logits


def validation_loss(model, data, config):
    x, capacity, rmi = data
    output, logits = infer(model, x, config["batch_size"])
    # Means are over the whole subset per task, not the mean of batch means.
    loss = masked_loss(output, logits, capacity, rmi, config)
    if loss is None:
        raise ValueError("Validation has no valid task labels")
    return float(loss)


def train_seed(dimensions, config, train_data, validation_data, seed, output):
    """Explicit training entry, called only by the train command."""
    output.mkdir(exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MaterialSystemModel(dimensions, config["model"])
    optimizer = make_optimizer(model, config["training"])
    cfg = config["training"]
    tracker = EarlyStopping(cfg["patience"], cfg["min_improvement"])
    x, capacity, rmi = train_data
    generator = torch.Generator().manual_seed(seed)
    best_epoch, best_state = None, None
    started = time.monotonic()
    with (output / "training_log.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "validation_loss",
                "checkpoint_updated",
                "patience_count",
                "updates",
                "seconds",
            ],
        )
        writer.writeheader()
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(capacity), generator=generator)
            updates = 0
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                if not (
                    torch.isfinite(capacity[batch]).any()
                    or torch.isfinite(rmi[batch]).any()
                ):
                    continue
                optimizer.zero_grad(set_to_none=True)
                cap, logits = model({r: x[r][batch] for r in ROLES})
                loss = masked_loss(cap, logits, capacity[batch], rmi[batch], cfg)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                updates += 1
            value = validation_loss(model, validation_data, cfg)
            improved, stop = tracker.update(value)
            if improved:
                best_epoch, best_state = epoch, snapshot(model)
                torch.save(best_state, output / "best_weights.pt")
            writer.writerow(
                dict(
                    epoch=epoch,
                    validation_loss=value,
                    checkpoint_updated=improved,
                    patience_count=tracker.waiting,
                    updates=updates,
                    seconds=time.monotonic() - started,
                )
            )
            handle.flush()
            if stop:
                break
    if best_state is None:
        raise ValueError("No valid checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    report = {
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_validation_loss": tracker.best,
        "seconds": time.monotonic() - started,
        "termination": "early_stopping" if stop else "maximum_epochs",
    }
    (output / "training_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return model


def write_predictions(
    model, encoded, ids, raw_capacity, raw_rmi, scaler, output, batch_size
):
    x = {r: torch.tensor(encoded[r], dtype=torch.float32) for r in ROLES}
    cap, logits = infer(model, x, batch_size)
    predicted = cap.numpy() * scaler["scale"] + scaler["mean"]
    probabilities = logits.softmax(dim=1).numpy()
    assigned = probabilities.argmax(axis=1)
    metrics = evaluate(raw_capacity, predicted, raw_rmi, assigned)
    with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_id",
                "capacity_predicted",
                "rmi_predicted",
                "rmi_probability_0",
                "rmi_probability_1",
                "rmi_probability_2",
                "rmi_probability_3",
            ]
        )
        for identity, c, r, probs in zip(ids, predicted, assigned, probabilities):
            writer.writerow([identity, float(c), int(r)] + [float(v) for v in probs])
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics
