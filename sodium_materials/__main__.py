"""Command line entry for input validation, explicit training and prediction."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
from .preprocessing import (
    RolePreprocessor,
    read_table,
    targets,
    fit_capacity_scaler,
    validate_schema,
    ROLES,
)
from .partitions import validate_partitions


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    m, t = config["model"], config["training"]
    for name in ("width", "heads", "feedforward_width", "head_width", "classes"):
        if type(m[name]) is not int or m[name] < 1:
            raise ValueError("Model sizes must be positive integers")
    if m["classes"] != 4 or m["width"] % m["heads"] or not 0 <= m["dropout"] < 1:
        raise ValueError("Invalid attention or dropout configuration")
    for name in ("layer_norm_epsilon", "role_embedding_std"):
        if not math.isfinite(m[name]) or m[name] <= 0:
            raise ValueError("Invalid model scale")
    for name in ("batch_size", "max_epochs", "patience"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError("Training counts must be positive integers")
    for name in ("learning_rate", "optimizer_epsilon", "capacity_weight", "rmi_weight"):
        if not math.isfinite(t[name]) or t[name] <= 0:
            raise ValueError("Learning rate, epsilon and task weights must be positive")
    for name in ("weight_decay", "min_improvement"):
        if not math.isfinite(t[name]) or t[name] < 0:
            raise ValueError(
                "Regularization and stopping threshold must be nonnegative"
            )
    if len(t["betas"]) != 2 or any(not 0 <= b < 1 for b in t["betas"]):
        raise ValueError("Invalid optimizer moment coefficients")
    seeds = t["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError(
            "Provide distinct integer seeds, with at least two runs for summary statistics"
        )


def prepare(args):
    config, schema, protocol = (
        load_json(args.config),
        load_json(args.schema),
        load_json(args.protocol),
    )
    validate_config(config)
    validate_schema(schema)
    rows = read_table(args.data)
    partitions = validate_partitions(
        rows, read_table(args.splits), schema, protocol, args.similarity
    )
    preprocessing = RolePreprocessor(schema).fit(partitions["train"])
    encoded = {k: preprocessing.transform(v) for k, v in partitions.items()}
    labels = {k: targets(v, schema) for k, v in partitions.items()}
    for split in ("train", "validation"):
        if not all(np.isfinite(v).any() for v in labels[split]):
            raise ValueError("Both tasks need valid labels in " + split)
    scaler = fit_capacity_scaler(
        labels["train"][0], schema["capacity_standard_deviation_ddof"]
    )
    dims = {r: encoded["train"][r].shape[1] for r in ROLES}
    summary = {
        "setting": protocol["setting"],
        "input_dimensions": dims,
        "subsets": {
            k: {
                "records": len(v),
                "capacity_labels": int(np.isfinite(labels[k][0]).sum()),
                "rmi_labels": int(np.isfinite(labels[k][1]).sum()),
                "rmi_class_counts": [int(np.sum(labels[k][1] == c)) for c in range(4)],
            }
            for k, v in partitions.items()
        },
        "similarity_check": "Complete supplied training to test score matrix checked; score computation is external",
        "source_sha256": {
            k: sha(getattr(args, k))
            for k in ("data", "splits", "schema", "protocol", "config", "similarity")
        },
    }
    return (
        config,
        schema,
        preprocessing,
        scaler,
        dims,
        encoded,
        labels,
        partitions,
        summary,
    )


def train(args):
    from .training import tensors, train_seed, write_predictions
    from .metrics import summarize_runs
    import torch

    if args.output.exists():
        raise ValueError("Output exists; choose a new directory")
    config, schema, pre, scaler, dims, encoded, labels, partitions, summary = prepare(
        args
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "input_summary.json", summary)
    write_json(args.output / "config.json", config)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "implementation_origin": "New code written from the manuscript description",
    }
    write_json(args.output / "environment.json", environment)
    data = {k: tensors(encoded[k], *labels[k], scaler) for k in ("train", "validation")}
    results = []
    for seed in config["training"]["seeds"]:
        destination = args.output / ("seed_" + str(seed))
        model = train_seed(
            dims, config, data["train"], data["validation"], seed, destination
        )
        write_json(
            destination / "model_metadata.json",
            {
                "model": config["model"],
                "dimensions": dims,
                "preprocessing": pre.export(),
                "capacity_scaler": scaler,
                "weights_sha256": sha(destination / "best_weights.pt"),
                "environment": environment,
            },
        )
        results.append(
            write_predictions(
                model,
                encoded["test"],
                [r[schema["record_id"]] for r in partitions["test"]],
                *labels["test"],
                scaler,
                destination,
                config["training"]["batch_size"],
            )
        )
    write_json(args.output / "summary.json", summarize_runs(results))
    print("Completed", len(results), "new runs. Results are in", args.output)


def predict(args):
    import torch
    from .model import MaterialSystemModel
    from .training import write_predictions

    if args.output.exists():
        raise ValueError("Output exists; choose a new directory")
    metadata = load_json(args.run / "model_metadata.json")
    weights = args.run / "best_weights.pt"
    if sha(weights) != metadata["weights_sha256"]:
        raise ValueError("Checkpoint checksum differs from model metadata")
    pre = RolePreprocessor.restore(metadata["preprocessing"])
    rows = read_table(args.data)
    ids = [r[pre.schema["record_id"]] for r in rows]
    if any(not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Prediction identifiers must be nonempty and unique")
    encoded = pre.transform(rows)
    # Targets are optional during prediction; absent targets produce null metrics.
    labeled = [
        dict(
            r,
            **{
                k: r.get(k, "")
                for k in (pre.schema["capacity_target"], pre.schema["rmi_target"])
            },
        )
        for r in rows
    ]
    cap, rmi = targets(labeled, pre.schema)
    model = MaterialSystemModel(metadata["dimensions"], metadata["model"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    args.output.mkdir(parents=True, exist_ok=False)
    write_predictions(
        model, encoded, ids, cap, rmi, metadata["capacity_scaler"], args.output, 32
    )
    print("Wrote predictions to", args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Model implementation from the sodium ion material systems manuscript"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train"):
        sub = commands.add_parser(
            command,
            help="Check inputs only"
            if command == "validate"
            else "Start new training runs explicitly",
        )
        for name in ("data", "schema", "splits", "protocol", "similarity"):
            sub.add_argument("--" + name, type=Path, required=True)
        sub.add_argument(
            "--config",
            type=Path,
            default=Path(__file__).resolve().parents[1] / "configs/model.json",
        )
        if command == "train":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser(
        "predict", help="Predict using a checkpoint produced by this implementation"
    )
    for name in ("data", "run", "output"):
        sub.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            print(json.dumps(prepare(args)[-1], indent=2, allow_nan=False))
        elif args.command == "train":
            train(args)
        else:
            predict(args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, "Input or execution error: {}\n".format(exc))


if __name__ == "__main__":
    main()
