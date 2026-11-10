"""Manuscript metrics, with explicit counts and undefined denominators."""

import numpy as np


def evaluate(capacity_true, capacity_predicted, rmi_true, rmi_predicted):
    y, pred, label, assigned = [
        np.asarray(v, dtype=float)
        for v in (capacity_true, capacity_predicted, rmi_true, rmi_predicted)
    ]
    if (
        any(a.ndim != 1 for a in (y, pred, label, assigned))
        or len({len(a) for a in (y, pred, label, assigned)}) != 1
    ):
        raise ValueError("Targets and predictions must be vectors of the same length")
    if not np.isfinite(pred).all() or not np.isfinite(assigned).all():
        raise ValueError("Predictions must be finite")
    if np.isinf(y).any() or np.isinf(label).any():
        raise ValueError("Targets may be missing but cannot be infinite")
    valid_c, valid_r = np.isfinite(y), np.isfinite(label)
    if np.any((assigned < 0) | (assigned > 3) | (assigned != np.floor(assigned))):
        raise ValueError("Predicted classes must be integers from zero to three")
    if np.any(
        (label[valid_r] < 0)
        | (label[valid_r] > 3)
        | (label[valid_r] != np.floor(label[valid_r]))
    ):
        raise ValueError("Valid target classes must be integers from zero to three")
    result = {
        "capacity_records": int(valid_c.sum()),
        "rmi_records": int(valid_r.sum()),
        "mae": None,
        "nmae_percent": None,
        "r_squared": None,
        "macro_f1": None,
    }
    if valid_c.any():
        actual, estimates = y[valid_c], pred[valid_c]
        error = np.mean(np.abs(actual - estimates))
        span = np.max(actual) - np.min(actual)
        total = np.sum((actual - np.mean(actual)) ** 2)
        result.update(
            mae=float(error),
            nmae_percent=float(100 * error / span) if span else None,
            r_squared=float(1 - np.sum((actual - estimates) ** 2) / total)
            if total
            else None,
        )
    if valid_r.any():
        scores = []
        for cls in range(4):
            truth, guess = label[valid_r] == cls, assigned[valid_r] == cls
            tp = int(np.sum(truth & guess))
            denominator = int(np.sum(truth) + np.sum(guess))
            scores.append(2 * tp / denominator if denominator else 0.0)
        result["macro_f1"] = float(np.mean(scores))
    return result


def summarize_runs(results):
    if len(results) < 2:
        raise ValueError(
            "Sample standard deviation requires at least two completed runs"
        )
    summary = {}
    for name in ("mae", "nmae_percent", "r_squared", "macro_f1"):
        values = [r[name] for r in results]
        defined = all(v is not None and np.isfinite(v) for v in values)
        summary[name] = {
            "runs": len(values),
            "defined_runs": sum(v is not None for v in values),
            "mean": float(np.mean(values)) if defined else None,
            "sample_standard_deviation": float(np.std(values, ddof=1))
            if defined
            else None,
        }
    return summary
