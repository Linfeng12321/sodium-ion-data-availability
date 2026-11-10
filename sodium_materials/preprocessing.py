"""Training subset transformations for four component roles."""

import csv
import math
from pathlib import Path
import numpy as np

ROLES = ("anode", "cathode", "electrolyte", "testing")


def read_table(path):
    """Read a values only CSV without automatic missing value inference."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if (
            not headers
            or any(not h for h in headers)
            or len(headers) != len(set(headers))
        ):
            raise ValueError("Input headers must be nonempty and unique")
        rows = list(reader)
    if not rows or any(
        None in row or any(v is None for v in row.values()) for row in rows
    ):
        raise ValueError(
            "Input must contain complete rows with the same width as its header"
        )
    return rows


def validate_schema(schema):
    for name in ("record_id", "capacity_target", "rmi_target"):
        if not isinstance(schema.get(name), str) or not schema[name]:
            raise ValueError("Specify " + name + " in the input schema")
    if len({schema[k] for k in ("record_id", "capacity_target", "rmi_target")}) != 3:
        raise ValueError("Identifiers and task targets must be distinct columns")
    if set(schema.get("roles", {})) != set(ROLES):
        raise ValueError("Specify exactly four component roles")
    columns = []
    for role in ROLES:
        spec = schema["roles"][role]
        if set(spec) != {"categorical", "numerical"}:
            raise ValueError("Each role needs categorical and numerical field lists")
        if not all(isinstance(spec[k], list) for k in ("categorical", "numerical")):
            raise ValueError("Categorical and numerical fields must be lists")
        fields = spec["categorical"] + spec["numerical"]
        if not fields or any(not isinstance(c, str) or not c for c in fields):
            raise ValueError("Specify at least one named input field for " + role)
        columns.extend(fields)
    if len(columns) != len(set(columns)):
        raise ValueError("Each input field must belong to exactly one role")
    forbidden = {schema[k] for k in ("record_id", "capacity_target", "rmi_target")}
    if forbidden.intersection(columns):
        raise ValueError(
            "Do not use record identifiers or task targets as model inputs"
        )
    markers = schema.get("missing_markers")
    if not isinstance(markers, list) or any(not isinstance(v, str) for v in markers):
        raise ValueError("Specify the missing value markers for the selected fields")
    if any(v.strip().lower() == "na" for v in markers):
        raise ValueError("Na must remain a valid chemistry entry in model inputs")
    if schema.get("numerical_statistics") not in (
        "before_imputation",
        "after_imputation",
    ):
        raise ValueError(
            "Select before_imputation or after_imputation for numerical statistics"
        )
    if type(schema.get("capacity_standard_deviation_ddof")) is not int or schema[
        "capacity_standard_deviation_ddof"
    ] not in (0, 1):
        raise ValueError("Explicitly select capacity standard deviation ddof 0 or 1")


def missing(value, markers):
    return (
        value is None
        or str(value).strip() == ""
        or str(value).strip().lower() in markers
    )


def number(value, markers):
    if missing(value, markers):
        return float("nan")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            "Nonfinite values must be explicit missing markers, not numeric inputs"
        )
    return value


def targets(rows, schema):
    markers = {v.strip().lower() for v in schema["missing_markers"]}
    cap = np.array(
        [number(r[schema["capacity_target"]], markers) for r in rows], dtype=np.float64
    )
    rmi = np.array(
        [number(r[schema["rmi_target"]], markers) for r in rows], dtype=np.float64
    )
    valid = np.isfinite(rmi)
    if np.any(
        (rmi[valid] != np.floor(rmi[valid])) | (rmi[valid] < 0) | (rmi[valid] > 3)
    ):
        raise ValueError(
            "Retention Milestone Index labels must be 0, 1, 2, 3 or missing"
        )
    return cap, rmi


def milestone_from_anchor(cycles, evidence_valid):
    """Convert an already reviewed 80% retention anchor or lower bound to a tier."""
    if type(evidence_valid) is not bool:
        raise ValueError("Evidence validity must be an explicit Boolean")
    if not evidence_valid:
        return None
    cycles = float(cycles)
    if not math.isfinite(cycles) or cycles < 1:
        raise ValueError("A valid anchor must contain at least one cycle")
    return sum(cycles >= boundary for boundary in (10, 100, 1000))


class RolePreprocessor:
    def __init__(self, schema):
        validate_schema(schema)
        self.schema = schema
        self.markers = {v.strip().lower() for v in schema["missing_markers"]}
        self.state = None

    def fit(self, rows):
        if not rows:
            raise ValueError("Training data are empty")
        state = {}
        for role in ROLES:
            config = self.schema["roles"][role]
            state[role] = {"categorical": {}, "numerical": {}}
            for column in config["categorical"]:
                # Index zero is a separate missing category, not the literal string Missing.
                vocabulary = sorted(
                    {
                        str(r[column]).strip()
                        for r in rows
                        if not missing(r[column], self.markers)
                    }
                )
                state[role]["categorical"][column] = vocabulary
            for column in config["numerical"]:
                values = np.array([number(r[column], self.markers) for r in rows])
                observed = values[np.isfinite(values)]
                median = float(np.median(observed)) if len(observed) else 0.0
                sample = (
                    np.where(np.isfinite(values), values, median)
                    if self.schema["numerical_statistics"] == "after_imputation"
                    else observed
                )
                mean = float(np.mean(sample)) if len(sample) else 0.0
                scale = float(np.std(sample, ddof=0)) if len(sample) else 1.0
                state[role]["numerical"][column] = {
                    "median": median,
                    "mean": mean,
                    "scale": scale or 1.0,
                }
        self.state = state
        return self

    def transform(self, rows):
        if self.state is None:
            raise ValueError("Fit preprocessing on the training subset first")
        output = {}
        for role in ROLES:
            config = self.schema["roles"][role]
            parts, indicators = [], []
            for column in config["categorical"]:
                vocab = self.state[role]["categorical"][column]
                index = {v: i + 1 for i, v in enumerate(vocab)}
                encoded = np.zeros((len(rows), len(vocab) + 1), dtype=np.float32)
                for i, row in enumerate(rows):
                    value = row[column]
                    j = (
                        0
                        if missing(value, self.markers)
                        else index.get(str(value).strip())
                    )
                    if j is not None:
                        encoded[i, j] = 1.0
                parts.append(encoded)
            for column in config["numerical"]:
                spec = self.state[role]["numerical"][column]
                x = np.array([number(r[column], self.markers) for r in rows])
                absent = ~np.isfinite(x)
                values = (np.where(absent, spec["median"], x) - spec["mean"]) / spec[
                    "scale"
                ]
                parts.append(values[:, None].astype(np.float32))
                indicators.append(absent[:, None].astype(np.float32))
            output[role] = np.concatenate(parts + indicators, axis=1)
            if not np.isfinite(output[role]).all():
                raise ValueError("Transformed input is outside finite float32 range")
        return output

    def export(self):
        return {"schema": self.schema, "state": self.state}

    @classmethod
    def restore(cls, value):
        obj = cls(value["schema"])
        obj.state = value["state"]
        return obj


def fit_capacity_scaler(values, ddof):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) <= ddof:
        raise ValueError("Insufficient valid training capacity labels")
    scale = float(np.std(valid, ddof=ddof))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Training capacity needs a positive standard deviation")
    return {"mean": float(np.mean(valid)), "scale": scale, "ddof": ddof}
