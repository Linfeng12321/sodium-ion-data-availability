"""Validate supplied partitions without generating new manuscript splits."""

import math
from .preprocessing import read_table


def validate_partitions(rows, assignments, schema, protocol, similarity_path):
    id_column = schema["record_id"]
    identifiers = [r[id_column] for r in rows]
    if any(not i.strip() for i in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("Record identifiers must be nonempty and unique")
    mapping = {}
    for entry in assignments:
        identity, split = entry["record_id"], entry["split"]
        if identity in mapping or split not in (
            "train",
            "validation",
            "test",
            "excluded",
        ):
            raise ValueError("Duplicate assignment or unsupported split")
        if split == "excluded" and not entry.get("reason", "").strip():
            raise ValueError("Excluded records need a reason in the split file")
        mapping[identity] = split
    if set(mapping) != set(identifiers):
        raise ValueError("Splits must identify every input record exactly once")
    result = {
        name: [r for r in rows if mapping[r[id_column]] == name]
        for name in ("train", "validation", "test")
    }
    expected = protocol.get("expected_counts")
    if not isinstance(expected, dict) or set(expected) != set(result):
        raise ValueError("Specify expected train, validation and test counts")
    if any(
        type(expected[k]) is not int or expected[k] < 1 or len(result[k]) != expected[k]
        for k in result
    ):
        raise ValueError("Split counts differ from the supplied evaluation protocol")
    setting = protocol.get("setting")
    if setting not in ("internal", "anode", "cathode", "electrolyte"):
        raise ValueError("Specify internal, anode, cathode or electrolyte evaluation")
    if setting == "internal" and expected != {
        "train": 860,
        "validation": 184,
        "test": 185,
    }:
        raise ValueError("The manuscript internal split uses 860, 184 and 185 records")
    if setting != "internal":
        column = protocol.get("family_column")
        held_out = protocol.get("held_out_families")
        if not column or not held_out or any(not str(v).strip() for v in held_out):
            raise ValueError("Specify family column and held out families")
        families = {k: {r[column].strip() for r in v} for k, v in result.items()}
        if any("" in s for s in families.values()):
            raise ValueError("Every evaluated record needs a family label")
        if families["test"] != set(held_out):
            raise ValueError("Test families differ from the declared held out families")
        if (families["train"] | families["validation"]) & families["test"]:
            raise ValueError(
                "Held out test families must not enter training or model selection"
            )
    definition = protocol.get("similarity_definition")
    threshold = protocol.get("similarity_threshold")
    if (
        not isinstance(definition, str)
        or not definition.strip()
        or type(threshold) not in (float, int)
        or not math.isfinite(threshold)
    ):
        raise ValueError(
            "Supply the similarity definition and a finite exclusion threshold"
        )
    # No similarity metric is invented: check a complete supplied train to test score table.
    train = {r[id_column] for r in result["train"]}
    test = {r[id_column] for r in result["test"]}
    seen = set()
    for entry in read_table(similarity_path):
        pair = (entry["train_id"], entry["test_id"])
        if pair in seen or pair[0] not in train or pair[1] not in test:
            raise ValueError(
                "Similarity pairs must be unique and match the supplied partitions"
            )
        score = float(entry["similarity"])
        if not math.isfinite(score) or not score < threshold:
            raise ValueError(
                "A formulation pair violates the declared similarity threshold"
            )
        seen.add(pair)
    if len(seen) != len(train) * len(test):
        raise ValueError("Similarity scores must cover every training and test pair")
    return result
