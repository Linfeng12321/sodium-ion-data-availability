"""Synthetic unit fixtures only. No manuscript data or training experiments."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from sodium_materials.preprocessing import (
    ROLES,
    RolePreprocessor,
    validate_schema,
    targets,
    milestone_from_anchor,
    fit_capacity_scaler,
)
from sodium_materials.model import MaterialSystemModel, make_optimizer
from sodium_materials.training import (
    EarlyStopping,
    masked_loss,
    snapshot,
    validation_loss,
)
from sodium_materials.metrics import evaluate, summarize_runs
from sodium_materials.partitions import validate_partitions
from sodium_materials.__main__ import validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))


def schema():
    return {
        "record_id": "id",
        "capacity_target": "capacity",
        "rmi_target": "rmi",
        "roles": {
            r: {"categorical": [r + "_category"], "numerical": [r + "_value"]}
            for r in ROLES
        },
        "missing_markers": ["", "unknown"],
        "numerical_statistics": "after_imputation",
        "capacity_standard_deviation_ddof": 0,
    }


def rows():
    result = []
    for i in range(4):
        row = {
            "id": str(i),
            "capacity": str(1 + i),
            "rmi": str(i),
            "family": "A" if i < 3 else "B",
        }
        for role in ROLES:
            row.update(
                {
                    role + "_category": "Na" if i == 0 else "host",
                    role + "_value": str(i),
                }
            )
        result.append(row)
    return result


class InputTests(unittest.TestCase):
    def test_template_requires_explicit_fields(self):
        with self.assertRaises(ValueError):
            validate_schema(
                json.loads((ROOT / "configs/input_schema.template.json").read_text())
            )

    def test_duplicate_fields_and_targets_rejected(self):
        s = schema()
        s["roles"]["anode"]["numerical"] = ["capacity"]
        with self.assertRaises(ValueError):
            validate_schema(s)
        s = schema()
        s["roles"]["anode"]["numerical"] = ["cathode_value"]
        with self.assertRaises(ValueError):
            validate_schema(s)

    def test_sodium_symbol_retained(self):
        pre = RolePreprocessor(schema()).fit(rows()[:2])
        self.assertIn("Na", pre.state["anode"]["categorical"]["anode_category"])
        bad = schema()
        bad["missing_markers"].append("na")
        with self.assertRaises(ValueError):
            validate_schema(bad)

    def test_unseen_and_missing_categories(self):
        pre = RolePreprocessor(schema()).fit(rows()[:2])
        case = rows()[2]
        case["anode_category"] = "new"
        case["cathode_category"] = ""
        encoded = pre.transform([case])
        self.assertEqual(float(encoded["anode"][0, :3].sum()), 0)
        self.assertEqual(float(encoded["cathode"][0, 0]), 1)

    def test_statistics_order_and_missing_indicator(self):
        r = rows()[:3]
        r[2]["anode_value"] = ""
        s = schema()
        after = RolePreprocessor(s).fit(r)
        s2 = copy.deepcopy(s)
        s2["numerical_statistics"] = "before_imputation"
        before = RolePreprocessor(s2).fit(r)
        self.assertNotEqual(
            after.state["anode"]["numerical"]["anode_value"]["scale"],
            before.state["anode"]["numerical"]["anode_value"]["scale"],
        )
        self.assertEqual(float(after.transform(r)["anode"][2, -1]), 1)

    def test_all_missing_and_constant_fields(self):
        r = rows()
        for row in r:
            row["anode_value"] = ""
            row["cathode_value"] = "5"
        pre = RolePreprocessor(schema()).fit(r)
        encoded = pre.transform(r)
        self.assertTrue(np.all(encoded["anode"][:, -2] == 0))
        self.assertTrue(np.all(encoded["anode"][:, -1] == 1))
        self.assertTrue(np.all(encoded["cathode"][:, -2] == 0))

    def test_no_refit_on_test_values_and_json_roundtrip(self):
        pre = RolePreprocessor(schema()).fit(rows()[:2])
        original = copy.deepcopy(pre.export())
        r = rows()[2:]
        r[0]["anode_value"] = "10000"
        pre.transform(r)
        self.assertEqual(original, pre.export())
        restored = RolePreprocessor.restore(json.loads(json.dumps(pre.export())))
        for key in ROLES:
            np.testing.assert_array_equal(
                pre.transform(r)[key], restored.transform(r)[key]
            )

    def test_label_validity_and_tier_boundaries(self):
        self.assertEqual(
            [milestone_from_anchor(v, True) for v in [1, 9, 10, 99, 100, 999, 1000]],
            [0, 0, 1, 1, 2, 2, 3],
        )
        self.assertIsNone(milestone_from_anchor(1000, False))
        with self.assertRaises(ValueError):
            milestone_from_anchor(1000, "false")
        r = rows()
        r[0]["rmi"] = "1.5"
        with self.assertRaises(ValueError):
            targets(r, schema())

    def test_target_scaling(self):
        population = fit_capacity_scaler(np.array([1, 3, np.nan]), 0)
        sample = fit_capacity_scaler(np.array([1, 3, np.nan]), 1)
        self.assertEqual(population["scale"], 1)
        self.assertAlmostEqual(sample["scale"], 2**0.5)
        with self.assertRaises(ValueError):
            fit_capacity_scaler(np.array([1, 1]), 0)

    def test_fixed_partitions_and_similarity(self):
        r = rows()
        assignments = [
            {"record_id": str(i), "split": s}
            for i, s in enumerate(["train", "train", "validation", "test"])
        ]
        protocol = {
            "setting": "anode",
            "family_column": "family",
            "held_out_families": ["B"],
            "expected_counts": {"train": 2, "validation": 1, "test": 1},
            "similarity_definition": "Unit fixture scores only",
            "similarity_threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "scores.csv"
            p.write_text(
                "train_id,test_id,similarity\n0,3,0.1\n1,3,0.2\n", encoding="utf-8"
            )
            self.assertEqual(
                len(
                    validate_partitions(r, assignments, schema(), protocol, p)["train"]
                ),
                2,
            )
            p.write_text("train_id,test_id,similarity\n0,3,0.1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_partitions(r, assignments, schema(), protocol, p)
            p.write_text(
                "train_id,test_id,similarity\n0,3,0.5\n1,3,0.2\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_partitions(r, assignments, schema(), protocol, p)

    def test_field_list_type_and_nonfinite_input(self):
        s = schema()
        s["roles"]["anode"]["categorical"] = "anode_category"
        with self.assertRaises(ValueError):
            validate_schema(s)
        r = rows()
        r[0]["anode_value"] = "inf"
        with self.assertRaises(ValueError):
            RolePreprocessor(schema()).fit(r)

    def test_prediction_input_dimensions(self):
        model = MaterialSystemModel({r: 2 for r in ROLES}, CONFIG["model"])
        bad = {r: torch.zeros(2, 2) for r in ROLES}
        bad["anode"] = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            model(bad)

    def test_family_leakage_rejected(self):
        r = rows()
        r[-1]["family"] = "A"
        a = [
            {"record_id": str(i), "split": s}
            for i, s in enumerate(["train", "train", "validation", "test"])
        ]
        p = {
            "setting": "anode",
            "family_column": "family",
            "held_out_families": ["A"],
            "expected_counts": {"train": 2, "validation": 1, "test": 1},
        }
        with self.assertRaisesRegex(ValueError, "must not enter"):
            validate_partitions(r, a, schema(), p, "not_read.csv")


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = MaterialSystemModel({r: 3 for r in ROLES}, CONFIG["model"])
        self.x = {r: torch.ones(3, 3) for r in ROLES}

    def test_shapes_and_evaluation_dropout(self):
        self.model.eval()
        cap, cls = self.model(self.x)
        self.assertEqual(tuple(cap.shape), (3,))
        self.assertEqual(tuple(cls.shape), (3, 4))
        torch.testing.assert_close(cap, self.model(self.x)[0], rtol=0, atol=0)
        self.assertEqual(self.model.attention.num_heads, 4)

    def test_gradient_connectivity_without_parameter_update(self):
        before = snapshot(self.model)
        cap, cls = self.model(self.x)
        loss = masked_loss(
            cap,
            cls,
            torch.tensor([0.0, float("nan"), 1.0]),
            torch.tensor([float("nan"), 1.0, 3.0]),
            CONFIG["training"],
        )
        loss.backward()
        for r in ROLES:
            grad = self.model.encoders[r][0].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_weight_decay_groups(self):
        optimizer = make_optimizer(self.model, CONFIG["training"])
        grouped = {
            id(p): group["weight_decay"]
            for group in optimizer.param_groups
            for p in group["params"]
        }
        self.assertEqual(len(grouped), len(list(self.model.parameters())))
        for name, p in self.model.named_parameters():
            expected = 1e-4 if name.endswith("weight") and p.ndim == 2 else 0
            self.assertEqual(grouped[id(p)], expected)

    def test_masks_and_no_labels(self):
        cap = torch.tensor([2.0, 900.0], requires_grad=True)
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = masked_loss(
            cap,
            logits,
            torch.tensor([1.0, float("nan")]),
            torch.tensor([float("nan"), 0.0]),
            CONFIG["training"],
        )
        self.assertAlmostEqual(float(loss.detach()), 1 + np.log(4), places=6)
        absent = torch.full((2,), float("nan"))
        self.assertIsNone(masked_loss(cap, logits, absent, absent, CONFIG["training"]))

    def test_snapshot_is_independent(self):
        saved = snapshot(self.model)
        first = next(self.model.parameters())
        key = next(iter(saved))
        old = saved[key].clone()
        with torch.no_grad():
            first.add_(1)
        torch.testing.assert_close(saved[key], old)

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "random_test_weights.pt"
            torch.save(snapshot(self.model), p)
            restored = MaterialSystemModel({r: 3 for r in ROLES}, CONFIG["model"])
            restored.load_state_dict(
                torch.load(p, weights_only=True, map_location="cpu")
            )
            self.model.eval()
            restored.eval()
            torch.testing.assert_close(self.model(self.x)[0], restored(self.x)[0])

    def test_validation_is_global_task_mean(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                n = len(x["anode"])
                return torch.zeros(n), torch.zeros(n, 4)

        cfg = dict(CONFIG["training"], batch_size=2)
        value = validation_loss(
            Constant(),
            (
                self.x,
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([0.0, float("nan"), float("nan")]),
            ),
            cfg,
        )
        self.assertAlmostEqual(value, 14 / 3 + np.log(4), places=5)


class LogicTests(unittest.TestCase):
    def test_stopping_and_checkpoint_are_separate(self):
        tracker = EarlyStopping(2, 1e-4)
        self.assertEqual(tracker.update(1), (True, False))
        self.assertEqual(tracker.update(1 - 5e-5), (True, False))
        self.assertEqual(tracker.update(1 - 5e-5), (False, True))
        self.assertEqual(tracker.reference, 1)
        with self.assertRaises(ValueError):
            tracker.update(float("nan"))

    def test_metrics(self):
        result = evaluate([1, 3], [2, 2], [0, 1], [0, 1])
        self.assertEqual(result["mae"], 1)
        self.assertEqual(result["nmae_percent"], 50)
        self.assertEqual(result["r_squared"], 0)
        self.assertEqual(result["macro_f1"], 0.5)

    def test_undefined_metrics_and_seed_summary(self):
        result = evaluate([1, 1], [1, 1], [np.nan, np.nan], [0, 0])
        self.assertIsNone(result["r_squared"])
        self.assertIsNone(result["nmae_percent"])
        self.assertIsNone(result["macro_f1"])
        a = dict(result, mae=1)
        b = dict(result, mae=3)
        self.assertAlmostEqual(
            summarize_runs([a, b])["mae"]["sample_standard_deviation"], 2**0.5
        )
        with self.assertRaises(ValueError):
            evaluate([float("inf")], [1], [0], [0])

    def test_configuration(self):
        validate_config(CONFIG)
        config = copy.deepcopy(CONFIG)
        config["training"]["batch_size"] = 0
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
