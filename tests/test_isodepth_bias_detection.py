from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.isodepth_bias_detection import (
    load_isodepth_bias_detection_spec,
    run_isodepth_bias_detection,
)


class TestIsodepthBiasDetection(unittest.TestCase):
    def test_runner_writes_outputs_single_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "bias_spec.json"
            base_config_payload = {
                "data": {
                    "source": "synthetic",
                    "mode": "radial",
                    "n_cells": 12,
                    "n_genes": 4,
                    "sigma": 0.05,
                    "seed": 11,
                },
                "test": {
                    "method": "parallel_permutation",
                    "metric": "mse",
                    "n_perms": 5,
                    "n_reruns": 3,
                    "epochs": 3,
                    "patience": 3,
                    "lr": 1e-3,
                    "seed": 13,
                    "device": "cpu",
                    "decoder": "linear",
                    "sgd_batch_size": 0,
                    "verbose": False,
                },
                "output": {
                    "out_dir": str(tmp_path / "results"),
                    "run_name": "base_run",
                },
            }
            spec_payload = {
                "experiment_name": "isodepth_bias_detection_unit",
                "base_config": str(base_config_path),
                "output_root": str(tmp_path / "bias_outputs"),
                "n_perms": 2,
                "epochs": 4,
                "devices": ["cpu"],
            }
            base_config_path.write_text(json.dumps(base_config_payload), encoding="utf-8")
            spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")

            spec = load_isodepth_bias_detection_spec(spec_path)
            payload = run_isodepth_bias_detection(spec)

            self.assertEqual(payload["epochs_override"], 4)
            self.assertEqual(payload["effective_run_config"]["test"]["epochs"], 4)
            self.assertEqual(payload["effective_run_config"]["test"]["n_reruns"], 1)
            self.assertEqual(payload["permutation_seed"], 13)
            self.assertEqual(payload["n_perms"], 2)
            self.assertEqual(payload["n_models"], 3)
            self.assertEqual(payload["devices"], ["cpu"])
            self.assertIsNone(payload["cross_device_slots"])

            for key in ["bias_detection_plot", "bias_detection_result_path"]:
                self.assertTrue(Path(payload[key]).exists(), key)

            saved_payload = json.loads(Path(payload["bias_detection_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["epochs_override"], 4)
            self.assertIn("summary", saved_payload)
            self.assertIn("per_device", saved_payload)
            self.assertIn("cpu", saved_payload["per_device"])

    def test_legacy_device_field_is_single_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "bias_spec.json"
            base_config_payload = {
                "data": {
                    "source": "synthetic",
                    "mode": "radial",
                    "n_cells": 8,
                    "n_genes": 3,
                    "sigma": 0.05,
                    "seed": 2,
                },
                "test": {
                    "method": "parallel_permutation",
                    "metric": "mse",
                    "n_perms": 1,
                    "n_reruns": 1,
                    "epochs": 2,
                    "patience": 3,
                    "lr": 1e-3,
                    "seed": 7,
                    "device": "cpu",
                    "decoder": "linear",
                    "sgd_batch_size": 0,
                    "verbose": False,
                },
                "output": {"out_dir": str(tmp_path / "results"), "run_name": "base_run"},
            }
            spec_payload = {
                "experiment_name": "legacy_device_key",
                "base_config": str(base_config_path),
                "output_root": str(tmp_path / "out"),
                "n_perms": 1,
                "epochs": 2,
                "device": "cpu",
            }
            base_config_path.write_text(json.dumps(base_config_payload), encoding="utf-8")
            spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")

            spec = load_isodepth_bias_detection_spec(spec_path)
            self.assertEqual(spec.devices, ["cpu"])


if __name__ == "__main__":
    unittest.main()
