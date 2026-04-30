from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig, TestConfig
from data.synthetic import generate_synthetic_dataset
from experiments.configuration import save_standardized_outputs
from methods.permutation import run_cross_validation_method
from methods.trainers import resolve_device


class TestCrossValidationSchema(unittest.TestCase):
    def test_cross_validation_config_is_valid(self) -> None:
        config = TestConfig(
            method="cross_validation",
            metric="mse",
            n_perms=3,
            train_fraction=0.75,
        )
        self.assertIs(config.validate(), config)

    def test_cross_validation_rejects_invalid_train_fraction(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="cross_validation", train_fraction=0.0).validate()
        with self.assertRaises(ValueError):
            TestConfig(method="cross_validation", train_fraction=1.0).validate()

    def test_cross_validation_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="cross_validation", metric="spearman_corr_mean").validate()


class TestCrossValidationMethod(unittest.TestCase):
    def setUp(self) -> None:
        s = np.asarray(
            [
                [0.0, 0.0],
                [0.25, 0.25],
                [0.5, 0.5],
                [0.75, 0.75],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        a = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        self.dataset = DatasetBundle(S=s, A=a).validate()
        self.config = TestConfig(
            method="cross_validation",
            metric="mse",
            n_perms=2,
            train_fraction=0.6,
            n_reruns=1,
            epochs=2,
            patience=2,
            verbose=False,
            device="cpu",
            seed=11,
        )

    def test_cross_validation_uses_shared_deterministic_holdout_loss(self) -> None:
        recorded_masks: list[np.ndarray] = []

        def _mock_train_parallel_isodepth_model(
            S,
            A,
            config,
            *,
            device=None,
            s_batched=None,
            latent_dim=1,
            model_label=None,
            a_batched=None,
            loss_mask_batched=None,
        ):
            assert s_batched is not None
            assert loss_mask_batched is not None
            s_batched_np = np.asarray(s_batched, dtype=np.float32)
            train_mask = np.asarray(loss_mask_batched, dtype=np.float32)
            recorded_masks.append(train_mask.copy())
            test_mask = 1.0 - train_mask
            n_models, n_cells, _ = s_batched_np.shape
            predictions = np.repeat(np.asarray(A, dtype=np.float32)[None, :, :], n_models, axis=0)

            predictions[0, train_mask[0, :, 0] > 0, 0] += 100.0
            predictions[1, test_mask[1, :, 0] > 0, 0] += 1.0
            predictions[2, test_mask[2, :, 0] > 0, 0] += 2.0

            class _MockModel:
                def __init__(self) -> None:
                    self.latent_dim = int(latent_dim)
                    self.training_metadata = {
                        "n_reruns": int(config.n_reruns),
                        "selection_loss": "training_reconstruction_loss",
                        "best_train_loss_per_model": np.zeros(n_models, dtype=np.float64),
                        "best_rerun_index_per_model": np.zeros(n_models, dtype=np.int64),
                        "train_loss_per_rerun": np.zeros((n_models, int(config.n_reruns)), dtype=np.float64),
                    }

                def encoder(self, s_t: torch.Tensor) -> torch.Tensor:
                    return s_t[:, :, :1]

            return _MockModel(), predictions

        with patch(
            "methods.permutation.train_parallel_isodepth_model",
            side_effect=_mock_train_parallel_isodepth_model,
        ):
            result_a = run_cross_validation_method(self.dataset, self.config, device=resolve_device("cpu"))
            result_b = run_cross_validation_method(self.dataset, self.config, device=resolve_device("cpu"))

        self.assertEqual(result_a.method_name, "cross_validation")
        self.assertAlmostEqual(result_a.stat_true, 0.0)
        np.testing.assert_allclose(result_a.stat_perm, np.asarray([1.0, 4.0], dtype=np.float64))
        self.assertAlmostEqual(result_a.p_value, 1.0 / 3.0)
        np.testing.assert_array_equal(result_a.artifacts["train_mask"] + result_a.artifacts["test_mask"], 1.0)
        np.testing.assert_array_equal(result_a.artifacts["train_mask"], result_b.artifacts["train_mask"])
        np.testing.assert_array_equal(result_a.artifacts["test_mask"], result_b.artifacts["test_mask"])
        self.assertEqual(result_a.artifacts["train_size"], 3)
        self.assertEqual(result_a.artifacts["test_size"], 2)
        self.assertEqual(len(recorded_masks), 2)
        self.assertEqual(recorded_masks[0].shape, (self.config.n_perms + 1, self.dataset.n_cells, 1))
        np.testing.assert_array_equal(recorded_masks[0][0], recorded_masks[0][1])
        np.testing.assert_array_equal(recorded_masks[0][0], recorded_masks[0][2])

    def test_cross_validation_rejects_empty_holdout_after_rounding(self) -> None:
        tiny_dataset = DatasetBundle(
            S=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            A=np.asarray([[0.0], [1.0]], dtype=np.float32),
        ).validate()
        config = TestConfig(
            method="cross_validation",
            metric="mse",
            train_fraction=0.99,
            n_perms=1,
            epochs=1,
            patience=1,
            verbose=False,
            device="cpu",
        )
        with self.assertRaises(ValueError):
            run_cross_validation_method(tiny_dataset, config, device=resolve_device("cpu"))


class TestCrossValidationOutputs(unittest.TestCase):
    def test_standardized_outputs_include_cross_validation_fields(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="noise", n_cells=16, n_genes=3, sigma=0.0, seed=5)
        )
        result = run_cross_validation_method(
            dataset,
            TestConfig(
                method="cross_validation",
                metric="mse",
                n_perms=2,
                train_fraction=0.75,
                n_reruns=1,
                epochs=5,
                patience=2,
                verbose=False,
                device="cpu",
            ),
            device=resolve_device("cpu"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                data=DataConfig(source="synthetic", mode="noise", n_cells=16, n_genes=3, sigma=0.0, seed=5),
                test=TestConfig(
                    method="cross_validation",
                    metric="mse",
                    n_perms=2,
                    train_fraction=0.75,
                    n_reruns=1,
                    epochs=5,
                    patience=2,
                    verbose=False,
                    device="cpu",
                ),
                output=OutputConfig(out_dir=tmpdir, run_name="cross_validation_test"),
            ).validate()
            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            self.assertEqual(payload["method_name"], "cross_validation")
            self.assertEqual(payload["config"]["test"]["train_fraction"], 0.75)
            self.assertIn("train_mask", payload["artifacts"])
            self.assertIn("test_mask", payload["artifacts"])
            self.assertIn("null_summary", payload["artifacts"])
            self.assertTrue((result_path.parent / "cross_validation_test_metric_distribution.png").exists())
            self.assertTrue((result_path.parent / "cross_validation_test_isodepth.png").exists())


if __name__ == "__main__":
    unittest.main()
