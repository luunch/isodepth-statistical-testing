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
from methods.permutation import (
    _aggregate_weighted_fold_losses,
    _build_kfold_assignments,
    run_cross_validation_method,
)
from methods.trainers import resolve_device


class TestCrossValidationSchema(unittest.TestCase):
    def test_cross_validation_config_is_valid(self) -> None:
        config = TestConfig(
            method="cross_validation",
            metric="mse",
            n_perms=3,
            n_folds=3,
        )
        self.assertIs(config.validate(), config)

    def test_cross_validation_accepts_poisson_metric(self) -> None:
        config = TestConfig(
            method="cross_validation",
            metric="nll_poisson_mse",
            n_folds=3,
        )
        self.assertIs(config.validate(), config)

    def test_cross_validation_rejects_invalid_n_folds(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="cross_validation", n_folds=1).validate()

    def test_cross_validation_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="cross_validation", metric="spearman_corr_mean").validate()


class TestKFoldHelpers(unittest.TestCase):
    def test_build_kfold_assignments_is_deterministic(self) -> None:
        fold_a, sizes_a = _build_kfold_assignments(10, 3, seed=7)
        fold_b, sizes_b = _build_kfold_assignments(10, 3, seed=7)
        np.testing.assert_array_equal(fold_a, fold_b)
        np.testing.assert_array_equal(sizes_a, sizes_b)
        self.assertEqual(int(sizes_a.sum()), 10)

    def test_weighted_aggregation(self) -> None:
        fold_true = [0.0, 4.0]
        fold_perm = [
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([3.0, 8.0], dtype=np.float64),
        ]
        weights = np.asarray([0.3, 0.7], dtype=np.float64)
        stat_true, stat_perm = _aggregate_weighted_fold_losses(fold_true, fold_perm, weights)
        self.assertAlmostEqual(stat_true, 2.8)
        np.testing.assert_allclose(stat_perm, np.asarray([2.4, 6.2]))


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
            n_folds=2,
            n_reruns=1,
            epochs=2,
            patience=2,
            verbose=False,
            device="cpu",
            seed=11,
        )

    def test_cross_validation_runs_k_folds_and_aggregates(self) -> None:
        recorded_masks: list[np.ndarray] = []
        fold_calls = {"count": 0}

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
            metric_loss_mask_batched=None,
            **kwargs,
        ):
            assert s_batched is not None
            assert loss_mask_batched is not None
            assert metric_loss_mask_batched is not None
            s_batched_np = np.asarray(s_batched, dtype=np.float32)
            train_mask = np.asarray(loss_mask_batched, dtype=np.float32)
            recorded_masks.append(train_mask.copy())
            test_mask = np.asarray(metric_loss_mask_batched, dtype=np.float32)
            n_models, n_cells, _ = s_batched_np.shape
            predictions = np.repeat(np.asarray(A, dtype=np.float32)[None, :, :], n_models, axis=0)

            fold_idx = fold_calls["count"]
            fold_calls["count"] += 1
            predictions[0, test_mask[0, :, 0] > 0, 0] += float(fold_idx)
            predictions[1, test_mask[1, :, 0] > 0, 0] += float(fold_idx) + 1.0
            predictions[2, test_mask[2, :, 0] > 0, 0] += float(fold_idx) + 2.0

            from methods.subsampling import compute_masked_losses
            from methods.trainers import BatchedTrainingOutputs

            held_out_losses = compute_masked_losses(
                predictions,
                np.repeat(np.asarray(A, dtype=np.float32)[None, :, :], n_models, axis=0),
                test_mask,
                metric=config.metric,
            )

            class _MockModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.latent_dim = int(latent_dim)
                    self.M = n_models
                    self._dummy = torch.nn.Parameter(torch.zeros(1))
                    self.training_metadata = {
                        "n_reruns": int(config.n_reruns),
                        "selection_loss": "training_reconstruction_loss",
                        "best_train_loss_per_model": np.zeros(n_models, dtype=np.float64),
                        "best_rerun_index_per_model": np.zeros(n_models, dtype=np.int64),
                        "train_loss_per_rerun": np.zeros((n_models, int(config.n_reruns)), dtype=np.float64),
                    }

                def encoder(self, s_t: torch.Tensor) -> torch.Tensor:
                    return s_t[:, :, :1]

            stat_perm = held_out_losses[1:]
            best_null_index = int(np.argmin(stat_perm))
            worst_null_index = int(np.argmax(stat_perm))
            outputs = BatchedTrainingOutputs(
                model_metrics=held_out_losses,
                pred_true=np.asarray(predictions[0], dtype=np.float32),
                pred_best_null=np.asarray(predictions[best_null_index + 1], dtype=np.float32),
                pred_worst_null=np.asarray(predictions[worst_null_index + 1], dtype=np.float32),
                best_null_index=best_null_index,
                worst_null_index=worst_null_index,
            )
            return _MockModel(), outputs, s_batched_np

        with patch(
            "methods.permutation.train_parallel_isodepth_model",
            side_effect=_mock_train_parallel_isodepth_model,
        ):
            result_a = run_cross_validation_method(self.dataset, self.config, device=resolve_device("cpu"))
            self.assertEqual(fold_calls["count"], self.config.n_folds)
            fold_calls["count"] = 0
            result_b = run_cross_validation_method(self.dataset, self.config, device=resolve_device("cpu"))
            self.assertEqual(fold_calls["count"], self.config.n_folds)

        self.assertEqual(result_a.method_name, "cross_validation")
        self.assertEqual(result_a.artifacts["n_folds"], self.config.n_folds)
        np.testing.assert_array_equal(
            result_a.artifacts["train_mask"] + result_a.artifacts["test_mask"],
            1.0,
        )
        np.testing.assert_array_equal(result_a.artifacts["train_mask"], result_b.artifacts["train_mask"])
        np.testing.assert_array_equal(result_a.artifacts["test_mask"], result_b.artifacts["test_mask"])
        self.assertEqual(len(recorded_masks), self.config.n_folds * 2)  # two deterministic runs
        np.testing.assert_array_equal(recorded_masks[0][0], recorded_masks[0][1])
        np.testing.assert_array_equal(recorded_masks[0][0], recorded_masks[0][2])

    def test_cross_validation_rejects_too_many_folds(self) -> None:
        tiny_dataset = DatasetBundle(
            S=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            A=np.asarray([[0.0], [1.0]], dtype=np.float32),
        ).validate()
        config = TestConfig(
            method="cross_validation",
            metric="mse",
            n_folds=3,
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
                n_folds=2,
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
                    n_folds=2,
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
            self.assertEqual(payload["config"]["test"]["n_folds"], 2)
            self.assertIn("train_mask", payload["artifacts"])
            self.assertIn("test_mask", payload["artifacts"])
            self.assertIn("per_fold_true_loss", payload["artifacts"])
            self.assertIn("per_fold_p_values", payload["artifacts"])
            self.assertIn("per_fold_true_isodepth", payload["artifacts"])
            self.assertTrue((result_path.parent / "cross_validation_test_metric_distribution.png").exists())
            self.assertTrue((result_path.parent / "cross_validation_test_isodepth.png").exists())
            self.assertTrue((result_path.parent / "cross_validation_test_cv_fold_isodepths.png").exists())
            self.assertTrue(
                (result_path.parent / "cross_validation_test_cv_per_fold_metric_distributions.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
