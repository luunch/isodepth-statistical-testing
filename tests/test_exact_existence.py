from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig, TestConfig
from data.synthetic import generate_synthetic_dataset
from experiments.configuration import save_standardized_outputs
from methods.architectures import ParallelLinear
from methods.permutation import run_exact_existence_method
from methods.trainers import get_training_metadata, resolve_device, train_batched_isodepth_model, train_isodepth_model


class TestExactExistenceSchema(unittest.TestCase):
    def test_exact_existence_config_is_valid(self) -> None:
        config = TestConfig(
            method="exact_existence",
            metric="mse",
            n_perms=3,
            max_spatial_dims=2,
            alpha=0.05,
        )
        self.assertIs(config.validate(), config)

    def test_exact_existence_rejects_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="exact_existence", alpha=0.0).validate()

    def test_exact_existence_rejects_invalid_max_spatial_dims(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="exact_existence", max_spatial_dims=0).validate()

    def test_exact_existence_rejects_correlation_metric(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="exact_existence", metric="spearman_corr_mean").validate()


class TestLatentDimensionTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="radial", n_cells=16, n_genes=3, sigma=0.0, seed=7)
        )
        self.config = TestConfig(
            method="parallel_permutation",
            metric="mse",
            n_perms=2,
            n_reruns=1,
            epochs=5,
            patience=2,
            verbose=False,
        )

    def test_train_isodepth_model_supports_multiple_latent_dims(self) -> None:
        device = resolve_device("cpu")
        for latent_dim in (1, 2, 3):
            model, predictions = train_isodepth_model(
                self.dataset.S,
                self.dataset.A,
                self.config,
                device=device,
                latent_dim=latent_dim,
            )
            self.assertEqual(predictions.shape, self.dataset.A.shape)
            self.assertEqual(int(getattr(model, "latent_dim")), latent_dim)
            self.assertIsInstance(model.decoder, nn.Linear)
            self.assertEqual(model.decoder.in_features, latent_dim)
            self.assertEqual(model.decoder.out_features, self.dataset.n_genes)

    def test_train_batched_isodepth_model_supports_multiple_latent_dims(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        device = resolve_device("cpu")
        for latent_dim in (1, 2, 3):
            model, predictions = train_batched_isodepth_model(
                s_batched,
                self.dataset.A,
                self.config,
                device=device,
                latent_dim=latent_dim,
            )
            self.assertEqual(predictions.shape, (2, self.dataset.n_cells, self.dataset.n_genes))
            self.assertIsInstance(model.decoder, ParallelLinear)
            self.assertEqual(model.decoder.weight.shape, (2, self.dataset.n_genes, latent_dim))
            with torch.no_grad():
                encoded = model.encoder(torch.tensor(s_batched, dtype=torch.float32, device=device))
            self.assertEqual(tuple(encoded.shape), (2, self.dataset.n_cells, latent_dim))

    def test_trainers_reject_zero_latent_dim(self) -> None:
        device = resolve_device("cpu")
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)

        with self.assertRaises(ValueError):
            train_isodepth_model(
                self.dataset.S,
                self.dataset.A,
                self.config,
                device=device,
                latent_dim=0,
            )

        with self.assertRaises(ValueError):
            train_batched_isodepth_model(
                s_batched,
                self.dataset.A,
                self.config,
                device=device,
                latent_dim=0,
            )

    def test_batched_training_is_deterministic_for_fixed_seed(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        device = resolve_device("cpu")
        _, predictions_a = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            self.config,
            device=device,
            latent_dim=2,
        )
        _, predictions_b = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            self.config,
            device=device,
            latent_dim=2,
        )
        np.testing.assert_allclose(predictions_a, predictions_b, atol=1e-7)

    def test_zero_sgd_batch_size_matches_default_training(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        device = resolve_device("cpu")
        _, predictions_default = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            self.config,
            device=device,
            latent_dim=2,
        )
        config_zero_sgd = TestConfig(**{**self.config.__dict__, "sgd_batch_size": 0})
        _, predictions_zero_sgd = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            config_zero_sgd,
            device=device,
            latent_dim=2,
        )
        np.testing.assert_allclose(predictions_default, predictions_zero_sgd, atol=1e-7)

    def test_batched_training_supports_sgd_minibatches(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        device = resolve_device("cpu")
        minibatch_config = TestConfig(**{**self.config.__dict__, "sgd_batch_size": 5})
        model_a, predictions_a = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            minibatch_config,
            device=device,
            latent_dim=2,
        )
        model_b, predictions_b = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            minibatch_config,
            device=device,
            latent_dim=2,
        )
        self.assertEqual(predictions_a.shape, (2, self.dataset.n_cells, self.dataset.n_genes))
        np.testing.assert_allclose(predictions_a, predictions_b, atol=1e-7)
        self.assertEqual(int(getattr(model_a, "latent_dim")), 2)
        self.assertEqual(int(getattr(model_b, "latent_dim")), 2)

    def test_batched_trainer_expands_parallel_slots_for_reruns(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        config = TestConfig(
            method="parallel_permutation",
            metric="mse",
            n_perms=2,
            n_reruns=3,
            epochs=2,
            patience=2,
            verbose=False,
            device="cpu",
        )
        recorded_slot_counts: list[int] = []

        from methods.trainers import isodepth as trainer_module

        original_snapshot = trainer_module._snapshot_parallel_model_state

        def _record_snapshot(model, n_models):
            recorded_slot_counts.append(int(n_models))
            return original_snapshot(model, n_models)

        with patch("methods.trainers.isodepth._snapshot_parallel_model_state", side_effect=_record_snapshot):
            model, predictions = train_batched_isodepth_model(
                s_batched,
                self.dataset.A,
                config,
                device=resolve_device("cpu"),
                latent_dim=2,
            )

        metadata = get_training_metadata(model)
        self.assertEqual(recorded_slot_counts[0], 6)
        self.assertEqual(predictions.shape, (2, self.dataset.n_cells, self.dataset.n_genes))
        self.assertEqual(metadata["train_loss_per_rerun"].shape, (2, 3))

    def test_n_reruns_one_returns_zero_winner_indices(self) -> None:
        s_batched = np.stack([self.dataset.S, self.dataset.S[::-1]], axis=0)
        model, _ = train_batched_isodepth_model(
            s_batched,
            self.dataset.A,
            self.config,
            device=resolve_device("cpu"),
            latent_dim=2,
        )
        metadata = get_training_metadata(model)
        np.testing.assert_array_equal(metadata["best_rerun_index_per_model"], np.zeros(2, dtype=np.int64))


class TestExactExistenceMethod(unittest.TestCase):
    def setUp(self) -> None:
        x = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        y = np.linspace(1.0, 0.0, 9, dtype=np.float32)
        S = np.stack([x, y], axis=1)
        A = np.stack([x, y, x + y], axis=1).astype(np.float32)
        self.dataset = DatasetBundle(S=S, A=A, meta={"source": "synthetic", "mode": "custom"}).validate()
        self.config = TestConfig(
            method="exact_existence",
            metric="mse",
            n_perms=20,
            n_reruns=1,
            max_spatial_dims=3,
            alpha=0.05,
            epochs=5,
            patience=2,
            verbose=False,
            device="cpu",
        )

    @staticmethod
    def _mock_train_parallel_isodepth_model(S, A, config, *, device=None, s_batched=None, latent_dim=1, model_label=None, a_batched=None, loss_mask_batched=None):
        if s_batched is None:
            s_batched_np = np.repeat(np.asarray(S, dtype=np.float32)[None, :, :], config.n_perms + 1, axis=0)
        else:
            s_batched_np = np.asarray(s_batched, dtype=np.float32)
        x = s_batched_np[:, :, 0]
        y = s_batched_np[:, :, 1]
        n_models = s_batched_np.shape[0]
        n_cells = s_batched_np.shape[1]
        baseline = np.broadcast_to(A.mean(axis=0, keepdims=True), (n_cells, A.shape[1])).astype(np.float32)
        preds = np.repeat(baseline[None, :, :], n_models, axis=0)

        if latent_dim == 1:
            preds[0, :, 0] = A[:, 0]
        elif latent_dim == 2:
            preds[0] = A
        elif latent_dim >= 3:
            preds[:] = np.repeat(A[None, :, :], n_models, axis=0)

        class _MockModel:
            def __init__(self, dim: int):
                self.latent_dim = dim
                self.training_metadata = {
                    "n_reruns": int(config.n_reruns),
                    "selection_loss": "training_reconstruction_loss",
                    "best_train_loss_per_model": np.zeros(n_models, dtype=np.float64),
                    "best_rerun_index_per_model": np.zeros(n_models, dtype=np.int64),
                    "train_loss_per_rerun": np.zeros((n_models, int(config.n_reruns)), dtype=np.float64),
                }

            def encoder(self, s_t: torch.Tensor) -> torch.Tensor:
                if self.latent_dim == 1:
                    return s_t[:, :, :1]
                if self.latent_dim == 2:
                    return s_t[:, :, :2]
                extra = torch.zeros((s_t.shape[0], s_t.shape[1], self.latent_dim - 2), dtype=s_t.dtype, device=s_t.device)
                return torch.cat([s_t[:, :, :2], extra], dim=2)

        model = _MockModel(latent_dim)
        return model, preds

    def test_exact_existence_selects_two_dimensions_and_stops(self) -> None:
        with patch("methods.permutation.train_parallel_isodepth_model", side_effect=self._mock_train_parallel_isodepth_model):
            result = run_exact_existence_method(self.dataset, self.config, device=resolve_device("cpu"))

        self.assertEqual(result.method_name, "exact_existence")
        self.assertEqual(result.artifacts["selected_spatial_dims"], 2)
        self.assertEqual(result.artifacts["tested_spatial_dims"], [1, 2, 3])
        self.assertAlmostEqual(result.p_value, float(result.artifacts["step_summaries"]["3"]["p_value"]))
        self.assertEqual(len(result.artifacts["step_summaries"]["1"]["null_distribution"]), self.config.n_perms)
        self.assertFalse(result.artifacts["step_summaries"]["3"]["significant"])
        self.assertEqual(np.asarray(result.artifacts["step_summaries"]["2"]["true_isodepth"]).shape, (self.dataset.n_cells, 2))

    def test_exact_existence_can_stop_at_zero_dimensions(self) -> None:
        def no_gain(*args, **kwargs):
            model, preds = self._mock_train_parallel_isodepth_model(*args, **kwargs)
            latent_dim = int(kwargs.get("latent_dim", 1))
            if latent_dim >= 1:
                preds[:] = self.dataset.A.mean(axis=0, keepdims=True)
            return model, preds

        with patch("methods.permutation.train_parallel_isodepth_model", side_effect=no_gain):
            result = run_exact_existence_method(self.dataset, self.config, device=resolve_device("cpu"))

        self.assertEqual(result.artifacts["selected_spatial_dims"], 0)
        self.assertEqual(result.artifacts["tested_spatial_dims"], [1])


class TestExactExistenceOutputs(unittest.TestCase):
    def test_standardized_outputs_include_exact_existence_artifacts(self) -> None:
        dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="noise", n_cells=16, n_genes=3, sigma=0.0, seed=5)
        )
        result = run_exact_existence_method(
            dataset,
            TestConfig(
                method="exact_existence",
                metric="mse",
                n_perms=2,
                n_reruns=1,
                max_spatial_dims=2,
                alpha=0.05,
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
                    method="exact_existence",
                    metric="mse",
                    n_perms=2,
                    n_reruns=1,
                    max_spatial_dims=2,
                    alpha=0.05,
                    epochs=5,
                    patience=2,
                    verbose=False,
                    device="cpu",
                ),
                output=OutputConfig(out_dir=tmpdir, run_name="exact_existence_test"),
            ).validate()
            payload, result_path = save_standardized_outputs(dataset, result, run_config)

            self.assertEqual(payload["method_name"], "exact_existence")
            self.assertIn("selected_spatial_dims", payload["artifacts"])
            self.assertIn("step_summaries", payload["artifacts"])
            self.assertEqual(payload["config"]["test"]["n_reruns"], 1)
            self.assertIn("rerun_summary", payload["artifacts"])
            self.assertEqual(payload["config"]["test"]["max_spatial_dims"], 2)
            self.assertEqual(payload["config"]["test"]["alpha"], 0.05)
            self.assertTrue((result_path.parent / "exact_existence_test_metric_distribution.png").exists())
            self.assertTrue((result_path.parent / "exact_existence_test_isodepth.png").exists())


if __name__ == "__main__":
    unittest.main()
