"""Tests for post-hoc Gaussian OLS decoder refit on isodepth slots."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from data.schemas import TestConfig
from data.transforms import celltype_expression_residuals
from methods.architectures import ParallelIsoDepthWithFixedCovariateNet
from methods.metrics import compute_metric
from methods.trainers.isodepth import (
    _batched_parallel_ols_loss,
    _solve_ols_beta,
    closed_form_gaussian_decode_torch,
    closed_form_gaussian_mse_torch,
    extract_model_isodepth,
    fit_closed_form_decoder,
    gaussian_parametric_decoder_uses_ols,
    get_training_metadata,
    poisson_parametric_decoder_uses_irls,
    train_celltype_parallel_isodepth_model,
    train_parallel_isodepth_model,
)


class TestGaussianOlsRefit(unittest.TestCase):
    def test_fixed_covariate_minibatch_ols_uses_matching_rows(self) -> None:
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        cov = rng.normal(size=20).astype(np.float32)
        model = ParallelIsoDepthWithFixedCovariateNet(2, 3, cov, decoder_type="linear")
        coords = torch.randn(2, 20, 2)
        rows = torch.tensor([13, 2, 17, 4, 8, 0, 19, 6])
        targets = torch.tensor(rng.normal(size=(2, len(rows), 3)), dtype=torch.float32)
        loss = _batched_parallel_ols_loss(
            model, coords, batch_indices=rows, fixed_latent_t=None,
            targets=targets, decoder_type="linear", loss_mask=None,
        )
        spatial = model.encoder(coords[:, rows]).detach().numpy()
        cov_rows = model.covariate_values.detach().numpy()[rows]
        expected = []
        for slot in range(2):
            design = np.c_[np.ones(len(rows)), spatial[slot], cov_rows]
            y = targets[slot].numpy()
            pred = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            expected.append(np.mean((y - pred) ** 2))
        np.testing.assert_allclose(loss.detach().numpy(), expected, atol=2e-6)
        loss.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.encoder.parameters()))

    def test_fixed_covariate_ols_training_selection_and_slot_metrics(self) -> None:
        rng = np.random.default_rng(42)
        n_cells = 40
        coords = rng.normal(size=(n_cells, 2)).astype(np.float32)
        cov = rng.normal(size=(n_cells, 2)).astype(np.float32)
        groups = np.arange(n_cells) % 2
        targets = np.stack([
            cov @ rng.normal(size=(2, 3)) + 0.1 * rng.normal(size=(n_cells, 3))
            for _ in range(3)
        ]).astype(np.float32)
        coords_batched = np.repeat(coords[None], 3, axis=0)
        for grouped in (False, True):
            # Opposite nuisance slopes require distinct cell-type decoders.
            y_slots = targets * (1 - 2 * groups)[None, :, None] if grouped else targets
            y_slots = y_slots.astype(np.float32)
            expected_targets = np.stack([
                celltype_expression_residuals(y, groups, n_cell_types=2)
                for y in y_slots
            ]) if grouped else y_slots
            for decoder in ("linear", "quadratic"):
                for batch_size in (None, 16):
                    with self.subTest(grouped=grouped, decoder=decoder, batch_size=batch_size):
                        config = TestConfig(
                            metric="nll_gaussian_mse", decoder=decoder, n_perms=2,
                            n_reruns=2, epochs=3, lr=0.001, sgd_batch_size=batch_size,
                            device="cpu", verbose=False, seed=8,
                        )
                        train = train_celltype_parallel_isodepth_model if grouped else train_parallel_isodepth_model
                        kwargs = {"cell_type_labels": groups, "n_cell_types": 2} if grouped else {}
                        model, outputs, _ = train(
                            coords, y_slots[0], config, s_batched=coords_batched,
                            a_batched=y_slots, fixed_covariate_values=cov, **kwargs,
                        )
                        with torch.no_grad():
                            spatial = model.encoder(torch.tensor(coords_batched)).numpy()
                        cov_fit = model.covariate_values.numpy()
                        expected_metrics = []
                        for slot in range(3):
                            z = np.c_[spatial[slot], cov_fit].astype(np.float64)
                            features = np.c_[z, z ** 2] if decoder == "quadratic" else z
                            design = np.c_[np.ones(n_cells), features]
                            y = expected_targets[slot]
                            pred = np.empty_like(y)
                            fit_groups = groups if grouped else np.zeros(n_cells)
                            for group in np.unique(fit_groups):
                                member = fit_groups == group
                                x = design[member]
                                pred[member] = x @ np.linalg.lstsq(x, y[member], rcond=None)[0]
                            expected_metrics.append(np.mean((y - pred) ** 2))
                            if slot == 0:
                                np.testing.assert_allclose(outputs.pred_true, pred, atol=2e-5)
                        np.testing.assert_allclose(outputs.model_metrics, expected_metrics, atol=2e-6)
                        metadata = get_training_metadata(model)
                        np.testing.assert_allclose(
                            metadata["train_loss_per_rerun"].min(axis=1),
                            expected_metrics, atol=2e-5,
                        )
                        # Nuisance-only targets should fit immediately, including during training.
                        self.assertLess(float(metadata["loss_history"][0]), 0.03)

    def test_gaussian_parametric_decoder_uses_ols_predicate(self) -> None:
        gauss_linear = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        gauss_quad = TestConfig(
            metric="nll_gaussian_mse",
            decoder="quadratic",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        gauss_nn = TestConfig(
            metric="nll_gaussian_mse",
            decoder="nn",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        poisson_linear = TestConfig(
            metric="nll_poisson_mse",
            decoder="linear",
            n_perms=1,
            n_reruns=1,
            epochs=1,
        )
        self.assertTrue(gaussian_parametric_decoder_uses_ols(gauss_linear))
        self.assertTrue(gaussian_parametric_decoder_uses_ols(gauss_quad))
        self.assertFalse(gaussian_parametric_decoder_uses_ols(gauss_nn))
        self.assertFalse(gaussian_parametric_decoder_uses_ols(poisson_linear))
        self.assertTrue(poisson_parametric_decoder_uses_irls(poisson_linear))
        self.assertFalse(poisson_parametric_decoder_uses_irls(gauss_linear))

    def test_predict_latent_applies_fit_coefficients_to_query_cells(self) -> None:
        rng = np.random.default_rng(0)
        n_cells, n_genes = 40, 6
        z = rng.normal(size=n_cells).astype(np.float64)
        A = (0.8 * z[:, None] + rng.normal(scale=0.1, size=(n_cells, n_genes))).astype(np.float64)
        train_z = z[:20]
        train_A = A[:20]
        direct = fit_closed_form_decoder(train_z, train_A, "linear")
        via_predict = fit_closed_form_decoder(
            train_z, train_A, "linear", predict_latent=train_z
        )
        np.testing.assert_allclose(direct, via_predict, rtol=1e-5, atol=1e-5)
        all_preds = fit_closed_form_decoder(
            train_z, train_A, "linear", predict_latent=z
        )
        self.assertEqual(all_preds.shape, A.shape)
        np.testing.assert_allclose(direct, all_preds[:20], rtol=1e-5, atol=1e-5)

    def test_2d_latent_linear_matches_column_stack_lstsq(self) -> None:
        rng = np.random.default_rng(1)
        n_cells, n_genes = 30, 4
        z = rng.normal(size=(n_cells, 2))
        A = z @ rng.normal(size=(2, n_genes)) + rng.normal(scale=0.05, size=(n_cells, n_genes))
        pred = fit_closed_form_decoder(z, A, "linear")
        design = np.column_stack([z, np.ones(n_cells)])
        beta, *_ = np.linalg.lstsq(design, A, rcond=None)
        np.testing.assert_allclose(pred, design @ beta, rtol=1e-5, atol=1e-5)

    def test_parallel_gaussian_linear_uses_ols_pred_true(self) -> None:
        rng = np.random.default_rng(2)
        n_cells, n_genes = 48, 8
        S = rng.uniform(size=(n_cells, 2)).astype(np.float32)
        axis = (S[:, 0] - S[:, 0].mean()) / (S[:, 0].std() + 1e-8)
        A = (
            2.0 * axis[:, None]
            + rng.normal(scale=0.15, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = ((A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)).astype(
            np.float32
        )
        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=3,
            n_reruns=2,
            epochs=40,
            lr=0.01,
            seed=0,
            device="cpu",
            verbose=False,
        )
        model, outputs, _ = train_parallel_isodepth_model(
            S, A, config, model_label="test gaussian ols refit"
        )
        iso = extract_model_isodepth(model, S, next(model.parameters()).device)
        z = np.asarray(iso, dtype=np.float64).reshape(n_cells, -1)
        if z.shape[1] == 1:
            z = z[:, 0]
        pred_ols = fit_closed_form_decoder(z, A, "linear")
        np.testing.assert_allclose(
            np.asarray(outputs.pred_true, dtype=np.float64),
            np.asarray(pred_ols, dtype=np.float64),
            rtol=1e-4,
            atol=1e-4,
        )
        mse_ols = float(np.mean((A - pred_ols) ** 2))
        self.assertLess(mse_ols, 0.85)
        stat_true = float(outputs.stat_true)
        self.assertAlmostEqual(stat_true, mse_ols, places=4)
        self.assertAlmostEqual(
            stat_true,
            float(compute_metric("nll_gaussian_mse", A, pred_ols)),
            places=4,
        )
        self.assertTrue(np.all(np.isfinite(outputs.stat_perm)))

    def test_torch_ols_matches_numpy_lstsq_on_standardized_latent(self) -> None:
        rng = np.random.default_rng(3)
        n_cells, n_genes = 40, 5
        z = rng.normal(size=n_cells).astype(np.float64)
        A = (1.5 * z[:, None] + rng.normal(scale=0.2, size=(n_cells, n_genes))).astype(np.float64)
        pred_np = fit_closed_form_decoder(z, A, "linear")
        z_t = torch.tensor(z.reshape(1, n_cells, 1), dtype=torch.float64)
        a_t = torch.tensor(A, dtype=torch.float64)
        pred_t = closed_form_gaussian_decode_torch(z_t, a_t, "linear")
        np.testing.assert_allclose(pred_t.numpy()[0], pred_np, rtol=1e-5, atol=1e-5)
        mse_from_pred = ((pred_t - a_t) ** 2).mean(dim=(1, 2))
        mse_direct = closed_form_gaussian_mse_torch(z_t, a_t, "linear")
        np.testing.assert_allclose(
            mse_direct.detach().numpy(), mse_from_pred.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        z_multi = torch.tensor(rng.normal(size=(4, n_cells, 1)), dtype=torch.float64)
        mse_multi = closed_form_gaussian_mse_torch(z_multi, a_t, "linear")
        pred_multi = closed_form_gaussian_decode_torch(z_multi, a_t, "linear")
        mse_from_pred_multi = ((pred_multi - a_t) ** 2).mean(dim=(1, 2))
        np.testing.assert_allclose(
            mse_multi.detach().numpy(),
            mse_from_pred_multi.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_linear_gaussian_training_loss_beats_mean_baseline(self) -> None:
        rng = np.random.default_rng(4)
        n_cells, n_genes = 64, 6
        S = rng.uniform(size=(n_cells, 2)).astype(np.float32)
        axis = (S[:, 0] - S[:, 0].mean()) / (S[:, 0].std() + 1e-8)
        A = (
            2.2 * axis[:, None] + rng.normal(scale=0.12, size=(n_cells, n_genes))
        ).astype(np.float32)
        A = ((A - A.mean(axis=0, keepdims=True)) / (A.std(axis=0, keepdims=True) + 1e-8)).astype(
            np.float32
        )
        config = TestConfig(
            metric="nll_gaussian_mse",
            decoder="linear",
            n_perms=2,
            n_reruns=2,
            epochs=30,
            lr=0.02,
            seed=1,
            device="cpu",
            verbose=False,
            sgd_batch_size=32,
        )
        model, outputs, _ = train_parallel_isodepth_model(
            S, A, config, model_label="test training-time ols"
        )
        history = get_training_metadata(model)["loss_history"]
        self.assertIsNotNone(history)
        self.assertGreater(len(history), 5)
        self.assertLess(float(history[-1]), 0.5)
        self.assertLess(float(outputs.stat_true), 0.5)
        iso = extract_model_isodepth(model, S, next(model.parameters()).device)
        self.assertGreater(float(np.std(iso)), 0.05)

    def test_quadratic_mse_stays_finite_when_one_slot_is_singular(self) -> None:
        rng = np.random.default_rng(0)
        n_cells, n_genes, n_slots = 64, 5, 4
        z = torch.tensor(rng.normal(size=(n_slots, n_cells, 1)), dtype=torch.float32)
        z[1] = 0.0
        targets = torch.tensor(rng.normal(size=(n_cells, n_genes)), dtype=torch.float32)
        loss = closed_form_gaussian_mse_torch(z, targets, decoder_type="quadratic")
        self.assertTrue(torch.isfinite(loss).all())

    def test_solve_ols_beta_falls_back_when_columns_are_duplicated(self) -> None:
        design = torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0]]],
            dtype=torch.float64,
        )
        targets = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        beta = _solve_ols_beta(design, targets, ridge=0.0)
        self.assertTrue(torch.isfinite(beta).all())
        pred = torch.matmul(design, beta)
        self.assertTrue(torch.allclose(pred.squeeze(), targets.squeeze(), atol=1e-8))


if __name__ == "__main__":
    unittest.main()
