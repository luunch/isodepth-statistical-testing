from __future__ import annotations

import json
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

from data.schemas import DataConfig, TestConfig
from data.synthetic import generate_synthetic_dataset
from experiments.parallelization_comparison import (
    load_parallelization_comparison_spec,
    run_parallelization_comparison,
)
from methods.trainers import (
    build_parallel_initial_state,
    extract_model_isodepth,
    extract_parallel_slot_initial_state,
    resolve_device,
    train_isodepth_model,
    train_parallel_isodepth_model,
)


class TestParallelizationComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = generate_synthetic_dataset(
            DataConfig(source="synthetic", mode="radial", n_cells=12, n_genes=4, sigma=0.05, seed=11)
        )
        self.config = TestConfig(
            method="parallel_permutation",
            metric="mse",
            n_perms=2,
            n_reruns=1,
            epochs=8,
            patience=9,
            lr=1e-3,
            seed=13,
            device="cpu",
            decoder="linear",
            sgd_batch_size=0,
            verbose=False,
        )
        self.device = resolve_device("cpu")
        self.s_batched = np.stack(
            [
                self.dataset.S,
                self.dataset.S[::-1],
                np.roll(self.dataset.S, shift=1, axis=0),
            ],
            axis=0,
        ).astype(np.float32)

    def _aligned_states(self) -> dict[str, object]:
        return build_parallel_initial_state(
            self.s_batched.shape[0],
            self.dataset.n_genes,
            latent_dim=1,
            decoder_type=self.config.decoder,
            seed=self.config.seed,
            device=self.device,
        )

    def test_initial_state_hook_matches_parallel_slot_pretraining_predictions(self) -> None:
        initial_state = self._aligned_states()

        def _noop_step(_optimizer, closure=None):
            return None

        with patch("methods.trainers.isodepth.optim.Adam.step", new=_noop_step):
            parallel_model, parallel_predictions = train_parallel_isodepth_model(
                self.dataset.S,
                self.dataset.A,
                self.config,
                device=self.device,
                s_batched=self.s_batched,
                initial_state=initial_state,
                gradient_scale_divisor=float(self.s_batched.shape[0]),
            )
            for model_index in range(self.s_batched.shape[0]):
                slot_state = extract_parallel_slot_initial_state(
                    initial_state,
                    slot_index=model_index,
                    n_genes=self.dataset.n_genes,
                    latent_dim=1,
                    decoder_type=self.config.decoder,
                    device=self.device,
                )
                sequential_model, sequential_prediction = train_isodepth_model(
                    self.s_batched[model_index],
                    self.dataset.A,
                    self.config,
                    device=self.device,
                    latent_dim=1,
                    seed_offset=0,
                    initial_state=slot_state,
                    gradient_scale_divisor=float(self.s_batched.shape[0]),
                    model_label=f"slot {model_index}",
                )
                np.testing.assert_allclose(
                    parallel_predictions[model_index],
                    sequential_prediction,
                    atol=1e-7,
                )
                with torch.no_grad():
                    parallel_depth = (
                        parallel_model.encoder(torch.tensor(self.s_batched, dtype=torch.float32, device=self.device))
                        .detach()
                        .cpu()
                        .numpy()[model_index]
                    )
                sequential_depth = extract_model_isodepth(sequential_model, self.s_batched[model_index], self.device)
                np.testing.assert_allclose(parallel_depth, sequential_depth, atol=1e-7)

    def test_aligned_parallel_and_sequential_training_match_on_cpu(self) -> None:
        initial_state = self._aligned_states()
        parallel_model, parallel_predictions = train_parallel_isodepth_model(
            self.dataset.S,
            self.dataset.A,
            self.config,
            device=self.device,
            s_batched=self.s_batched,
            initial_state=initial_state,
            gradient_scale_divisor=float(self.s_batched.shape[0]),
        )
        for model_index in range(self.s_batched.shape[0]):
            slot_state = extract_parallel_slot_initial_state(
                initial_state,
                slot_index=model_index,
                n_genes=self.dataset.n_genes,
                latent_dim=1,
                decoder_type=self.config.decoder,
                device=self.device,
            )
            sequential_model, sequential_prediction = train_isodepth_model(
                self.s_batched[model_index],
                self.dataset.A,
                self.config,
                device=self.device,
                latent_dim=1,
                seed_offset=0,
                initial_state=slot_state,
                gradient_scale_divisor=float(self.s_batched.shape[0]),
                model_label=f"slot {model_index}",
            )
            np.testing.assert_allclose(
                parallel_predictions[model_index],
                sequential_prediction,
                atol=1e-5,
            )
            with torch.no_grad():
                parallel_depth = (
                    parallel_model.encoder(torch.tensor(self.s_batched, dtype=torch.float32, device=self.device))
                    .detach()
                    .cpu()
                    .numpy()[model_index]
                )
            sequential_depth = extract_model_isodepth(sequential_model, self.s_batched[model_index], self.device)
            np.testing.assert_allclose(parallel_depth, sequential_depth, atol=1e-5)

    def test_comparison_runner_writes_outputs_and_preserves_slot_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "comparison_spec.json"
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
                "experiment_name": "radial_parallelization_check",
                "base_config": str(base_config_path),
                "output_root": str(tmp_path / "comparison_outputs"),
                "n_perms": 2,
                "epochs": 4,
                "device": "cpu",
            }
            base_config_path.write_text(json.dumps(base_config_payload), encoding="utf-8")
            spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")

            spec = load_parallelization_comparison_spec(spec_path)
            payload = run_parallelization_comparison(spec)

            self.assertEqual(payload["epochs_override"], 4)
            self.assertEqual(payload["effective_run_config"]["test"]["epochs"], 4)
            self.assertEqual(payload["effective_run_config"]["test"]["n_reruns"], 1)
            self.assertEqual(payload["permutation_seed"], 13)
            self.assertEqual(payload["n_perms"], 2)
            self.assertEqual(payload["n_models"], 3)
            self.assertEqual(len(payload["per_slot"]), 3)
            for expected_model_index, row in enumerate(payload["per_slot"]):
                self.assertEqual(row["model_index"], expected_model_index)
                expected_perm_index = None if expected_model_index == 0 else expected_model_index - 1
                self.assertEqual(row["perm_index"], expected_perm_index)

            for key in [
                "parallel_grid_plot",
                "sequential_grid_plot",
                "paired_comparison_plot",
                "comparison_result_path",
            ]:
                self.assertTrue(Path(payload[key]).exists(), key)

            saved_payload = json.loads(Path(payload["comparison_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_payload["epochs_override"], 4)
            self.assertIn("summary", saved_payload)
            self.assertIn("permutation_seed", saved_payload)
            self.assertNotIn("parallel_isodepths", saved_payload)
            self.assertNotIn("sequential_isodepths", saved_payload)
            self.assertNotIn("parallel_predictions", saved_payload)
            self.assertNotIn("sequential_predictions", saved_payload)


if __name__ == "__main__":
    unittest.main()
