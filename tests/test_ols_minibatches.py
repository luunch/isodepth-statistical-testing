"""Gaussian OLS training must honor cell minibatches in both cell-type modes."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig
from methods.permutation import run_parallel_permutation_method
from methods.trainers import isodepth as trainer


class OlsMinibatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_celltype_batch_sizes_and_update_counts(self):
        rng = np.random.default_rng(31)
        s = rng.normal(size=(30, 2)).astype(np.float32)
        a = (s[:, :1] + rng.normal(scale=0.2, size=(30, 4))).astype(np.float32)
        for mode in ("separate", "together"):
            dataset = DatasetBundle(S=s, A=a, meta={
                "cell_type_mode": mode,
                "cell_type_labels": np.repeat([0, 1], [17, 13]),
                "cell_type_names": ["first", "second"],
                "n_cell_types": 2,
            })
            for decoder in ("linear", "quadratic"):
                for batch_size in (None, 0, 8, 64):
                    with self.subTest(mode=mode, decoder=decoder, batch_size=batch_size):
                        config = TestConfig(
                            metric="nll_gaussian_mse", decoder=decoder,
                            n_perms=1, n_reruns=2, epochs=3, patience=0,
                            sgd_batch_size=batch_size,
                            sgd_cosine_lr_decay=bool(batch_size),
                            device="cpu", verbose=False, seed=31,
                        )
                        training_sizes = []
                        ols_loss = trainer.closed_form_gaussian_mse_torch

                        def record_loss(z, *args, **kwargs):
                            if torch.is_grad_enabled():
                                training_sizes.append(z.shape[1])
                            return ols_loss(z, *args, **kwargs)

                        with patch.object(trainer, "closed_form_gaussian_mse_torch", record_loss):
                            result = run_parallel_permutation_method(dataset, config)
                        self.assertTrue(training_sizes)
                        if batch_size:
                            self.assertLessEqual(max(training_sizes), batch_size)
                        if mode == "separate":
                            runs = result.artifacts["per_type_results"].values()
                            # Includes the one-cell remainder of the 17-cell subset.
                            expected_sizes = []
                            for n in (17, 13):
                                b = min(batch_size or n, n)
                                expected_sizes.extend(
                                    [min(b, n - start) for start in range(0, n, b)] * config.epochs
                                )
                            self.assertEqual(training_sizes, expected_sizes)
                        else:
                            runs = [{**result.artifacts, "n_cells": 30,
                                     "stat_true": result.stat_true, "stat_perm": result.stat_perm}]
                        for run in runs:
                            n = run["n_cells"]
                            b = min(batch_size or n, n)
                            steps = (n + b - 1) // b
                            metadata = trainer.get_training_metadata(run["model"])
                            self.assertEqual(metadata["executed_gradient_steps"], config.epochs * steps)
                            np.testing.assert_array_equal(
                                metadata["loss_history_gradient_updates"],
                                np.arange(1, config.epochs + 1) * steps,
                            )
                            self.assertTrue(np.isfinite(run["stat_true"]))
                            self.assertTrue(np.all(np.isfinite(run["stat_perm"])))
                            self.assertTrue(np.all(np.isfinite(run["pred_true"])))


if __name__ == "__main__":
    unittest.main()
