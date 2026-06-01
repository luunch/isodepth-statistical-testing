from __future__ import annotations

import unittest

import numpy as np

from experiments.batchsize_comparison import (
    _build_fixed_update_schedule,
    _build_regime_list,
    _correlation_to_synthetic,
    _epochs_for_equal_updates,
    _resolve_regime_test_config,
)
from data.schemas import TestConfig


class BatchSizeComparisonScheduleTests(unittest.TestCase):
    def test_build_regime_list(self) -> None:
        regimes = _build_regime_list([512, 8])
        self.assertEqual(len(regimes), 3)
        self.assertEqual(regimes[0]["label"], "true_full_batch")

    def test_epochs_for_equal_updates_minibatch(self) -> None:
        self.assertEqual(_epochs_for_equal_updates(5000, 1250), 4)

    def test_build_fixed_update_schedule(self) -> None:
        schedule = _build_fixed_update_schedule(10_000, 5000, [512, 8])
        batch8 = next(item for item in schedule if item["label"] == "batch_8")
        self.assertEqual(batch8["planned_total_updates"], 5000)

    def test_resolve_regime_test_config_uses_time_budget(self) -> None:
        base = TestConfig()
        config = _resolve_regime_test_config(
            base,
            batch_size=128,
            n_perms=10,
            n_reruns=2,
            n_cells=1000,
            base_updates=500,
            time_budget_sec=60.0,
            record_loss_history=True,
        )
        self.assertEqual(config.max_wall_time_sec, 60.0)
        self.assertEqual(config.sgd_batch_size, 128)
        self.assertTrue(config.record_loss_history)


class BatchSizeComparisonCorrelationTests(unittest.TestCase):
    def test_correlation_to_synthetic_perfect_and_inverted(self) -> None:
        true = np.linspace(0.0, 1.0, 8, dtype=np.float64)
        learned = true.copy()
        pearson, spearman = _correlation_to_synthetic(learned, true)
        self.assertAlmostEqual(pearson, 1.0, places=8)
        self.assertAlmostEqual(spearman, 1.0, places=8)

        inverted = -true
        pearson_inv, spearman_inv = _correlation_to_synthetic(inverted, true)
        self.assertAlmostEqual(pearson_inv, -1.0, places=8)
        self.assertAlmostEqual(spearman_inv, -1.0, places=8)


if __name__ == "__main__":
    unittest.main()
