from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import TestConfig
from methods.metrics import compute_metric, permutation_p_value


class TestPerturbationSchema(unittest.TestCase):
    def test_perturbation_robustness_config_is_valid(self) -> None:
        config = TestConfig(
            method="perturbation_robustness",
            metric="spearman_corr_mean",
            delta=0.1,
            perturb_target="coordinates",
        )
        self.assertIs(config.validate(), config)

    def test_invalid_perturb_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(
                method="perturbation_robustness",
                delta=0.1,
                perturb_target="expression",
            ).validate()

    def test_invalid_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(method="perturbation_robustness", delta=0.0).validate()


class TestMetricUtilities(unittest.TestCase):
    def test_spearman_metric_handles_reversed_order(self) -> None:
        y_true = [[0.0], [1.0], [2.0], [3.0]]
        y_pred = [[3.0], [2.0], [1.0], [0.0]]
        self.assertAlmostEqual(compute_metric("spearman_corr_mean", y_true, y_pred), -1.0)

    def test_permutation_p_value_respects_metric_tail(self) -> None:
        lower_tail = permutation_p_value("mse", 0.05, [0.1, 0.2, 0.3])
        upper_tail = permutation_p_value("spearman_corr_mean", 0.9, [0.2, 0.3, 0.4])
        self.assertLess(lower_tail, 0.5)
        self.assertLess(upper_tail, 0.5)


if __name__ == "__main__":
    unittest.main()
