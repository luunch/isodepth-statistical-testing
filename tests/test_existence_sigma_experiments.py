from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DatasetBundle
from experiments.existence_sigma import (
    ExistenceSigmaStudySpec,
    SweepCondition,
    expand_existence_sigma_conditions,
    extract_result_record,
    load_existence_sigma_spec,
    matched_shuffle_dataset,
    summarize_condition_rows,
)
from experiments.existence_sigma_analysis import analyze_existence_sigma_results
from experiments.existence_sigma_sweep import run_existence_sigma_sweep


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestExistenceSigmaSpec(unittest.TestCase):
    def test_load_spec_validates_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            spec_path = tmp_path / "spec.json"
            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "synthetic",
                            "mode": "radial",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 0,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {
                            "out_dir": str(tmp_path / "results"),
                            "run_name": "base",
                            "save_preds": False,
                            "save_perm_stats": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "study",
                        "base_config": str(base_config),
                        "output_root": str(tmp_path / "study_outputs"),
                        "alpha": 0.05,
                        "sigma_values": [0.0, 0.1],
                        "seeds": [0, 1],
                        "include_radial": True,
                        "fourier_k_values": [1, 3],
                        "null_family": "matched_shuffled",
                        "reuse_result_roots": [str(tmp_path / "old_results")],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_existence_sigma_spec(spec_path)

            self.assertEqual(spec.experiment_name, "study")
            self.assertEqual(spec.sigma_values, [0.0, 0.1])
            self.assertEqual(spec.seeds, [0, 1])
            self.assertEqual(spec.fourier_k_values, [1, 3])
            self.assertTrue(spec.base_config.is_absolute())
            self.assertTrue(spec.output_root.is_absolute())

    def test_expand_conditions_produces_stable_grid_and_names(self) -> None:
        spec = ExistenceSigmaStudySpec(
            experiment_name="study",
            base_config=REPO_ROOT / "configs/synthetic_existence_base.json",
            output_root=REPO_ROOT / "results/experiments/test",
            alpha=0.05,
            sigma_values=[0.0, 0.1],
            seeds=[0],
            include_radial=True,
            fourier_k_values=[2],
            null_family="matched_shuffled",
            reuse_result_roots=[],
        ).validate()

        conditions = expand_existence_sigma_conditions(spec)

        self.assertEqual(len(conditions), 8)
        self.assertEqual(conditions[0].run_name, "study__truth-alternative__mode-radial__sigma-0__seed-000")
        self.assertEqual(
            conditions[3].run_name,
            "study__truth-null__mode-fourier__sigma-0__seed-000__k-02__null-matched_shuffled",
        )


class TestExistenceSigmaHelpers(unittest.TestCase):
    def test_matched_shuffle_is_reproducible_and_preserves_values(self) -> None:
        dataset = DatasetBundle(
            S=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            A=np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            meta={"mode": "radial", "sigma": 0.1},
        ).validate()
        condition = SweepCondition(
            truth_label="null",
            mode="radial",
            sigma=0.1,
            seed=7,
            k=None,
            null_family="matched_shuffled",
            run_name="study__truth-null__mode-radial__sigma-0p1__seed-007__null-matched_shuffled",
        )

        shuffled_a = matched_shuffle_dataset(dataset, seed=7, condition=condition)
        shuffled_b = matched_shuffle_dataset(dataset, seed=7, condition=condition)

        np.testing.assert_allclose(shuffled_a.S, dataset.S)
        np.testing.assert_allclose(shuffled_a.A, shuffled_b.A)
        np.testing.assert_allclose(np.sort(shuffled_a.A.reshape(-1)), np.sort(dataset.A.reshape(-1)))
        self.assertEqual(shuffled_a.meta["truth_label"], "null")
        self.assertEqual(shuffled_a.meta["null_family"], "matched_shuffled")

    def test_extract_result_record_warns_on_name_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "radial_existence_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "parallel_permutation",
                        "metric": "mse",
                        "p_value": 0.2,
                        "stat_true": 1.0,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "noise",
                                "sigma": 0.1,
                                "seed": 3,
                            },
                            "output": {"run_name": "radial_existence"},
                        },
                        "artifacts": {
                            "dataset_meta": {
                                "source": "synthetic",
                                "mode": "noise",
                                "sigma": 0.1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_result_record(result_path)

            self.assertIsNotNone(record)
            self.assertEqual(record["truth_label"], "null")
            self.assertEqual(record["mode"], "noise")
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["warning_type"], "run_name_mode_mismatch")

    def test_summary_computes_rates(self) -> None:
        rows = [
            {
                "truth_label": "alternative",
                "mode": "radial",
                "family_label": "radial",
                "sigma": 0.1,
                "k": "",
                "reject": 1,
            },
            {
                "truth_label": "alternative",
                "mode": "radial",
                "family_label": "radial",
                "sigma": 0.1,
                "k": "",
                "reject": 0,
            },
        ]

        summaries = summarize_condition_rows(rows, alpha=0.05)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["summary_metric"], "power")
        self.assertEqual(summaries[0]["n_runs"], 2)
        self.assertAlmostEqual(float(summaries[0]["rate"]), 0.5)


@unittest.skipUnless(HAS_TORCH, "torch is required for existence sigma smoke tests")
class TestExistenceSigmaSmoke(unittest.TestCase):
    def test_sweep_and_analysis_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            output_root = tmp_path / "study_outputs"
            spec_path = tmp_path / "spec.json"

            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "synthetic",
                            "mode": "radial",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 0,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {
                            "out_dir": str(output_root / "runs"),
                            "run_name": "base",
                            "save_preds": False,
                            "save_perm_stats": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "study",
                        "base_config": str(base_config),
                        "output_root": str(output_root),
                        "alpha": 0.05,
                        "sigma_values": [0.1],
                        "seeds": [0],
                        "include_radial": True,
                        "fourier_k_values": [1],
                        "null_family": "matched_shuffled",
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_existence_sigma_spec(spec_path)
            manifest = run_existence_sigma_sweep(spec, max_runs=None)
            analysis = analyze_existence_sigma_results(spec_path)

            self.assertEqual(len(manifest["runs"]), 4)
            self.assertTrue((output_root / "analysis" / "per_run_results.csv").exists())
            self.assertTrue((output_root / "analysis" / "summary_by_condition.csv").exists())
            self.assertTrue((output_root / "analysis" / "power_vs_sigma.png").exists())
            self.assertTrue((output_root / "analysis" / "null_rejection_vs_sigma.png").exists())
            self.assertTrue((output_root / "analysis" / "fourier_power_heatmap.png").exists())
            self.assertTrue((output_root / "analysis" / "fourier_null_rejection_heatmap.png").exists())
            self.assertEqual(analysis["n_rows_analyzed"], 4)


if __name__ == "__main__":
    unittest.main()
