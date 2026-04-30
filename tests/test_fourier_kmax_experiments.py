from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.configuration import build_run_config
from experiments.fourier_kmax import (
    FourierKmaxStudySpec,
    build_fourier_kmax_run_config,
    expand_fourier_kmax_conditions,
    extract_kmax_record,
    load_fourier_kmax_spec,
    summarize_kmax_rows,
)
from experiments.fourier_kmax_analysis import analyze_fourier_kmax_results
from experiments.fourier_kmax_sweep import run_fourier_kmax_sweep


class TestFourierKmaxSpec(unittest.TestCase):
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
                            "mode": "fourier",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "k_min": 1,
                            "k_max": 3,
                            "dependent_xy": False,
                            "poly_degree": 1,
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
                        "experiment_name": "fourier_kmax_study",
                        "base_config": str(base_config),
                        "output_root": str(tmp_path / "study_outputs"),
                        "k_min": 1,
                        "k_max_values": [1, 2, 4],
                        "seeds": [0, 1],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_spec(spec_path)

            self.assertEqual(spec.experiment_name, "fourier_kmax_study")
            self.assertEqual(spec.k_min, 1)
            self.assertEqual(spec.k_max_values, [1, 2, 4])
            self.assertEqual(spec.seeds, [0, 1])
            self.assertTrue(spec.base_config.is_absolute())
            self.assertTrue(spec.output_root.is_absolute())

    def test_expand_conditions_produces_stable_grid_and_names(self) -> None:
        spec = FourierKmaxStudySpec(
            experiment_name="fourier_kmax_study",
            base_config=Path("/tmp/base.json"),
            output_root=Path("/tmp/out"),
            k_min=1,
            k_max_values=[1, 4],
            seeds=[0, 2],
            reuse_result_roots=[],
        )

        conditions = expand_fourier_kmax_conditions(spec)

        self.assertEqual(len(conditions), 4)
        self.assertEqual(
            conditions[0].run_name,
            "fourier_kmax_study__kmin-01__kmax-01__seed-000",
        )
        self.assertEqual(
            conditions[-1].run_name,
            "fourier_kmax_study__kmin-01__kmax-04__seed-002",
        )

    def test_cross_validation_base_config_is_accepted_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            spec_path = tmp_path / "spec.json"
            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "synthetic",
                            "mode": "fourier",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "k_min": 1,
                            "k_max": 3,
                            "dependent_xy": False,
                            "poly_degree": 1,
                            "seed": 0,
                        },
                        "test": {
                            "method": "cross_validation",
                            "metric": "mse",
                            "n_perms": 2,
                            "train_fraction": 0.7,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 0,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(tmp_path / "results"), "run_name": "base"},
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "fourier_kmax_study",
                        "base_config": str(base_config),
                        "output_root": str(tmp_path / "study_outputs"),
                        "k_min": 1,
                        "k_max_values": [2],
                        "seeds": [7],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_spec(spec_path)
            condition = expand_fourier_kmax_conditions(spec)[0]
            run_config = build_fourier_kmax_run_config(
                base_run_config=build_run_config(str(spec.base_config), {}),
                spec=spec,
                condition=condition,
            )

        self.assertEqual(run_config.test.method, "cross_validation")
        self.assertEqual(run_config.test.train_fraction, 0.7)


class TestFourierKmaxHelpers(unittest.TestCase):
    def test_extract_record_reads_kmax_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "fourier_existence_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "parallel_permutation",
                        "metric": "mse",
                        "p_value": 0.125,
                        "stat_true": 1.5,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "fourier",
                                "sigma": 0.1,
                                "seed": 7,
                                "k_min": 1,
                                "k_max": 4,
                                "dependent_xy": False,
                                "poly_degree": 1,
                            },
                            "output": {"run_name": "fourier_existence"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_kmax_record(
                result_path,
                expected_sigma=0.1,
                expected_dependent_xy=False,
                expected_poly_degree=1,
            )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record["k_min"], 1)
            self.assertEqual(record["k_max"], 4)
            self.assertEqual(record["seed"], 7)
            self.assertAlmostEqual(float(record["p_value"]), 0.125)

    def test_extract_record_accepts_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "fourier_cross_validation_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "cross_validation",
                        "metric": "mse",
                        "p_value": 0.125,
                        "stat_true": 1.5,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "fourier",
                                "sigma": 0.1,
                                "seed": 7,
                                "k_min": 1,
                                "k_max": 4,
                                "dependent_xy": False,
                                "poly_degree": 1,
                            },
                            "output": {"run_name": "fourier_cross_validation"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_kmax_record(
                result_path,
                expected_sigma=0.1,
                expected_dependent_xy=False,
                expected_poly_degree=1,
            )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["method_name"], "cross_validation")

    def test_summary_groups_by_kmax(self) -> None:
        rows = [
            {"k_max": 2, "p_value": 0.2, "stat_true": 2.0},
            {"k_max": 2, "p_value": 0.4, "stat_true": 4.0},
            {"k_max": 5, "p_value": 0.1, "stat_true": 6.0},
        ]

        summaries = summarize_kmax_rows(rows)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["k_max"], 2)
        self.assertEqual(summaries[0]["n_runs"], 2)
        self.assertAlmostEqual(float(summaries[0]["p_value_mean"]), 0.3)
        self.assertEqual(summaries[1]["k_max"], 5)

    def test_analysis_builds_csvs_and_plot_from_saved_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            output_root = tmp_path / "study_outputs"
            result_root = output_root / "runs" / "manual"
            result_root.mkdir(parents=True, exist_ok=True)
            spec_path = tmp_path / "spec.json"
            result_path = result_root / "manual_result.json"

            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "synthetic",
                            "mode": "fourier",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "k_min": 1,
                            "k_max": 3,
                            "dependent_xy": False,
                            "poly_degree": 1,
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
                        "experiment_name": "fourier_kmax_study",
                        "base_config": str(base_config),
                        "output_root": str(output_root),
                        "k_min": 1,
                        "k_max_values": [2],
                        "seeds": [0],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )
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
                                "mode": "fourier",
                                "sigma": 0.1,
                                "seed": 0,
                                "k_min": 1,
                                "k_max": 2,
                                "dependent_xy": False,
                                "poly_degree": 1,
                            },
                            "output": {"run_name": "manual"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "fourier_kmax_study",
                        "runs": [
                            {
                                "run_name": "manual",
                                "seed": 0,
                                "k_min": 1,
                                "k_max": 2,
                                "result_json_path": str(result_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = analyze_fourier_kmax_results(spec_path)

            self.assertEqual(analysis["n_rows_analyzed"], 1)
            self.assertTrue((output_root / "analysis" / "per_run_results.csv").exists())
            self.assertTrue((output_root / "analysis" / "summary_by_kmax.csv").exists())
            self.assertTrue((output_root / "analysis" / "pvalue_vs_kmax.png").exists())


@unittest.skipUnless(HAS_TORCH, "torch is required for Fourier k_max smoke tests")
class TestFourierKmaxSmoke(unittest.TestCase):
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
                            "mode": "fourier",
                            "n_cells": 16,
                            "n_genes": 2,
                            "sigma": 0.1,
                            "k_min": 1,
                            "k_max": 2,
                            "dependent_xy": False,
                            "poly_degree": 1,
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
                        "experiment_name": "fourier_kmax_study",
                        "base_config": str(base_config),
                        "output_root": str(output_root),
                        "k_min": 1,
                        "k_max_values": [1, 2],
                        "seeds": [0],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_spec(spec_path)
            manifest = run_fourier_kmax_sweep(spec)
            analysis = analyze_fourier_kmax_results(spec_path)

            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(analysis["n_rows_analyzed"], 2)
            self.assertTrue((output_root / "analysis" / "pvalue_vs_kmax.png").exists())


if __name__ == "__main__":
    unittest.main()
