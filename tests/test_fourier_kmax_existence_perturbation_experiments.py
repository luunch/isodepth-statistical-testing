from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.configuration import build_run_config
from experiments.fourier_kmax_existence_perturbation import (
    FourierKmaxExistencePerturbationStudySpec,
    build_fourier_kmax_study_run_config,
    build_boxplot_rows,
    expand_fourier_kmax_study_conditions,
    extract_existence_record,
    extract_perturbation_records,
    load_fourier_kmax_existence_perturbation_spec,
    summarize_perturbation_rows,
)
from experiments.fourier_kmax_existence_perturbation_analysis import (
    analyze_fourier_kmax_existence_perturbation_results,
)
from experiments.fourier_kmax_existence_perturbation_sweep import (
    run_fourier_kmax_existence_perturbation_sweep,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestFourierKmaxExistencePerturbationSpec(unittest.TestCase):
    def test_load_spec_validates_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            existence_base = tmp_path / "existence.json"
            perturbation_base = tmp_path / "perturbation.json"
            spec_path = tmp_path / "spec.json"

            existence_base.write_text(
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
                            "seed": 42,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(tmp_path / "results"), "run_name": "existence"},
                    }
                ),
                encoding="utf-8",
            )
            perturbation_base.write_text(
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
                            "seed": 42,
                        },
                        "test": {
                            "method": "perturbation_test",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "delta": [0.01, 0.05],
                            "verbose": False,
                        },
                        "output": {"out_dir": str(tmp_path / "results"), "run_name": "perturbation"},
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "combined",
                        "existence_base_config": str(existence_base),
                        "perturbation_base_config": str(perturbation_base),
                        "output_root": str(tmp_path / "study_outputs"),
                        "k_min": 1,
                        "k_max_values": [1, 2, 3],
                        "seeds": [42, 43],
                        "delta_values": [0.01, 0.05],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_existence_perturbation_spec(spec_path)

            self.assertEqual(spec.experiment_name, "combined")
            self.assertEqual(spec.k_max_values, [1, 2, 3])
            self.assertEqual(spec.seeds, [42, 43])
            self.assertEqual(spec.delta_values, [0.01, 0.05])

    def test_expand_conditions_creates_both_test_kinds(self) -> None:
        spec = FourierKmaxExistencePerturbationStudySpec(
            experiment_name="combined",
            existence_base_config=Path("/tmp/existence.json"),
            perturbation_base_config=Path("/tmp/perturbation.json"),
            output_root=Path("/tmp/out"),
            k_min=1,
            k_max_values=[1, 2],
            seeds=[42, 43],
            delta_values=[0.01, 0.05],
            reuse_result_roots=[],
        )

        conditions = expand_fourier_kmax_study_conditions(spec)

        self.assertEqual(len(conditions), 8)
        self.assertEqual(conditions[0].test_kind, "existence")
        self.assertEqual(conditions[1].test_kind, "perturbation")

    def test_cross_validation_existence_base_is_accepted_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            existence_base = tmp_path / "existence.json"
            perturbation_base = tmp_path / "perturbation.json"
            spec_path = tmp_path / "spec.json"

            existence_base.write_text(
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
                            "seed": 42,
                        },
                        "test": {
                            "method": "cross_validation",
                            "metric": "mse",
                            "n_perms": 2,
                            "train_fraction": 0.7,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(tmp_path / "results"), "run_name": "existence"},
                    }
                ),
                encoding="utf-8",
            )
            perturbation_base.write_text(
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
                            "seed": 42,
                        },
                        "test": {
                            "method": "perturbation_test",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "delta": [0.01, 0.05],
                            "verbose": False,
                        },
                        "output": {"out_dir": str(tmp_path / "results"), "run_name": "perturbation"},
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "combined",
                        "existence_base_config": str(existence_base),
                        "perturbation_base_config": str(perturbation_base),
                        "output_root": str(tmp_path / "study_outputs"),
                        "k_min": 1,
                        "k_max_values": [2],
                        "seeds": [42],
                        "delta_values": [0.01, 0.05],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_existence_perturbation_spec(spec_path)
            condition = expand_fourier_kmax_study_conditions(spec)[0]
            run_config = build_fourier_kmax_study_run_config(
                existence_base_run_config=build_run_config(str(spec.existence_base_config), {}),
                perturbation_base_run_config=build_run_config(str(spec.perturbation_base_config), {}),
                spec=spec,
                condition=condition,
            )

        self.assertEqual(run_config.test.method, "cross_validation")
        self.assertEqual(run_config.test.train_fraction, 0.7)


class TestFourierKmaxExistencePerturbationHelpers(unittest.TestCase):
    def test_extract_perturbation_records_expands_each_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "perturbation_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "perturbation_test",
                        "metric": "mse",
                        "p_value": 0.2,
                        "stat_true": 1.5,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "fourier",
                                "sigma": 0.1,
                                "seed": 42,
                                "k_min": 1,
                                "k_max": 5,
                                "dependent_xy": False,
                                "poly_degree": 1,
                            },
                            "output": {"run_name": "perturbation"},
                        },
                        "artifacts": {
                            "delta_summaries": {
                                "0.01": {"delta": 0.01, "p_value": 0.2, "score_mean": 1.5},
                                "0.05": {"delta": 0.05, "p_value": 0.7, "score_mean": 1.1},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            rows, warnings = extract_perturbation_records(
                result_path,
                expected_sigma=0.1,
                expected_dependent_xy=False,
                expected_poly_degree=1,
            )

            self.assertEqual(warnings, [])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["delta"], 0.01)
            self.assertEqual(rows[0]["p_value"], 0.2)
            self.assertEqual(rows[1]["delta"], 0.05)
            self.assertEqual(rows[1]["p_value"], 0.7)

    def test_boxplot_rows_keep_one_p_value_per_delta(self) -> None:
        existence_rows = [{"k_max": 2, "seed": 42, "p_value": 0.1}]
        perturbation_rows = [
            {"k_max": 2, "seed": 42, "delta": 0.01, "p_value": 0.2},
            {"k_max": 2, "seed": 42, "delta": 0.05, "p_value": 0.6},
        ]

        rows = build_boxplot_rows(existence_rows, perturbation_rows)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["test_label"], "existence")
        self.assertEqual(rows[1]["test_label"], "delta=0.01")
        self.assertEqual(rows[2]["test_label"], "delta=0.05")

    def test_extract_existence_record_accepts_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "existence_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "cross_validation",
                        "metric": "mse",
                        "p_value": 0.1,
                        "stat_true": 1.5,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "fourier",
                                "sigma": 0.1,
                                "seed": 42,
                                "k_min": 1,
                                "k_max": 5,
                                "dependent_xy": False,
                                "poly_degree": 1,
                            },
                            "output": {"run_name": "existence"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_existence_record(
                result_path,
                expected_sigma=0.1,
                expected_dependent_xy=False,
                expected_poly_degree=1,
            )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["method_name"], "cross_validation")

    def test_summarize_perturbation_rows_groups_by_kmax_and_delta(self) -> None:
        summaries = summarize_perturbation_rows(
            [
                {"k_max": 2, "delta": 0.01, "p_value": 0.2, "stat_true": 1.0},
                {"k_max": 2, "delta": 0.01, "p_value": 0.4, "stat_true": 1.0},
                {"k_max": 2, "delta": 0.05, "p_value": 0.6, "stat_true": 1.0},
            ]
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["k_max"], 2)
        self.assertEqual(summaries[0]["delta"], 0.01)
        self.assertAlmostEqual(float(summaries[0]["p_value_mean"]), 0.3)

    def test_analysis_generates_csvs_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            existence_base = tmp_path / "existence.json"
            perturbation_base = tmp_path / "perturbation.json"
            output_root = tmp_path / "study_outputs"
            spec_path = tmp_path / "spec.json"
            manual_root = output_root / "runs" / "manual"
            manual_root.mkdir(parents=True, exist_ok=True)
            existence_result = manual_root / "existence_result.json"
            perturbation_result = manual_root / "perturbation_result.json"

            base_data = {
                "source": "synthetic",
                "mode": "fourier",
                "n_cells": 16,
                "n_genes": 2,
                "sigma": 0.1,
                "k_min": 1,
                "k_max": 2,
                "dependent_xy": False,
                "poly_degree": 1,
                "seed": 42,
            }
            existence_base.write_text(
                json.dumps(
                    {
                        "data": base_data,
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(output_root / "runs"), "run_name": "existence"},
                    }
                ),
                encoding="utf-8",
            )
            perturbation_base.write_text(
                json.dumps(
                    {
                        "data": base_data,
                        "test": {
                            "method": "perturbation_test",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "delta": [0.01, 0.05],
                            "verbose": False,
                        },
                        "output": {"out_dir": str(output_root / "runs"), "run_name": "perturbation"},
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "combined",
                        "existence_base_config": str(existence_base),
                        "perturbation_base_config": str(perturbation_base),
                        "output_root": str(output_root),
                        "k_min": 1,
                        "k_max_values": [2],
                        "seeds": [42],
                        "delta_values": [0.01, 0.05],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )
            existence_result.write_text(
                json.dumps(
                    {
                        "method_name": "parallel_permutation",
                        "metric": "mse",
                        "p_value": 0.2,
                        "stat_true": 1.0,
                        "runtime_sec": 0.5,
                        "config": {"data": base_data, "output": {"run_name": "existence"}},
                    }
                ),
                encoding="utf-8",
            )
            perturbation_result.write_text(
                json.dumps(
                    {
                        "method_name": "perturbation_test",
                        "metric": "mse",
                        "p_value": 0.2,
                        "stat_true": 1.0,
                        "runtime_sec": 0.5,
                        "config": {"data": base_data, "output": {"run_name": "perturbation"}},
                        "artifacts": {
                            "delta_summaries": {
                                "0.01": {"delta": 0.01, "p_value": 0.2, "score_mean": 1.0},
                                "0.05": {"delta": 0.05, "p_value": 0.6, "score_mean": 1.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (output_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "combined",
                        "runs": [
                            {
                                "run_name": "existence",
                                "test_kind": "existence",
                                "seed": 42,
                                "k_min": 1,
                                "k_max": 2,
                                "result_json_path": str(existence_result),
                            },
                            {
                                "run_name": "perturbation",
                                "test_kind": "perturbation",
                                "seed": 42,
                                "k_min": 1,
                                "k_max": 2,
                                "result_json_path": str(perturbation_result),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = analyze_fourier_kmax_existence_perturbation_results(spec_path)

            self.assertEqual(analysis["n_existence_rows_analyzed"], 1)
            self.assertEqual(analysis["n_perturbation_rows_analyzed"], 2)
            self.assertTrue((output_root / "analysis" / "existence_per_run_results.csv").exists())
            self.assertTrue((output_root / "analysis" / "perturbation_per_run_results.csv").exists())
            self.assertTrue((output_root / "analysis" / "boxplot_values_by_kmax.csv").exists())
            self.assertTrue((output_root / "analysis" / "existence_pvalue_vs_kmax.png").exists())
            self.assertTrue((output_root / "analysis" / "perturbation_pvalue_vs_kmax.png").exists())
            self.assertTrue((output_root / "analysis" / "boxplots_kmax_2.png").exists())


@unittest.skipUnless(HAS_TORCH, "torch is required for Fourier k_max combined smoke tests")
class TestFourierKmaxExistencePerturbationSmoke(unittest.TestCase):
    def test_sweep_and_analysis_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            existence_base = tmp_path / "existence.json"
            perturbation_base = tmp_path / "perturbation.json"
            output_root = tmp_path / "study_outputs"
            spec_path = tmp_path / "spec.json"

            base_data = {
                "source": "synthetic",
                "mode": "fourier",
                "n_cells": 16,
                "n_genes": 2,
                "sigma": 0.1,
                "k_min": 1,
                "k_max": 2,
                "dependent_xy": False,
                "poly_degree": 1,
                "seed": 42,
            }
            existence_base.write_text(
                json.dumps(
                    {
                        "data": base_data,
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(output_root / "runs"), "run_name": "existence"},
                    }
                ),
                encoding="utf-8",
            )
            perturbation_base.write_text(
                json.dumps(
                    {
                        "data": base_data,
                        "test": {
                            "method": "perturbation_test",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 42,
                            "device": "cpu",
                            "delta": [0.01, 0.05],
                            "verbose": False,
                        },
                        "output": {"out_dir": str(output_root / "runs"), "run_name": "perturbation"},
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "combined",
                        "existence_base_config": str(existence_base),
                        "perturbation_base_config": str(perturbation_base),
                        "output_root": str(output_root),
                        "k_min": 1,
                        "k_max_values": [1, 2],
                        "seeds": [42],
                        "delta_values": [0.01, 0.05],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_fourier_kmax_existence_perturbation_spec(spec_path)
            manifest = run_fourier_kmax_existence_perturbation_sweep(spec)
            analysis = analyze_fourier_kmax_existence_perturbation_results(spec_path)

            self.assertEqual(len(manifest["runs"]), 4)
            self.assertEqual(analysis["n_existence_rows_analyzed"], 2)
            self.assertEqual(analysis["n_perturbation_rows_analyzed"], 4)


if __name__ == "__main__":
    unittest.main()
