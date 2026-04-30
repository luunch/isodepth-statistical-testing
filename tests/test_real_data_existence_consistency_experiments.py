from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiment_plots import (
    save_repeat_null_boxplot,
    save_repeat_null_density_overlay,
    save_repeat_pvalue_plot,
    save_repeat_true_loss_plot,
    save_spearman_histogram,
    save_spearman_matrix_heatmap,
)
from experiments.configuration import build_run_config
from experiments.real_data_existence_consistency import (
    RealDataExistenceConsistencyStudySpec,
    build_pairwise_isodepth_rows,
    build_repeat_run_config,
    expand_real_data_existence_repeat_conditions,
    extract_repeat_payload,
    load_real_data_existence_consistency_spec,
)
from experiments.real_data_existence_consistency_analysis import (
    analyze_real_data_existence_consistency_results,
)
from experiments.real_data_existence_consistency_sweep import (
    run_real_data_existence_consistency_sweep,
)


class TestRealDataExistenceConsistencySpec(unittest.TestCase):
    def test_load_spec_validates_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            spec_path = tmp_path / "spec.json"
            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "h5ad",
                            "h5ad": "data/h5ad/example.h5ad",
                            "spatial_key": "spatial",
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 10,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 9,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {
                            "out_dir": str(tmp_path / "results"),
                            "run_name": "base",
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
                        "n_repeats": 3,
                        "repeat_seeds": [10, 11, 12],
                        "n_perms": 100,
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_real_data_existence_consistency_spec(spec_path)

            self.assertEqual(spec.experiment_name, "study")
            self.assertEqual(spec.repeat_seeds, [10, 11, 12])
            self.assertEqual(spec.n_perms, 100)
            self.assertTrue(spec.base_config.is_absolute())
            self.assertTrue(spec.output_root.is_absolute())

    def test_spec_rejects_invalid_lengths(self) -> None:
        spec = RealDataExistenceConsistencyStudySpec(
            experiment_name="study",
            base_config=REPO_ROOT / "configs/mouse_hippocampus_existence.json",
            output_root=REPO_ROOT / "results/experiments/test",
            n_repeats=2,
            repeat_seeds=[0],
            n_perms=100,
            reuse_result_roots=[],
        )
        with self.assertRaises(ValueError):
            spec.validate()

    def test_spec_rejects_non_positive_n_perms(self) -> None:
        spec = RealDataExistenceConsistencyStudySpec(
            experiment_name="study",
            base_config=REPO_ROOT / "configs/mouse_hippocampus_existence.json",
            output_root=REPO_ROOT / "results/experiments/test",
            n_repeats=1,
            repeat_seeds=[0],
            n_perms=0,
            reuse_result_roots=[],
        )
        with self.assertRaises(ValueError):
            spec.validate()

    def test_expand_conditions_and_run_config_are_deterministic(self) -> None:
        spec = RealDataExistenceConsistencyStudySpec(
            experiment_name="study",
            base_config=REPO_ROOT / "configs/mouse_hippocampus_existence.json",
            output_root=REPO_ROOT / "results/experiments/test",
            n_repeats=2,
            repeat_seeds=[7, 9],
            n_perms=100,
            reuse_result_roots=[],
        ).validate()

        conditions = expand_real_data_existence_repeat_conditions(spec)

        self.assertEqual(conditions[0].run_name, "study__repeat-000__seed-007")
        self.assertEqual(conditions[1].run_name, "study__repeat-001__seed-009")

        loaded_spec = load_real_data_existence_consistency_spec(
            REPO_ROOT / "configs/experiments/mouse_hippocampus_existence_consistency_study.json"
        )
        base_run_config = build_run_config(str(loaded_spec.base_config), {})
        run_config = build_repeat_run_config(
            base_run_config,
            spec,
            conditions[1],
        )
        self.assertEqual(run_config.test.seed, 9)
        self.assertEqual(run_config.test.n_perms, 100)
        self.assertTrue(run_config.output.run_name.endswith("__seed-009"))

    def test_cross_validation_base_config_is_accepted_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            spec_path = tmp_path / "spec.json"
            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "h5ad",
                            "h5ad": "data/h5ad/example.h5ad",
                            "spatial_key": "spatial",
                            "seed": 0,
                        },
                        "test": {
                            "method": "cross_validation",
                            "metric": "mse",
                            "n_perms": 10,
                            "train_fraction": 0.7,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 9,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {
                            "out_dir": str(tmp_path / "results"),
                            "run_name": "base",
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
                        "n_repeats": 1,
                        "repeat_seeds": [7],
                        "n_perms": 5,
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_real_data_existence_consistency_spec(spec_path)
            condition = expand_real_data_existence_repeat_conditions(spec)[0]
            run_config = build_repeat_run_config(build_run_config(str(spec.base_config), {}), spec, condition)

        self.assertEqual(run_config.test.method, "cross_validation")
        self.assertEqual(run_config.test.train_fraction, 0.7)


class TestRealDataExistenceConsistencyHelpers(unittest.TestCase):
    def test_extract_repeat_payload_expands_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "run_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "parallel_permutation",
                        "metric": "mse",
                        "p_value": 0.25,
                        "stat_true": 1.2,
                        "stat_perm": [1.5, 1.8, 1.6],
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {"source": "h5ad", "h5ad": "data/example.h5ad"},
                            "test": {"seed": 13, "n_perms": 3},
                            "output": {"run_name": "study__repeat-000__seed-013"},
                        },
                        "artifacts": {"true_isodepth": [0.0, 1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_repeat_payload(
                result_path,
                manifest_entry={"repeat_index": 0, "seed": 13, "run_name": "study__repeat-000__seed-013"},
            )

            self.assertEqual(warnings, [])
            self.assertIsNotNone(record)
            self.assertEqual(record["repeat_index"], 0)
            self.assertEqual(record["seed"], 13)
            self.assertAlmostEqual(float(record["null_mean"]), 1.6333333333)

    def test_extract_repeat_payload_accepts_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "run_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "cross_validation",
                        "metric": "mse",
                        "p_value": 0.25,
                        "stat_true": 1.2,
                        "stat_perm": [1.5, 1.8, 1.6],
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {"source": "h5ad", "h5ad": "data/example.h5ad"},
                            "test": {"seed": 13, "n_perms": 3, "train_fraction": 0.75},
                            "output": {"run_name": "study__repeat-000__seed-013"},
                        },
                        "artifacts": {"true_isodepth": [0.0, 1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            record, warnings = extract_repeat_payload(
                result_path,
                manifest_entry={"repeat_index": 0, "seed": 13, "run_name": "study__repeat-000__seed-013"},
            )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["method_name"], "cross_validation")

    def test_pairwise_rows_build_symmetric_spearman_matrix(self) -> None:
        rows = [
            {"repeat_index": 0, "seed": 0, "true_isodepth": np.asarray([0.0, 1.0, 2.0], dtype=np.float64)},
            {"repeat_index": 1, "seed": 1, "true_isodepth": np.asarray([2.0, 1.0, 0.0], dtype=np.float64)},
        ]

        pairwise_rows, matrix = build_pairwise_isodepth_rows(rows)

        self.assertEqual(len(pairwise_rows), 4)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)
        self.assertAlmostEqual(float(matrix[1, 1]), 1.0)
        self.assertAlmostEqual(float(matrix[0, 1]), -1.0)
        self.assertAlmostEqual(float(matrix[1, 0]), -1.0)

    def test_sweep_dry_run_reports_repeat_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "h5ad",
                            "h5ad": "data/h5ad/example.h5ad",
                            "spatial_key": "spatial",
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 10,
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
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec = RealDataExistenceConsistencyStudySpec(
                experiment_name="study",
                base_config=base_config,
                output_root=tmp_path / "study_outputs",
                n_repeats=2,
                repeat_seeds=[5, 6],
                n_perms=100,
                reuse_result_roots=[],
            ).validate()

            payload = run_real_data_existence_consistency_sweep(spec, dry_run=True)

            self.assertEqual(payload["planned_run_count"], 2)
            self.assertEqual(payload["planned_runs_preview"][0]["seed"], 5)


class TestRealDataExistenceConsistencyAnalysis(unittest.TestCase):
    def test_analysis_generates_csvs_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            output_root = tmp_path / "study_outputs"
            analysis_input_root = output_root / "runs" / "manual"
            analysis_input_root.mkdir(parents=True, exist_ok=True)
            spec_path = tmp_path / "spec.json"
            result_paths = []

            base_config.write_text(
                json.dumps(
                    {
                        "data": {
                            "source": "h5ad",
                            "h5ad": "data/h5ad/example.h5ad",
                            "spatial_key": "spatial",
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 100,
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
                        "n_repeats": 2,
                        "repeat_seeds": [0, 1],
                        "n_perms": 100,
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            for repeat_index, (seed, p_value, stat_true, depth) in enumerate(
                [
                    (0, 0.1, 1.0, [0.0, 1.0, 2.0, 3.0]),
                    (1, 0.2, 1.4, [0.0, 1.0, 2.0, 2.5]),
                ]
            ):
                result_path = analysis_input_root / f"repeat_{repeat_index}_result.json"
                result_path.write_text(
                    json.dumps(
                        {
                            "method_name": "parallel_permutation",
                            "metric": "mse",
                            "p_value": p_value,
                            "stat_true": stat_true,
                            "stat_perm": [1.5 + repeat_index, 1.7 + repeat_index, 1.9 + repeat_index],
                            "runtime_sec": 0.5 + repeat_index,
                            "config": {
                                "data": {"source": "h5ad", "h5ad": "data/h5ad/example.h5ad"},
                                "test": {"seed": seed, "n_perms": 100},
                                "output": {"run_name": f"study__repeat-{repeat_index:03d}__seed-{seed:03d}"},
                            },
                            "artifacts": {"true_isodepth": depth},
                        }
                    ),
                    encoding="utf-8",
                )
                result_paths.append(str(result_path.resolve()))

            (output_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {"repeat_index": 0, "seed": 0, "run_name": "study__repeat-000__seed-000", "result_json_path": result_paths[0]},
                            {"repeat_index": 1, "seed": 1, "run_name": "study__repeat-001__seed-001", "result_json_path": result_paths[1]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = analyze_real_data_existence_consistency_results(spec_path)

            self.assertEqual(payload["n_runs_analyzed"], 2)
            self.assertTrue(Path(payload["per_run_results_csv"]).exists())
            self.assertTrue(Path(payload["null_losses_long_csv"]).exists())
            self.assertTrue(Path(payload["pairwise_isodepth_spearman_csv"]).exists())
            self.assertTrue(Path(payload["true_isodepth_spearman_matrix_csv"]).exists())
            self.assertTrue(Path(payload["pvalue_by_repeat_plot"]).exists())
            self.assertTrue(Path(payload["true_isodepth_spearman_matrix_plot"]).exists())

    def test_analysis_warns_on_unreadable_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = tmp_path / "base.json"
            output_root = tmp_path / "study_outputs"
            spec_path = tmp_path / "spec.json"
            bad_result_path = output_root / "runs" / "bad_result.json"
            bad_result_path.parent.mkdir(parents=True, exist_ok=True)

            base_config.write_text(
                json.dumps(
                    {
                        "data": {"source": "h5ad", "h5ad": "data/h5ad/example.h5ad", "spatial_key": "spatial"},
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 100,
                            "epochs": 2,
                            "lr": 0.01,
                            "patience": 2,
                            "seed": 0,
                            "device": "cpu",
                            "verbose": False,
                        },
                        "output": {"out_dir": str(output_root / "runs"), "run_name": "base"},
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
                        "n_repeats": 1,
                        "repeat_seeds": [0],
                        "n_perms": 100,
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )
            bad_result_path.write_text("{not valid json", encoding="utf-8")
            (output_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {"repeat_index": 0, "seed": 0, "run_name": "study__repeat-000__seed-000", "result_json_path": str(bad_result_path.resolve())}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = analyze_real_data_existence_consistency_results(spec_path)

            self.assertEqual(payload["n_runs_analyzed"], 0)
            self.assertEqual(payload["n_warnings"], 1)

    def test_plot_helpers_save_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            per_run_rows = [
                {"repeat_index": 0, "p_value": 0.1, "stat_true": 1.0},
                {"repeat_index": 1, "p_value": 0.2, "stat_true": 1.1},
            ]
            null_rows = [
                {"repeat_index": 0, "null_loss": 1.2},
                {"repeat_index": 0, "null_loss": 1.3},
                {"repeat_index": 1, "null_loss": 1.4},
                {"repeat_index": 1, "null_loss": 1.5},
            ]
            matrix = np.asarray([[1.0, 0.8], [0.8, 1.0]], dtype=np.float64)

            self.assertTrue(save_repeat_pvalue_plot(per_run_rows, tmp_path / "pvalue.png"))
            self.assertTrue(save_repeat_true_loss_plot(per_run_rows, tmp_path / "loss.png"))
            self.assertTrue(save_repeat_null_boxplot(null_rows, per_run_rows, tmp_path / "box.png"))
            self.assertTrue(save_repeat_null_density_overlay(null_rows, per_run_rows, tmp_path / "density.png"))
            self.assertTrue(save_spearman_matrix_heatmap(matrix, tmp_path / "matrix.png"))
            self.assertTrue(save_spearman_histogram([0.8, 0.75], tmp_path / "hist.png"))


if __name__ == "__main__":
    unittest.main()
