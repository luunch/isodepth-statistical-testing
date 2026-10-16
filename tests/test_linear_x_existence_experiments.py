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
from experiments.linear_x_existence import (
    build_linear_x_existence_run_config,
    expand_linear_x_existence_conditions,
    load_linear_x_existence_spec,
)
from experiments.linear_x_existence_analysis import analyze_linear_x_existence_results
from experiments.linear_x_existence_sweep import run_linear_x_existence_sweep


STUDY_SPEC = REPO_ROOT / "configs/experiments/linear_x_existence_study.json"


class TestLinearXExistenceSpec(unittest.TestCase):
    def test_committed_spec_is_three_fractions_by_ten_seeds(self) -> None:
        spec = load_linear_x_existence_spec(STUDY_SPEC)
        conditions = expand_linear_x_existence_conditions(spec)
        self.assertEqual(len(conditions), 30)
        self.assertEqual(spec.signal_fractions, [0.01, 0.05, 0.1])
        self.assertEqual(spec.seeds, list(range(10)))

        base = build_run_config(str(spec.base_config), {})
        run = build_linear_x_existence_run_config(base, spec, conditions[11])
        self.assertEqual(run.data.mode, "linear_x")
        self.assertEqual(run.data.signal_fraction, 0.05)
        self.assertEqual(run.data.seed, 1)
        self.assertEqual(run.test.seed, 1)
        self.assertEqual(run.test.method, "correlated_null_adaptive_permutation")
        self.assertEqual(run.test.variogram_fit_mode, "variance_constrained")

    def test_dry_run_does_not_execute(self) -> None:
        spec = load_linear_x_existence_spec(STUDY_SPEC)
        payload = run_linear_x_existence_sweep(spec, spec_path=str(STUDY_SPEC), dry_run=True)
        self.assertEqual(payload["planned_run_count"], 30)
        self.assertEqual(len(payload["planned_runs_preview"]), 10)
        self.assertEqual(payload["runs"], [])

    def test_analysis_writes_pvalue_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "study_outputs"
            spec_path = Path(tmpdir) / "spec.json"
            result_path = output_root / "runs" / "manual_result.json"
            result_path.parent.mkdir(parents=True)
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "linear_x_existence_study",
                        "base_config": "configs/synthetic/linear_x_existence.json",
                        "output_root": str(output_root),
                        "signal_fractions": [0.01, 0.05],
                        "seeds": [0, 1],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(
                    {
                        "method_name": "correlated_null_adaptive_permutation",
                        "metric": "nll_gaussian_mse",
                        "p_value": 0.2,
                        "stat_true": 1.5,
                        "runtime_sec": 0.5,
                        "config": {
                            "data": {
                                "source": "synthetic",
                                "mode": "linear_x",
                                "sigma": 1.0,
                                "signal_fraction": 0.01,
                                "seed": 0,
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
                        "experiment_name": "linear_x_existence_study",
                        "runs": [
                            {
                                "run_name": "manual",
                                "seed": 0,
                                "signal_fraction": 0.01,
                                "result_json_path": str(result_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            analysis = analyze_linear_x_existence_results(spec_path)
            self.assertEqual(analysis["n_rows_analyzed"], 1)
            self.assertTrue((output_root / "analysis" / "per_run_results.csv").exists())
            self.assertTrue((output_root / "analysis" / "summary_by_signal_fraction.csv").exists())
            self.assertTrue((output_root / "analysis" / "pvalue_vs_signal_fraction.png").exists())


@unittest.skipUnless(HAS_TORCH, "torch is required for linear-x existence smoke tests")
class TestLinearXExistenceSmoke(unittest.TestCase):
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
                            "mode": "linear_x",
                            "n_cells": 16,
                            "n_genes": 4,
                            "sigma": 1.0,
                            "signal_fraction": 0.5,
                            "seed": 0,
                        },
                        "test": {
                            "method": "parallel_permutation",
                            "metric": "mse",
                            "n_perms": 2,
                            "epochs": 2,
                            "n_reruns": 1,
                            "lr": 0.01,
                            "patience": 0,
                            "seed": 0,
                            "device": "cpu",
                            "decoder": "linear",
                            "verbose": False,
                        },
                        "output": {
                            "out_dir": str(output_root / "runs"),
                            "run_name": "base",
                            "save_preds": False,
                            "save_perm_stats": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "linear_x_existence_study",
                        "base_config": str(base_config),
                        "output_root": str(output_root),
                        "signal_fractions": [0.5],
                        "seeds": [0],
                        "reuse_result_roots": [],
                    }
                ),
                encoding="utf-8",
            )

            spec = load_linear_x_existence_spec(spec_path)
            manifest = run_linear_x_existence_sweep(spec, spec_path=str(spec_path), max_runs=1)
            analysis = analyze_linear_x_existence_results(spec_path)
            self.assertEqual(len(manifest["runs"]), 1)
            self.assertEqual(analysis["n_rows_analyzed"], 1)
            self.assertTrue((output_root / "analysis" / "pvalue_vs_signal_fraction.png").exists())


if __name__ == "__main__":
    unittest.main()
