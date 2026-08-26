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

from experiments.configuration import build_run_config
from experiments.studies.random_gene_panel_null.lib import (
    RandomGenePanelNullStudySpec,
    build_condition_run_config,
    expand_conditions,
    load_random_gene_panel_null_spec,
    sample_random_gene_panel,
)


class TestRandomGenePanelNullSpec(unittest.TestCase):
    def _write_base_config(self, tmp_path: Path) -> Path:
        base_config = tmp_path / "base.json"
        base_config.write_text(
            json.dumps(
                {
                    "data": {
                        "source": "h5ad",
                        "h5ad": "data/h5ad/example.h5ad",
                        "spatial_key": "spatial",
                        "gene_list": ["ACTB", "VEGFA", "HK1"],
                        "top_var_genes": 0,
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
        return base_config

    def test_load_spec_validates_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = self._write_base_config(tmp_path)
            spec_path = tmp_path / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "hypoxia_panel_specificity",
                        "base_config": str(base_config),
                        "output_root": str(tmp_path / "study_outputs"),
                        "n_panels": 3,
                        "panel_size": 5,
                        "panel_seeds": [10, 11, 12],
                        "n_perms": 249,
                        "n_reruns": 30,
                    }
                ),
                encoding="utf-8",
            )

            spec = load_random_gene_panel_null_spec(spec_path)

            self.assertEqual(spec.experiment_name, "hypoxia_panel_specificity")
            self.assertEqual(spec.panel_seeds, [10, 11, 12])
            self.assertEqual(spec.n_perms, 249)
            self.assertEqual(spec.n_reruns, 30)
            self.assertTrue(spec.base_config.is_absolute())
            self.assertTrue(spec.output_root.is_absolute())

    def test_sample_random_gene_panel_is_deterministic_and_unique(self) -> None:
        eligible = [f"GENE_{index}" for index in range(100)]
        first = sample_random_gene_panel(eligible, panel_size=20, seed=7)
        second = sample_random_gene_panel(eligible, panel_size=20, seed=7)
        third = sample_random_gene_panel(eligible, panel_size=20, seed=8)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)
        self.assertNotEqual(first, third)

    def test_expand_conditions_includes_target_and_random_panels(self) -> None:
        spec = RandomGenePanelNullStudySpec(
            experiment_name="hypoxia_panel_specificity",
            base_config=Path("unused"),
            output_root=Path("unused"),
            n_panels=2,
            panel_size=3,
            panel_seeds=[0, 1],
            include_target_run=True,
        )
        eligible = [f"GENE_{index}" for index in range(20)]
        target = ["ACTB", "VEGFA", "HK1"]

        conditions = expand_conditions(spec, eligible_genes=eligible, target_gene_list=target)

        self.assertEqual(len(conditions), 3)
        self.assertEqual(conditions[0].condition_type, "target")
        self.assertEqual(conditions[0].gene_list, target)
        self.assertEqual(conditions[1].condition_type, "random_panel")
        self.assertEqual(conditions[1].panel_index, 0)
        self.assertEqual(len(conditions[1].gene_list), 3)
        self.assertEqual(len(set(conditions[1].gene_list)), 3)

    def test_build_condition_run_config_overrides_gene_list_and_test_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_config = self._write_base_config(tmp_path)
            spec = RandomGenePanelNullStudySpec(
                experiment_name="hypoxia_panel_specificity",
                base_config=base_config,
                output_root=tmp_path / "study_outputs",
                n_panels=1,
                panel_size=2,
                panel_seeds=[0],
                n_perms=249,
                n_reruns=30,
            ).validate()
            base_run_config = build_run_config(str(base_config), {})
            condition = expand_conditions(
                spec,
                eligible_genes=["A", "B", "C", "D"],
                target_gene_list=["ACTB", "VEGFA", "HK1"],
            )[1]

            run_config = build_condition_run_config(base_run_config, spec, condition)

            self.assertEqual(run_config.data.gene_list, condition.gene_list)
            self.assertEqual(run_config.test.n_perms, 249)
            self.assertEqual(run_config.test.n_reruns, 30)
            self.assertEqual(run_config.output.run_name, condition.run_name)
            self.assertEqual(run_config.output.out_dir, str((tmp_path / "study_outputs" / "runs").resolve()))


if __name__ == "__main__":
    unittest.main()
