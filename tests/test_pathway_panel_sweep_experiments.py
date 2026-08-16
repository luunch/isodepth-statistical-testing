from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.configuration import build_run_config
from experiments.studies.pathway_panel_sweep.lib import (
    build_pathway_run_config,
    expand_pathway_conditions,
    load_gmt_gene_sets,
    load_pathway_panel_sweep_spec,
)


class TestPathwayPanelSweep(unittest.TestCase):
    def test_load_gmt_has_fifty_hallmark_pathways(self) -> None:
        gmt_path = REPO_ROOT / "data/gmt/h.all.v2026.1.Hs.symbols.gmt"
        gene_sets = load_gmt_gene_sets(gmt_path)
        self.assertEqual(len(gene_sets), 50)
        self.assertIn("HALLMARK_HYPOXIA", gene_sets)
        self.assertGreaterEqual(len(gene_sets["HALLMARK_HYPOXIA"]), 15)

    def test_expand_pathway_conditions_respects_min_gene_threshold(self) -> None:
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
                            "gene_list": ["ACTB"],
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
            spec_path = tmp_path / "spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "hallmark_pathway_sweep",
                        "base_config": str(base_config),
                        "output_root": str(tmp_path / "study_outputs"),
                        "gmt_path": str(REPO_ROOT / "data/gmt/h.all.v2026.1.Hs.symbols.gmt"),
                        "n_perms": 249,
                        "n_reruns": 30,
                        "min_requested_genes": 15,
                    }
                ),
                encoding="utf-8",
            )
            spec = load_pathway_panel_sweep_spec(spec_path)
            gene_sets = load_gmt_gene_sets(spec.gmt_path)
            conditions = expand_pathway_conditions(spec, gene_sets=gene_sets)
            self.assertEqual(len(conditions), 50)
            hypoxia = next(c for c in conditions if c.pathway_name == "HALLMARK_HYPOXIA")
            run_config = build_pathway_run_config(
                build_run_config(str(spec.base_config), {}),
                spec,
                hypoxia,
            )
            self.assertEqual(run_config.test.n_perms, 249)
            self.assertEqual(run_config.test.n_reruns, 30)
            self.assertEqual(run_config.data.gene_list, hypoxia.gene_list)
            self.assertEqual(run_config.data.top_var_genes, 0)


if __name__ == "__main__":
    unittest.main()
