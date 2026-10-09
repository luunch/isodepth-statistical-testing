from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DatasetBundle, run_config_from_mapping
from experiments.configuration import _resolve_config_relative_paths, load_json_config


HISTORICAL_CONFIGS = (
    REPO_ROOT / "configs/hypothalamus/hypothalamus_midline_together_gaussian_linear_p499.json",
    REPO_ROOT / "configs/hypothalamus/hypothalamus_midline_separate_gaussian_linear_p499.json",
    REPO_ROOT / "configs/hippocampus/mouse_hippocampus_coordinate_separate_gaussian_linear_p499.json",
)


def _load_overlay_module():
    path = REPO_ROOT / "scripts/posthoc/overlay_blurred_perm_losses.py"
    spec = importlib.util.spec_from_file_location("overlay_blurred_perm_losses", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["overlay_blurred_perm_losses"] = module
    spec.loader.exec_module(module)
    return module


def _parse_config(path: Path):
    raw = _resolve_config_relative_paths(load_json_config(str(path)), str(path))
    return run_config_from_mapping(raw)


class TestHistoricalDatasetConfigs(unittest.TestCase):
    def test_configs_parse_with_expected_recipe(self) -> None:
        together, separate, hippocampus = (_parse_config(path) for path in HISTORICAL_CONFIGS)

        self.assertEqual(together.data.cell_type_mode, "together")
        self.assertEqual(together.data.cell_type_key, "dominant_cell_type")
        self.assertEqual(together.test.covariate.type, "midline")
        self.assertEqual(together.test.coordinate_um_per_unit, 1.0)

        self.assertEqual(separate.data.cell_type_mode, "separate")
        self.assertEqual(separate.data.min_cells_per_celltype, 30)
        self.assertEqual(separate.test.covariate.type, "midline")

        self.assertEqual(hippocampus.data.cell_type_mode, "separate")
        self.assertEqual(hippocampus.data.input_expression_representation, "log1p_normalized")
        self.assertEqual(hippocampus.data.input_normalization_target, 10000)
        self.assertTrue(hippocampus.data.make_var_names_unique)
        self.assertEqual(hippocampus.test.covariate.type, "coordinate")
        self.assertAlmostEqual(
            float(hippocampus.test.coordinate_um_per_unit), 0.2738242950835738
        )

        for cfg in (together, separate, hippocampus):
            self.assertEqual(cfg.test.method, "parallel_permutation")
            self.assertEqual(cfg.test.metric, "nll_gaussian_mse")
            self.assertEqual(cfg.test.n_perms, 499)
            self.assertEqual(cfg.test.epochs, 1000)
            self.assertEqual(cfg.test.n_reruns, 30)
            self.assertEqual(cfg.test.sgd_batch_size, 128)
            self.assertEqual(cfg.test.decoder, "linear")
            self.assertEqual(cfg.test.device, "cuda")
            self.assertTrue(cfg.data.normalize_total)
            self.assertTrue(cfg.data.log1p)
            self.assertTrue(cfg.data.standardize_expression)
            blur = cfg.blurred_permutations
            self.assertIsNotNone(blur)
            self.assertEqual(blur.sigmas_um, [10.0, 20.0, 50.0])
            self.assertEqual(blur.n_per_sigma, 20)
            self.assertTrue(blur.within_cell_type)

    def test_dry_load_retains_covariates_and_gene_cap(self) -> None:
        from data import load_dataset
        from methods.permutation import _preprocess_separate_subset

        for path in HISTORICAL_CONFIGS:
            cfg = _parse_config(path)
            h5ad = Path(cfg.data.h5ad)
            if not h5ad.is_file():
                self.skipTest(f"missing {h5ad}")
            dataset = load_dataset(cfg.data, covariate=cfg.test.covariate)
            names = list(dataset.meta.get("cell_type_names") or [])
            self.assertGreater(len(names), 0)
            if cfg.data.cell_type_mode == "together":
                self.assertLessEqual(dataset.n_genes, 3000)
                self.assertEqual(cfg.test.covariate.type, "midline")
            else:
                labels = np.asarray(dataset.meta["cell_type_labels"])
                for type_index, type_name in enumerate(names):
                    mask = labels == type_index
                    A_c, _, _ = _preprocess_separate_subset(dataset, dataset.A[mask], type_index)
                    self.assertLessEqual(A_c.shape[1], 3000, msg=type_name)
                    self.assertGreaterEqual(int(mask.sum()), cfg.data.min_cells_per_celltype)
            if path.name.startswith("mouse_hippocampus"):
                lowered = {str(name).upper() for name in names}
                self.assertTrue({"CA1", "CA2", "CA3"}.issubset(lowered) or all("CA" in n.upper() for n in names))
                self.assertIn("covariate_values", dataset.meta)
                self.assertEqual(dataset.meta.get("covariate_obs_key"), "coordinate")
                self.assertTrue(dataset.meta.get("restored_linear_expression"))
                self.assertTrue(dataset.meta.get("make_var_names_unique"))


class TestOverlayDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.overlay = _load_overlay_module()

    def test_from_config_reads_blur_block(self) -> None:
        cfg = _parse_config(HISTORICAL_CONFIGS[0])
        settings = self.overlay.blur_settings_from_run_config(cfg)
        self.assertEqual(settings["sigmas_um"], [10.0, 20.0, 50.0])
        self.assertEqual(settings["n_blur"], 20)
        self.assertTrue(settings["within_cell_type"])

    def test_together_uses_shared_encoder_and_within_type_blur(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {
                    "source": "h5ad",
                    "h5ad": "data/example.h5ad",
                    "cell_type": "together",
                    "cell_type_key": "cell_type",
                },
                "test": {
                    "method": "parallel_permutation",
                    "metric": "nll_gaussian_mse",
                    "n_perms": 2,
                    "seed": 42,
                    "coordinate_um_per_unit": 1.0,
                },
                "output": {"out_dir": "results/tmp", "run_name": "together_run"},
            }
        )
        dataset = DatasetBundle(
            S=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            A=np.ones((4, 3), dtype=np.float32),
            meta={
                "cell_type_labels": np.array([0, 0, 1, 1], dtype=np.int64),
                "cell_type_names": ["a", "b"],
                "n_cell_types": 2,
                "cell_type_mode": "together",
                "coord_mean": np.zeros(2, dtype=np.float32),
                "coord_std": np.ones(2, dtype=np.float32),
            },
        ).validate()
        payload = {
            "method_name": "parallel_permutation",
            "metric": "nll_gaussian_mse",
            "p_value": 0.02,
            "stat_true": 1.0,
            "stat_perm": [2.0, 3.0],
            "n_cells": 4,
            "n_genes": 3,
        }
        targets = self.overlay.build_overlay_targets(
            cfg, payload, dataset=dataset, within_cell_type=True
        )
        self.assertEqual(len(targets), 1)
        self.assertIsNotNone(targets[0].training_cell_type_labels)
        self.assertEqual(targets[0].n_cell_types, 2)
        np.testing.assert_array_equal(targets[0].blur_group_labels, [0, 0, 1, 1])

    def test_separate_builds_one_target_per_type(self) -> None:
        cfg = run_config_from_mapping(
            {
                "data": {
                    "source": "h5ad",
                    "h5ad": "data/example.h5ad",
                    "cell_type": "separate",
                    "cell_type_key": "cell_type",
                },
                "test": {
                    "method": "parallel_permutation",
                    "metric": "nll_gaussian_mse",
                    "n_perms": 2,
                    "seed": 7,
                    "coordinate_um_per_unit": 1.0,
                },
                "output": {"out_dir": "results/tmp", "run_name": "separate_run"},
            }
        )
        dataset = DatasetBundle(
            S=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            A=np.ones((4, 3), dtype=np.float32),
            meta={
                "cell_type_labels": np.array([0, 0, 1, 1], dtype=np.int64),
                "cell_type_names": ["CA1", "CA3"],
                "n_cell_types": 2,
                "cell_type_mode": "separate",
                "coord_mean": np.zeros(2, dtype=np.float32),
                "coord_std": np.ones(2, dtype=np.float32),
            },
        ).validate()
        payload = {
            "method_name": "parallel_permutation",
            "metric": "nll_gaussian_mse",
            "p_value": 0.04,
            "stat_true": 1.0,
            "stat_perm": [2.0, 3.0],
            "n_cells": 4,
            "n_genes": 3,
            "artifacts": {
                "cell_type_names": ["CA1", "CA3"],
                "per_type_summaries": {
                    "CA1": {
                        "p_value": 0.02,
                        "stat_true": 1.1,
                        "stat_perm": [2.1, 3.1],
                        "n_cells": 2,
                    },
                    "CA3": {
                        "p_value": 0.04,
                        "stat_true": 1.2,
                        "stat_perm": [2.2, 3.2],
                        "n_cells": 2,
                    },
                },
            },
        }
        targets = self.overlay.build_overlay_targets(
            cfg, payload, dataset=dataset, within_cell_type=True
        )
        self.assertEqual([target.label for target in targets], ["CA1", "CA3"])
        self.assertIsNone(targets[0].training_cell_type_labels)
        self.assertIsNone(targets[0].blur_group_labels)
        self.assertEqual(targets[0].seed, 7)
        self.assertEqual(targets[1].seed, 8)
        self.assertEqual(targets[0].dataset.n_cells, 2)
        self.assertEqual(targets[1].dataset.n_cells, 2)
        self.assertTrue(str(targets[0].plot_path).endswith("CA1/CA1_metric_distribution.png"))

    def test_separate_combined_plot_gets_blur_overlays(self) -> None:
        import json
        import tempfile

        from data.schemas import run_config_from_mapping

        cfg = run_config_from_mapping(
            {
                "data": {
                    "source": "h5ad",
                    "h5ad": "data/example.h5ad",
                    "cell_type": "separate",
                    "cell_type_key": "cell_type",
                },
                "test": {
                    "method": "parallel_permutation",
                    "metric": "nll_gaussian_mse",
                    "covariate": "coordinate",
                    "n_perms": 2,
                    "seed": 7,
                    "coordinate_um_per_unit": 1.0,
                },
                "output": {"out_dir": "results/tmp", "run_name": "separate_run"},
            }
        )
        payload = {
            "method_name": "parallel_permutation",
            "metric": "nll_gaussian_mse",
            "p_value": 0.04,
            "stat_true": 0.99,
            "stat_perm": [1.01, 1.02],
            "n_cells": 4,
            "n_genes": 3,
            "artifacts": {
                "cell_type_names": ["CA1", "CA3"],
                "per_type_summaries": {
                    "CA1": {
                        "p_value": 0.02,
                        "stat_true": 0.99,
                        "stat_perm": [1.01, 1.02, 1.03],
                        "n_cells": 2,
                        "stat_covariate": 0.995,
                        "p_value_covariate": 0.02,
                    },
                    "CA3": {
                        "p_value": 0.04,
                        "stat_true": 1.00,
                        "stat_perm": [1.02, 1.03, 1.04],
                        "n_cells": 2,
                    },
                },
            },
        }
        OverlayTarget = self.overlay.OverlayTarget
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "separate_run"
            cfg.output.out_dir = str(Path(tmp))
            cfg.output.run_name = "separate_run"
            targets = []
            for name, losses in (
                ("CA1", [0.991, 0.992]),
                ("CA3", [1.001, 1.002]),
            ):
                type_dir = run_dir / name
                type_dir.mkdir(parents=True)
                sidecar = {
                    "overlays": [
                        {
                            "sigma_um": 10.0,
                            "stat_blurred_perm": losses,
                        },
                        {
                            "sigma_um": 20.0,
                            "stat_blurred_perm": [x + 0.002 for x in losses],
                        },
                        {
                            "sigma_um": 50.0,
                            "stat_blurred_perm": [x + 0.004 for x in losses],
                        },
                    ]
                }
                sidecar_path = type_dir / f"{name}_blurred_perm_losses.json"
                sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
                targets.append(
                    OverlayTarget(
                        label=name,
                        plot_path=type_dir / f"{name}_metric_distribution.png",
                        sidecar_path=sidecar_path,
                        metrics=self.overlay._payload_metrics(
                            payload,
                            type_summary=payload["artifacts"]["per_type_summaries"][name],
                        ),
                        seed=7,
                    )
                )
            combined = self.overlay._write_combined_separate_overlay_plot(
                cfg, payload, targets, [10.0, 20.0, 50.0]
            )
            self.assertIsNotNone(combined)
            self.assertTrue(combined.is_file())
            self.assertGreater(combined.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
