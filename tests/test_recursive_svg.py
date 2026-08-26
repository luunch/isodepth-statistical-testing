"""Tests for recursive SVG gradient detection (experiments/recursive_svg.py)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schemas import DataConfig, DatasetBundle, OutputConfig, RunConfig, TestConfig

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    from experiments.recursive_svg import (
        _gene_names_from_meta,
        _subset_dataset,
        _write_combined_sig_genes_csv,
        run_recursive_svg,
    )
    from analysis.plots import compute_isodepth_sig_genes


# ---------------------------------------------------------------------------
# Schema-level validation tests (no torch required)
# ---------------------------------------------------------------------------

class TestRecursiveSchemaValidation(unittest.TestCase):
    def test_recursive_rejected_for_nn_decoder(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            TestConfig(
                method="parallel_permutation",
                decoder="nn",
                recursive=True,
            ).validate()
        self.assertIn("nn", str(ctx.exception))
        self.assertIn("recursive", str(ctx.exception).lower())

    def test_recursive_rejected_for_non_parallel_permutation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            TestConfig(
                method="cross_validation",
                decoder="linear",
                recursive=True,
                n_perms=2,
            ).validate()
        self.assertIn("parallel_permutation", str(ctx.exception))

    def test_recursive_valid_with_linear_decoder(self) -> None:
        config = TestConfig(
            method="parallel_permutation",
            decoder="linear",
            recursive=True,
            n_perms=2,
        )
        self.assertIs(config.validate(), config)

    def test_recursive_valid_with_quadratic_decoder(self) -> None:
        config = TestConfig(
            method="parallel_permutation",
            decoder="quadratic",
            recursive=True,
            n_perms=2,
        )
        self.assertIs(config.validate(), config)

    def test_recursive_false_is_default(self) -> None:
        config = TestConfig()
        self.assertFalse(config.recursive)

    def test_max_gradients_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(recursive=False, max_gradients=0).validate()

    def test_alpha_already_validated(self) -> None:
        with self.assertRaises(ValueError):
            TestConfig(alpha=0.0).validate()
        with self.assertRaises(ValueError):
            TestConfig(alpha=1.0).validate()


# ---------------------------------------------------------------------------
# Unit tests for helper functions (no torch required)
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def _make_dataset(self, n_cells: int = 20, n_genes: int = 5) -> DatasetBundle:
        rng = np.random.default_rng(0)
        S = rng.random((n_cells, 2)).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        var_names = [f"gene_{i}" for i in range(n_genes)]
        return DatasetBundle(S=S, A=A, meta={"var_names": var_names}).validate()

    def test_subset_dataset_reduces_columns(self) -> None:
        dataset = self._make_dataset(n_genes=6)
        indices = np.array([0, 2, 4], dtype=np.intp)
        sub = _subset_dataset(dataset, indices)
        self.assertEqual(sub.n_genes, 3)
        self.assertEqual(sub.n_cells, dataset.n_cells)
        np.testing.assert_allclose(
            sub.A, np.asarray(dataset.A, dtype=np.float32)[:, indices]
        )

    def test_subset_dataset_updates_var_names(self) -> None:
        dataset = self._make_dataset(n_genes=4)
        sub = _subset_dataset(dataset, np.array([1, 3]))
        self.assertEqual(sub.meta["var_names"], ["gene_1", "gene_3"])

    def test_subset_dataset_without_var_names(self) -> None:
        rng = np.random.default_rng(1)
        S = rng.random((10, 2)).astype(np.float32)
        A = rng.standard_normal((10, 4)).astype(np.float32)
        dataset = DatasetBundle(S=S, A=A, meta={}).validate()
        sub = _subset_dataset(dataset, np.array([0, 2]))
        self.assertEqual(sub.n_genes, 2)
        self.assertIsNone(sub.meta.get("var_names"))

    def test_gene_names_from_meta_with_names(self) -> None:
        meta = {"var_names": ["a", "b", "c"]}
        names = _gene_names_from_meta(meta, 3)
        self.assertEqual(names, ["a", "b", "c"])

    def test_gene_names_from_meta_without_names(self) -> None:
        names = _gene_names_from_meta({}, 3)
        self.assertEqual(names, ["gene_0", "gene_1", "gene_2"])

    def test_write_combined_sig_genes_csv_columns(self) -> None:
        entries = [
            {
                "gradient_idx": 1,
                "gene_names": ["a", "b", "c"],
                "pvalues": np.array([0.001, 0.01, 0.5]),
                "qvalues": np.array([0.003, 0.015, 0.5]),
                "sig_indices": np.array([0, 1]),
            },
            {
                "gradient_idx": 2,
                "gene_names": ["c"],
                "pvalues": np.array([0.02]),
                "qvalues": np.array([0.04]),
                "sig_indices": np.array([0]),
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "combined.csv"
            _write_combined_sig_genes_csv(out_path, entries)
            with open(out_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            headers = list(rows[0].keys())
            self.assertIn("gene", headers)
            self.assertIn("p_value", headers)
            self.assertIn("q_value", headers)
            self.assertIn("corresponding_gradient", headers)
            # gradient 1 has 2 sig genes, gradient 2 has 1 → 3 rows
            self.assertEqual(len(rows), 3)
            gradients = [int(r["corresponding_gradient"]) for r in rows]
            self.assertIn(1, gradients)
            self.assertIn(2, gradients)

    def test_write_combined_sig_genes_csv_sorted_within_gradient(self) -> None:
        entries = [
            {
                "gradient_idx": 1,
                "gene_names": ["x", "y"],
                "pvalues": np.array([0.05, 0.001]),
                "qvalues": np.array([0.05, 0.002]),
                "sig_indices": np.array([0, 1]),
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "combined.csv"
            _write_combined_sig_genes_csv(out_path, entries)
            with open(out_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            # Lower p-value (y, 0.001) should come first
            self.assertEqual(rows[0]["gene"], "y")
            self.assertEqual(rows[1]["gene"], "x")


@unittest.skipUnless(HAS_TORCH, "torch is required for compute_isodepth_sig_genes tests")
class TestComputeIsodepthSigGenes(unittest.TestCase):
    """Tests for the extracted SVG helper function."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.n_cells = 80
        self.n_genes = 10
        self.coord = np.linspace(0, 1, self.n_cells)
        # First 3 genes strongly follow coord; rest are pure noise
        noise = rng.standard_normal((self.n_cells, self.n_genes))
        A = noise.copy()
        for g in range(3):
            A[:, g] = 5.0 * self.coord + 0.1 * rng.standard_normal(self.n_cells)
        self.A = A
        self.gene_names = [f"gene_{i}" for i in range(self.n_genes)]

    def test_returns_required_keys(self) -> None:
        result = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.05,
        )
        for key in ("sig_indices", "sig_names", "pvalues", "qvalues"):
            self.assertIn(key, result)

    def test_pvalues_shape(self) -> None:
        result = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.05,
        )
        self.assertEqual(result["pvalues"].shape, (self.n_genes,))

    def test_spatially_varying_genes_are_significant(self) -> None:
        result = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.05,
        )
        sig = set(result["sig_indices"].tolist())
        # First 3 genes should be significant
        for g in range(3):
            self.assertIn(g, sig, f"gene_{g} should be significant")

    def test_pred_isodepth_overrides_polynomial_fit(self) -> None:
        preds = np.column_stack([
            np.poly1d(np.polyfit(self.coord, self.A[:, g], 1))(self.coord)
            for g in range(self.n_genes)
        ])
        result_preds = compute_isodepth_sig_genes(
            self.A, self.gene_names, preds, 1, coord=self.coord, alpha=0.05,
        )
        result_poly = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.05,
        )
        np.testing.assert_allclose(
            result_preds["pvalues"], result_poly["pvalues"], rtol=1e-5,
        )

    def test_alpha_controls_threshold(self) -> None:
        result_strict = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.001,
        )
        result_lenient = compute_isodepth_sig_genes(
            self.A, self.gene_names, None, 1, coord=self.coord, alpha=0.5,
        )
        self.assertLessEqual(
            len(result_strict["sig_indices"]),
            len(result_lenient["sig_indices"]),
        )

    def test_raises_without_coord_and_preds(self) -> None:
        with self.assertRaises(ValueError):
            compute_isodepth_sig_genes(
                self.A, self.gene_names, None, 1, coord=None, alpha=0.05,
            )


# ---------------------------------------------------------------------------
# Integration tests for run_recursive_svg (mock permutation method)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_TORCH, "torch is required for recursive SVG integration tests")
class TestRunRecursiveSvg(unittest.TestCase):
    """Integration tests using a mocked run_permutation_method to control outcomes."""

    def _make_run_config(self, tmpdir: str, n_perms: int = 2) -> RunConfig:
        return RunConfig(
            data=DataConfig(source="synthetic", n_cells=50, n_genes=8),
            test=TestConfig(
                method="parallel_permutation",
                decoder="linear",
                recursive=True,
                alpha=0.05,
                max_gradients=5,
                n_perms=n_perms,
                n_reruns=1,
                epochs=2,
                verbose=False,
                device="cpu",
                seed=0,
            ).validate(),
            output=OutputConfig(
                out_dir=tmpdir,
                run_name="test_run",
            ),
        )

    def _make_sig_result(
        self,
        n_cells: int,
        n_genes: int,
        n_sig: int,
        p_value: float = 0.01,
        seed: int = 0,
    ):
        """Build a fake TestResult with an isodepth and pred_true that make
        the first n_sig genes look spatially varying."""
        from data.schemas import TestResult
        rng = np.random.default_rng(seed)
        coord = np.linspace(0, 1, n_cells)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float64)
        for g in range(n_sig):
            A[:, g] = 5.0 * coord + 0.05 * rng.standard_normal(n_cells)
        # pred_true: polynomial fits for each gene
        pred_true = np.stack(
            [np.poly1d(np.polyfit(coord, A[:, g], 1))(coord) for g in range(n_genes)],
            axis=1,
        ).astype(np.float32)
        S = np.zeros((n_cells, 2), dtype=np.float32)
        return TestResult(
            method_name="parallel_permutation",
            metric="nll_gaussian_mse",
            p_value=p_value,
            stat_true=0.5,
            stat_perm=np.array([1.0, 1.1, 1.2]),
            runtime_sec=0.1,
            n_cells=n_cells,
            n_genes=n_genes,
            config={},
            artifacts={
                "true_isodepth": coord.astype(np.float32),
                "pred_true": pred_true,
                "lowest_isodepth": coord.astype(np.float32),
                "lowest_S": S,
                "lowest_stat": 1.0,
                "lowest_perm_index": 0,
                "lowest_rerun_index": 0,
                "lowest_train_loss": 0.0,
                "highest_isodepth": coord.astype(np.float32),
                "highest_S": S,
                "highest_stat": 1.2,
                "highest_perm_index": 2,
                "highest_rerun_index": 0,
                "highest_train_loss": 0.0,
            },
        ).validate()

    def _make_dataset(self, n_cells: int = 50, n_genes: int = 8, seed: int = 0) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        S = rng.random((n_cells, 2)).astype(np.float32)
        coord = np.linspace(0, 1, n_cells)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        for g in range(4):
            A[:, g] = 5.0 * coord.astype(np.float32) + 0.05 * rng.standard_normal(n_cells).astype(np.float32)
        var_names = [f"gene_{i}" for i in range(n_genes)]
        return DatasetBundle(S=S, A=A, meta={"var_names": var_names}).validate()

    def test_stops_when_not_significant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config(tmpdir)
            dataset = self._make_dataset()
            # Always return a non-significant result
            non_sig = self._make_sig_result(50, 8, n_sig=0, p_value=0.5)
            with patch(
                "experiments.recursive_svg.run_permutation_method",
                return_value=non_sig,
            ):
                payload, summary_path = run_recursive_svg(dataset, run_config)
            self.assertEqual(payload["n_gradients_found"], 0)
            # No combined CSV should be written
            combined = Path(tmpdir) / "test_run" / "recursive" / "combined_sig_genes.csv"
            self.assertFalse(combined.exists())

    def test_nn_decoder_raises_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Bypass schema validation by not calling validate() after changing decoder
            run_config = self._make_run_config(tmpdir)
            run_config.test.decoder = "nn"  # force nn after validate
            dataset = self._make_dataset()
            with self.assertRaises(ValueError) as ctx:
                run_recursive_svg(dataset, run_config)
            self.assertIn("nn", str(ctx.exception))

    def test_one_gradient_produces_expected_files(self) -> None:
        n_cells, n_genes = 50, 8
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config(tmpdir)
            dataset = self._make_dataset(n_cells=n_cells, n_genes=n_genes)
            sig_result = self._make_sig_result(n_cells, n_genes, n_sig=3, p_value=0.01, seed=0)
            non_sig_result = self._make_sig_result(n_cells, n_genes - 3, n_sig=0, p_value=0.5, seed=1)
            call_count = 0
            def mock_run(ds, cfg):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return sig_result
                return non_sig_result
            with patch("experiments.recursive_svg.run_permutation_method", side_effect=mock_run):
                payload, summary_path = run_recursive_svg(dataset, run_config)

            self.assertEqual(payload["n_gradients_found"], 1)
            recursive_dir = Path(tmpdir) / "test_run" / "recursive"
            # Combined CSV exists and has correct columns
            combined_csv = recursive_dir / "combined_sig_genes.csv"
            self.assertTrue(combined_csv.exists())
            with open(combined_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertGreater(len(rows), 0)
            self.assertIn("corresponding_gradient", rows[0])
            self.assertIn("gene", rows[0])
            self.assertIn("p_value", rows[0])
            self.assertIn("q_value", rows[0])

            # gradient_1 directory exists
            gradient_dir = recursive_dir / "gradient_1"
            self.assertTrue(gradient_dir.exists())

            # result JSON exists and is valid
            result_json = gradient_dir / "gradient_1_result.json"
            self.assertTrue(result_json.exists())
            with open(result_json) as fh:
                result_data = json.load(fh)
            self.assertIn("p_value", result_data)
            artifacts = result_data.get("artifacts", {})
            self.assertIn("svg_genes_plot", artifacts)
            self.assertTrue((gradient_dir / "gradient_1_svg_genes.png").exists())
            self.assertNotIn("top_genes_plot", artifacts)

            # null distribution NPY exists
            null_npy = gradient_dir / "gradient_1_null_distribution.npy"
            self.assertTrue(null_npy.exists())
            null_arr = np.load(null_npy)
            np.testing.assert_allclose(null_arr, sig_result.stat_perm)

            # summary JSON contains expected keys
            with open(summary_path) as fh:
                summary = json.load(fh)
            self.assertIn("n_gradients_found", summary)
            self.assertIn("gradients", summary)
            self.assertIn("combined_sig_genes_csv", summary)
            self.assertIn("svg_count_plot", summary)
            self.assertTrue((recursive_dir / "svg_counts_by_gradient.png").exists())

    def test_recursive_svg_detection_uses_saved_isodepth_not_decoder_predictions(self) -> None:
        from data.schemas import TestResult

        n_cells, n_genes = 80, 4
        coord = np.linspace(0.0, 1.0, n_cells)
        rng = np.random.default_rng(123)
        S = rng.random((n_cells, 2)).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        A[:, 1] = (4.0 * coord + 0.03 * rng.standard_normal(n_cells)).astype(np.float32)
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={"var_names": [f"gene_{i}" for i in range(n_genes)]},
        ).validate()

        misleading_preds = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        misleading_preds[:, 0] = (6.0 * coord).astype(np.float32)
        result = TestResult(
            method_name="parallel_permutation",
            metric="nll_gaussian_mse",
            p_value=0.01,
            stat_true=0.5,
            stat_perm=np.array([1.0, 1.1, 1.2]),
            runtime_sec=0.1,
            n_cells=n_cells,
            n_genes=n_genes,
            config={},
            artifacts={
                "true_isodepth": coord.astype(np.float32),
                "pred_true": misleading_preds,
                "lowest_isodepth": coord.astype(np.float32),
                "lowest_S": S,
                "lowest_stat": 1.0,
                "lowest_perm_index": 0,
                "highest_isodepth": coord.astype(np.float32),
                "highest_S": S,
                "highest_stat": 1.2,
                "highest_perm_index": 2,
            },
        ).validate()
        non_sig = self._make_sig_result(n_cells, n_genes - 1, n_sig=0, p_value=0.5, seed=22)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config(tmpdir)
            results = iter([result, non_sig])
            with patch("experiments.recursive_svg.run_permutation_method", side_effect=lambda ds, cfg: next(results)):
                payload, _ = run_recursive_svg(dataset, run_config)

            self.assertEqual(payload["n_gradients_found"], 1)
            csv_path = Path(tmpdir) / "test_run" / "recursive" / "combined_sig_genes.csv"
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            genes = {row["gene"] for row in rows}
            self.assertIn("gene_1", genes)
            self.assertNotIn("gene_0", genes)

    def test_each_gene_appears_at_most_once_in_combined_csv(self) -> None:
        from data.schemas import TestResult

        n_cells, n_genes = 80, 8
        rng = np.random.default_rng(7)
        coord1 = np.linspace(0.0, 1.0, n_cells)
        coord2 = np.cos(2.0 * np.pi * coord1)
        S = rng.random((n_cells, 2)).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        for g in range(3):
            A[:, g] = (6.0 * coord1 + 0.03 * rng.standard_normal(n_cells)).astype(np.float32)
        for g in range(3, 5):
            A[:, g] = (6.0 * coord2 + 0.03 * rng.standard_normal(n_cells)).astype(np.float32)
        dataset = DatasetBundle(
            S=S,
            A=A,
            meta={"var_names": [f"gene_{i}" for i in range(n_genes)]},
        ).validate()

        def make_result(coord: np.ndarray, n_genes_in: int, p_value: float) -> TestResult:
            preds = np.zeros((n_cells, n_genes_in), dtype=np.float32)
            return TestResult(
                method_name="parallel_permutation",
                metric="nll_gaussian_mse",
                p_value=p_value,
                stat_true=0.5,
                stat_perm=np.array([1.0, 1.1, 1.2]),
                runtime_sec=0.1,
                n_cells=n_cells,
                n_genes=n_genes_in,
                config={},
                artifacts={
                    "true_isodepth": coord.astype(np.float32),
                    "pred_true": preds,
                    "lowest_isodepth": coord.astype(np.float32),
                    "lowest_S": S,
                    "lowest_stat": 1.0,
                    "lowest_perm_index": 0,
                    "highest_isodepth": coord.astype(np.float32),
                    "highest_S": S,
                    "highest_stat": 1.2,
                    "highest_perm_index": 2,
                },
            ).validate()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config(tmpdir)
            result1 = make_result(coord1, n_genes, 0.01)
            result2 = make_result(coord2, n_genes - 3, 0.01)
            result3 = make_result(coord1, n_genes - 5, 0.5)
            results = iter([result1, result2, result3])

            with patch(
                "experiments.recursive_svg.run_permutation_method",
                side_effect=lambda ds, cfg: next(results),
            ):
                payload, _ = run_recursive_svg(dataset, run_config)

            self.assertEqual(payload["n_gradients_found"], 2)
            combined_csv = Path(tmpdir) / "test_run" / "recursive" / "combined_sig_genes.csv"
            with open(combined_csv, newline="") as fh:
                rows = list(csv.DictReader(fh))
            gene_names_in_csv = [r["gene"] for r in rows]
            # No duplicates — primary invariant
            self.assertEqual(len(gene_names_in_csv), len(set(gene_names_in_csv)))
            # Both gradients contributed their intended SVG groups
            gradients_present = {int(r["corresponding_gradient"]) for r in rows}
            self.assertIn(1, gradients_present)
            self.assertIn(2, gradients_present)
            grad1_genes = {r["gene"] for r in rows if r["corresponding_gradient"] == "1"}
            grad2_genes = {r["gene"] for r in rows if r["corresponding_gradient"] == "2"}
            self.assertEqual(grad1_genes, {"gene_0", "gene_1", "gene_2"})
            self.assertEqual(grad2_genes, {"gene_3", "gene_4"})

    def test_max_gradients_limits_iterations(self) -> None:
        n_cells, n_genes = 50, 8
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config(tmpdir)
            # Limit to 2 gradients
            run_config.test.max_gradients = 2
            dataset = self._make_dataset(n_cells=n_cells, n_genes=n_genes)

            # Always return 1 sig gene per round (so we'd run forever without cap)
            def make_one_sig(n_g: int) -> "TestResult":
                return self._make_sig_result(n_cells, max(n_g, 1), n_sig=min(1, n_g), p_value=0.01)

            call_count = 0
            n_remaining = [n_genes]
            def mock_run(ds, cfg):
                nonlocal call_count
                r = make_one_sig(n_remaining[0])
                call_count += 1
                n_remaining[0] = max(n_remaining[0] - 1, 0)
                return r

            with patch("experiments.recursive_svg.run_permutation_method", side_effect=mock_run):
                payload, _ = run_recursive_svg(dataset, run_config)

            self.assertLessEqual(payload["n_gradients_found"], 2)


# ---------------------------------------------------------------------------
# Integration tests for run_recursive_svg in separate cell-type mode
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_TORCH, "torch is required for recursive SVG separate mode tests")
class TestRunRecursiveSvgSeparate(unittest.TestCase):
    """Tests for recursive SVG with cell_type='separate' mode."""

    def _make_run_config_separate(self, tmpdir: str) -> RunConfig:
        data_cfg = DataConfig(source="synthetic", n_cells=50, n_genes=8)
        data_cfg.cell_type = "separate"
        return RunConfig(
            data=data_cfg,
            test=TestConfig(
                method="parallel_permutation",
                decoder="linear",
                recursive=True,
                alpha=0.05,
                max_gradients=3,
                n_perms=2,
                n_reruns=1,
                epochs=2,
                verbose=False,
                device="cpu",
                seed=0,
            ).validate(),
            output=OutputConfig(out_dir=tmpdir, run_name="test_sep_run"),
        )

    def _make_type_data(
        self,
        n_cells: int,
        n_genes: int,
        n_sig: int = 3,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        coord = np.linspace(0, 1, n_cells)
        S = rng.random((n_cells, 2)).astype(np.float32)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float64)
        for g in range(min(n_sig, n_genes)):
            A[:, g] = 5.0 * coord + 0.05 * rng.standard_normal(n_cells)
        return S, A.astype(np.float32)

    def _make_dataset(self, type_arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> DatasetBundle:
        names = list(type_arrays.keys())
        S = np.vstack([type_arrays[name][0] for name in names]).astype(np.float32)
        A = np.vstack([type_arrays[name][1] for name in names]).astype(np.float32)
        labels = np.concatenate([
            np.full(type_arrays[name][0].shape[0], i, dtype=np.int64)
            for i, name in enumerate(names)
        ])
        return DatasetBundle(
            S=S,
            A=A,
            meta={
                "var_names": [f"gene_{i}" for i in range(A.shape[1])],
                "cell_type_labels": labels,
                "cell_type_names": names,
                "n_cell_types": len(names),
                "cell_type_mode": "separate",
            },
        ).validate()

    def _make_result(self, A: np.ndarray, p_value: float, seed: int = 0):
        from data.schemas import TestResult
        rng = np.random.default_rng(seed)
        n_cells, n_genes = A.shape
        coord = np.linspace(0, 1, n_cells)
        pred_true = np.stack(
            [np.poly1d(np.polyfit(coord, A[:, g].astype(np.float64), 1))(coord)
             for g in range(n_genes)],
            axis=1,
        ).astype(np.float32)
        S = rng.random((n_cells, 2)).astype(np.float32)
        return TestResult(
            method_name="parallel_permutation",
            metric="nll_gaussian_mse",
            p_value=p_value,
            stat_true=0.5,
            stat_perm=np.array([1.0, 1.1, 1.2]),
            runtime_sec=0.1,
            n_cells=n_cells,
            n_genes=n_genes,
            config={},
            artifacts={
                "true_isodepth": coord.astype(np.float32),
                "pred_true": pred_true,
                "lowest_isodepth": coord.astype(np.float32),
                "lowest_S": S,
                "lowest_stat": 1.0,
                "lowest_perm_index": 0,
                "lowest_rerun_index": 0,
                "lowest_train_loss": 0.0,
                "highest_isodepth": coord.astype(np.float32),
                "highest_S": S,
                "highest_stat": 1.2,
                "highest_perm_index": 2,
                "highest_rerun_index": 0,
                "highest_train_loss": 0.0,
            },
        ).validate()

    def _make_noise_result(self, n_cells: int, n_genes: int, p_value: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        return self._make_result(A, p_value=p_value, seed=seed + 100)

    def _fake_save_gradient_outputs(
        self,
        gradient_idx,
        dataset,
        result,
        svg_info,
        gradient_dir,
        decoder_df,
        alpha,
        **kwargs,
    ):
        gradient_dir.mkdir(parents=True, exist_ok=True)
        result_json = gradient_dir / f"gradient_{gradient_idx}_result.json"
        result_json.write_text(json.dumps({"p_value": float(result.p_value)}))
        return {"result_json": str(result_json)}

    def _patch_plot_writers(self):
        return (
            patch("experiments.recursive_svg._save_gradient_outputs", side_effect=self._fake_save_gradient_outputs),
            patch("experiments.recursive_svg.save_celltype_dataset_plot", side_effect=lambda dataset, path: Path(path)),
            patch("experiments.recursive_svg.save_recursive_celltype_isodepth_grid", side_effect=lambda *args, **kwargs: Path(args[2])),
            patch("experiments.recursive_svg.save_recursive_celltype_metric_distribution_grid", side_effect=lambda *args, **kwargs: Path(args[2])),
        )

    def test_each_type_gets_direct_gradient_folders_including_terminal(self) -> None:
        n_cells_a, n_cells_b, n_genes = 60, 20, 8
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config_separate(tmpdir)
            S_a, A_a = self._make_type_data(n_cells_a, n_genes, n_sig=3, seed=0)
            S_b, A_b = self._make_type_data(n_cells_b, n_genes, n_sig=0, seed=5)
            dataset = self._make_dataset({"TypeA": (S_a, A_a), "TypeB": (S_b, A_b)})

            iter_a1 = self._make_result(A_a, p_value=0.01, seed=10)
            iter_a2 = self._make_noise_result(n_cells_a, n_genes - 3, p_value=0.5, seed=11)
            iter_b1 = self._make_noise_result(n_cells_b, n_genes, p_value=0.5, seed=12)
            results = iter([iter_a1, iter_a2, iter_b1])

            p1, p2, p3, p4 = self._patch_plot_writers()
            with patch("experiments.recursive_svg.run_permutation_method", side_effect=lambda ds, cfg: next(results)), p1, p2, p3, p4:
                run_recursive_svg(dataset, run_config)

            out_dir = Path(tmpdir) / "test_sep_run"
            self.assertTrue((out_dir / "TypeA" / "gradient_1").exists())
            self.assertTrue((out_dir / "TypeA" / "gradient_2").exists())
            self.assertFalse((out_dir / "TypeA" / "recursive").exists())
            self.assertTrue((out_dir / "TypeB" / "gradient_1").exists())
            self.assertFalse((out_dir / "TypeB" / "recursive").exists())

            with open(out_dir / "TypeA" / "recursive_summary.json") as fh:
                type_a_summary = json.load(fh)
            self.assertEqual(type_a_summary["n_gradients_found"], 1)
            self.assertEqual(type_a_summary["n_tested_gradients"], 2)
            self.assertEqual(type_a_summary["tested_gradients"][-1]["passed_permutation"], False)

            with open(out_dir / "TypeB" / "recursive_summary.json") as fh:
                type_b_summary = json.load(fh)
            self.assertEqual(type_b_summary["n_gradients_found"], 0)
            self.assertEqual(type_b_summary["n_tested_gradients"], 1)

    def test_top_level_summary_reports_all_regions_as_processed(self) -> None:
        n_cells, n_genes = 40, 6
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config_separate(tmpdir)
            S_a, A_a = self._make_type_data(n_cells, n_genes, n_sig=2, seed=0)
            S_b, A_b = self._make_type_data(n_cells, n_genes, n_sig=0, seed=5)
            dataset = self._make_dataset({"TypeA": (S_a, A_a), "TypeB": (S_b, A_b)})

            iter_a1 = self._make_result(A_a, p_value=0.01, seed=10)
            iter_a2 = self._make_noise_result(n_cells, n_genes - 2, p_value=0.5, seed=11)
            iter_b1 = self._make_noise_result(n_cells, n_genes, p_value=0.5, seed=12)
            results = iter([iter_a1, iter_a2, iter_b1])

            p1, p2, p3, p4 = self._patch_plot_writers()
            with patch("experiments.recursive_svg.run_permutation_method", side_effect=lambda ds, cfg: next(results)), p1, p2, p3, p4:
                payload, summary_path = run_recursive_svg(dataset, run_config)

            summary_file = Path(tmpdir) / "test_sep_run" / "test_sep_run_celltype_recursive_summary.json"
            self.assertEqual(summary_path, summary_file)
            self.assertTrue(summary_file.exists())
            self.assertIn("combined_isodepth_plot", payload)
            self.assertIn("combined_metric_distribution_plot", payload)
            self.assertIn("svg_count_plot", payload)
            self.assertTrue((summary_file.parent / "test_sep_run_svg_counts_by_gradient.png").exists())
            self.assertFalse(payload["per_type_recursive"]["TypeA"]["skipped"])
            self.assertIn("svg_count_plot", payload["per_type_recursive"]["TypeA"])
            self.assertFalse(payload["per_type_recursive"]["TypeB"]["skipped"])
            self.assertEqual(payload["per_type_recursive"]["TypeB"]["n_gradients_found"], 0)

    def test_same_svg_can_be_reported_in_multiple_cell_types(self) -> None:
        n_cells, n_genes = 40, 6
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._make_run_config_separate(tmpdir)
            S_a, A_a = self._make_type_data(n_cells, n_genes, n_sig=2, seed=0)
            S_b, A_b = self._make_type_data(n_cells, n_genes, n_sig=2, seed=5)
            dataset = self._make_dataset({"TypeA": (S_a, A_a), "TypeB": (S_b, A_b)})

            iter_a1 = self._make_result(A_a, p_value=0.01, seed=10)
            iter_a2 = self._make_noise_result(n_cells, n_genes - 2, p_value=0.5, seed=11)
            iter_b1 = self._make_result(A_b, p_value=0.01, seed=12)
            iter_b2 = self._make_noise_result(n_cells, n_genes - 2, p_value=0.5, seed=13)
            results = iter([iter_a1, iter_a2, iter_b1, iter_b2])

            p1, p2, p3, p4 = self._patch_plot_writers()
            with patch("experiments.recursive_svg.run_permutation_method", side_effect=lambda ds, cfg: next(results)), p1, p2, p3, p4:
                payload, _ = run_recursive_svg(dataset, run_config)

            combined_csv = Path(payload["combined_sig_genes_csv"])
            with open(combined_csv, newline="") as fh:
                rows = list(csv.DictReader(fh))

            gene0_rows = [row for row in rows if row["gene"] == "gene_0"]
            self.assertEqual(
                {row["cell_type"] for row in gene0_rows},
                {"TypeA", "TypeB"},
            )
            self.assertEqual(len(gene0_rows), 2)


if __name__ == "__main__":
    unittest.main()
