import json
from pathlib import Path

import numpy as np

from data.schemas import (
    DataConfig,
    DatasetBundle,
    OutputConfig,
    RunConfig,
    TestConfig,
    TestResult,
    run_config_from_mapping,
)
from experiments.studies.kernel_blurred_sweep.sweep import (
    DatasetKey,
    build_data_config,
    load_kernel_blurred_sweep_spec,
    run_kernel_blurred_sweep,
)
from methods.visium_pseudobulk import build_visium_pseudobulk_dataset

REPO = Path(__file__).resolve().parents[1]
VISIUM_STUDY_SPEC = REPO / "configs/experiments/kernel_blurred_visium_pseudobulk_sweep.json"
PSEUDOBULK_STUDY_SPEC = REPO / "configs/experiments/kernel_blurred_pseudobulk_sweep.json"


def _square_grid_dataset(n_side: int = 20, spacing: float = 5.0) -> DatasetBundle:
    """Dense square lattice of unit-count cells, big enough for several 100µm spots."""
    xs, ys = np.meshgrid(
        np.arange(n_side, dtype=np.float64) * spacing,
        np.arange(n_side, dtype=np.float64) * spacing,
        indexing="ij",
    )
    S = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    A = np.ones((S.shape[0], 2), dtype=np.float32)
    return DatasetBundle(
        S=S,
        A=A,
        meta={
            "coordinate_um_per_unit": 1.0,
            "visium_pseudobulk_preprocessing": {
                "normalize_total": False,
                "log1p": False,
                "standardize_expression": False,
            },
        },
    ).validate()


def test_visium_pseudobulk_drops_gap_cells_and_sums_counts():
    dataset = _square_grid_dataset(n_side=60, spacing=5.0)
    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    binned, artifacts = build_visium_pseudobulk_dataset(dataset, config)
    summary = artifacts["visium_pseudobulk_summary"]

    # Spots don't cover the full tissue (diameter 55 < distance 100), so some cells
    # must be dropped in the gaps, and the retained cell count must be < original.
    assert summary["n_cells_dropped_in_gaps"] > 0
    assert summary["original_n_cells"] == dataset.n_cells
    assert int(artifacts["visium_pseudobulk_cell_counts"].sum()) == (
        summary["original_n_cells"] - summary["n_cells_dropped_in_gaps"]
    )
    # Every retained pseudospot must actually have >=1 cell.
    assert np.all(artifacts["visium_pseudobulk_cell_counts"] >= 1)
    # Counts (2 genes, value 1 per cell) should equal 2x cell count per spot.
    assert np.allclose(binned.A[:, 0], artifacts["visium_pseudobulk_cell_counts"])
    assert np.allclose(binned.A[:, 1], artifacts["visium_pseudobulk_cell_counts"])
    assert binned.n_cells == summary["n_pseudospots"]


def test_visium_pseudobulk_discards_spots_whose_circle_exits_tissue_mask():
    dataset = _square_grid_dataset(n_side=60, spacing=5.0)
    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    # Dataset has no "shape" in meta (not marked synthetic), so this falls back to the
    # density-based tissue mask: spots near the point-cloud boundary should still be
    # discarded relative to the raw candidate-lattice count.
    binned, artifacts = build_visium_pseudobulk_dataset(dataset, config)
    summary = artifacts["visium_pseudobulk_summary"]
    assert summary["n_spots_in_tissue"] < summary["n_candidate_spots"]
    assert summary["n_pseudospots"] <= summary["n_spots_in_tissue"]


def test_visium_pseudobulk_uses_synthetic_shape_mask_for_edge_effects():
    dataset = _square_grid_dataset(n_side=20, spacing=5.0)
    dataset.meta["source"] = "synthetic"
    dataset.meta["shape"] = "square"
    # Tissue extent is [0, 95] x [0, 95] in native units after this z-score-free setup;
    # _point_in_spatial_shape for "square" is defined on [0, 1]^2, so mark coordinates as
    # already "native" (no z-score) -- membership will therefore be false almost everywhere
    # except near the origin, exercising the synthetic-shape branch without asserting exact
    # counts (covered by the unit-square shape tests in test_synthetic.py).
    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    try:
        build_visium_pseudobulk_dataset(dataset, config)
    except ValueError as exc:
        assert "no spot whose full circular footprint" in str(exc)


def test_visium_pseudobulk_keeps_cell_types_as_separate_pseudospots():
    dataset = _square_grid_dataset(n_side=60, spacing=5.0)
    n_cells = dataset.n_cells
    dataset.meta["cell_type_mode"] = "together"
    dataset.meta["cell_type_labels"] = np.array(
        [i % 2 for i in range(n_cells)], dtype=np.int64
    )
    dataset.meta["cell_type_names"] = ["a", "b"]
    dataset.meta["n_cell_types"] = 2
    dataset.A = np.column_stack(
        [
            np.where(dataset.meta["cell_type_labels"] == 0, 1.0, 0.0),
            np.where(dataset.meta["cell_type_labels"] == 1, 1.0, 0.0),
        ]
    ).astype(np.float32)

    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()

    binned, artifacts = build_visium_pseudobulk_dataset(dataset, config)
    # Both cell types should be represented among the resulting pseudospots.
    assert set(np.unique(binned.meta["cell_type_labels"]).tolist()) <= {0, 1}
    assert binned.n_cells == artifacts["visium_pseudobulk_summary"]["n_pseudospots"]


def test_visium_pseudobulk_requires_diameter_less_than_distance():
    try:
        TestConfig(
            method="visium_pseudobulk",
            n_perms=1,
            visium_spot_distance_um=50.0,
            visium_spot_diameter_um=55.0,
            coordinate_um_per_unit=1.0,
        ).validate()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_visium_pseudobulk_overlay_writes_png():
    import tempfile
    from pathlib import Path

    from analysis.plots import save_visium_pseudobulk_overlay

    dataset = _square_grid_dataset(n_side=60, spacing=5.0)
    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()
    _, artifacts = build_visium_pseudobulk_dataset(dataset, config)
    out = Path(tempfile.mkdtemp()) / "overlay.png"
    path = save_visium_pseudobulk_overlay(
        artifacts["visium_overlay_S_um"],
        artifacts["visium_overlay_assigned"],
        artifacts["visium_overlay_centers_in_tissue_um"],
        artifacts["visium_overlay_occupied_ids"],
        hex_radius_um=artifacts["visium_overlay_hex_radius_um"],
        spot_radius_um=artifacts["visium_overlay_spot_radius_um"],
        out_path=out,
        run_name="overlay_test",
        spot_distance_um=100.0,
        spot_diameter_um=55.0,
    )
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_visium_pseudobulk_config_round_trips():
    run_config = run_config_from_mapping(
        {
            "data": {
                "source": "synthetic",
            },
            "test": {
                "method": "visium_pseudobulk",
                "visium_spot_distance_um": 100.0,
                "visium_spot_diameter_um": 55.0,
                "coordinate_um_per_unit": 960.0,
            },
        }
    )

    assert run_config.test.method == "visium_pseudobulk"
    assert run_config.test.visium_spot_distance_um == 100.0
    assert run_config.test.visium_spot_diameter_um == 55.0


def test_visium_pseudobulk_study_uses_kernel_blurred_sweep():
    spec = load_kernel_blurred_sweep_spec(VISIUM_STUDY_SPEC)
    assert spec.bandwidths_um == [15.0, 30.0, 55.0, 100.0]
    assert spec.experiment_name == "kernel_blurred_visium_pseudobulk_sweep"
    base = json.loads(spec.base_config.read_text(encoding="utf-8"))
    assert base["data"]["kernel"]["type"] == "smooth"
    assert float(base["data"]["delta"]) == 0.0
    cfg = build_data_config(
        base, key=DatasetKey(bandwidth_um=30.0, data_seed=0)
    )
    assert cfg.kernel.type == "smooth"
    assert float(cfg.kernel.distance) == 30.0
    assert float(cfg.delta) == 0.0
    payload = run_kernel_blurred_sweep(spec, dry_run=True)
    assert payload["planned_run_count"] == 40
    assert payload["planned_datasets_preview"][0].startswith("bw15_seed")


def test_kernel_blurred_sweep_rejects_nonsmooth_kernel():
    spec = load_kernel_blurred_sweep_spec(VISIUM_STUDY_SPEC)
    spec.base_config = REPO / "configs/synthetic/kernel_noise_binning_square_uniform.json"
    try:
        run_kernel_blurred_sweep(spec, dry_run=True)
        raised = False
    except ValueError as exc:
        raised = True
        assert "smooth" in str(exc)
    assert raised


def test_kernel_blurred_binning_sweep_matches_visium_pitch():
    spec = load_kernel_blurred_sweep_spec(PSEUDOBULK_STUDY_SPEC)
    visium = load_kernel_blurred_sweep_spec(VISIUM_STUDY_SPEC)
    assert spec.bandwidths_um == visium.bandwidths_um
    assert spec.seeds == visium.seeds
    base = json.loads(spec.base_config.read_text(encoding="utf-8"))
    visium_base = json.loads(visium.base_config.read_text(encoding="utf-8"))
    assert base["test"]["method"] == "binning"
    assert float(base["test"]["bin_spot_distance_um"]) == float(
        visium_base["test"]["visium_spot_distance_um"]
    )
    assert base["test"]["bin_shape"] == "hexagon"
    assert base["data"]["kernel"]["type"] == "smooth"
    payload = run_kernel_blurred_sweep(spec, dry_run=True)
    assert payload["test_method"] == "binning"
    assert payload["planned_run_count"] == 40
    assert payload["planned_runs_preview"][0]["run_name"].endswith("_binning")


def test_kernel_blurred_sweep_arms_share_data_generation_recipe():
    visium = load_kernel_blurred_sweep_spec(VISIUM_STUDY_SPEC)
    binning = load_kernel_blurred_sweep_spec(PSEUDOBULK_STUDY_SPEC)
    visium_data = json.loads(visium.base_config.read_text(encoding="utf-8"))["data"]
    binning_data = json.loads(binning.base_config.read_text(encoding="utf-8"))["data"]
    assert visium_data == binning_data


def test_visium_standardized_outputs_save_original_dataset_plot():
    import tempfile

    from experiments.configuration import save_standardized_outputs

    dataset = _square_grid_dataset(n_side=60, spacing=5.0)
    dataset.meta["kernel"] = {"type": "smooth", "distance": 15.0}
    dataset.meta["smooth_bandwidth_um"] = 15.0
    dataset.meta["kernel_noise_sample"] = np.linspace(-1.0, 1.0, dataset.n_cells, dtype=np.float32)
    config = TestConfig(
        method="visium_pseudobulk",
        n_perms=1,
        visium_spot_distance_um=100.0,
        visium_spot_diameter_um=55.0,
        coordinate_um_per_unit=1.0,
        block_jitter=False,
    ).validate()
    binned, artifacts = build_visium_pseudobulk_dataset(dataset, config)
    result = TestResult(
        method_name="visium_pseudobulk",
        metric="mse",
        p_value=0.5,
        stat_true=0.1,
        stat_perm=np.asarray([0.2], dtype=np.float64),
        runtime_sec=0.01,
        n_cells=binned.n_cells,
        n_genes=binned.n_genes,
        config={},
        artifacts={
            **artifacts,
            "lowest_S": np.asarray(binned.S, dtype=np.float32),
            "highest_S": np.asarray(binned.S, dtype=np.float32),
            "lowest_stat": 0.2,
            "highest_stat": 0.3,
        },
    ).validate()

    with tempfile.TemporaryDirectory() as tmpdir:
        run_config = RunConfig(
            data=DataConfig(source="synthetic", mode="noise"),
            test=config,
            output=OutputConfig(out_dir=tmpdir, run_name="visium_original_plot"),
        ).validate()
        payload, result_path = save_standardized_outputs(dataset, result, run_config)
        original_path = result_path.parent / "visium_original_plot_original_dataset.png"
        assert original_path.exists()
        assert original_path.stat().st_size > 0
        assert payload["artifacts"]["original_dataset_plot"] == str(original_path)
        assert payload["artifacts"]["visium_pseudobulk_summary"]["n_pseudospots"] == binned.n_cells
