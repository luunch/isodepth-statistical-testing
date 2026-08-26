"""Standalone dry run: adaptive Voronoi tessellation for Visium-style pseudobulking.

Not wired into the pipeline. Prototypes the idea discussed for the
`260825_jfan_merfish_visium_pseudobulk` run: instead of dropping every cell that
falls outside a fixed 55 um capture circle, grow each bin (via nearest-neighbor
accretion, cf. Cappellari & Copin 2003's Voronoi-binning bin-accretion phase) until
it covers roughly a target physical footprint area *and* has a minimum cell count,
then regularize with Lloyd/CVT iterations so every bin is a proper Voronoi cell
around its own centroid. Every cell always ends up in some bin -- nothing is
dropped in gaps, and bin size adapts to local density.

Usage::

    python scripts/posthoc/adaptive_voronoi_dryrun.py \
        --result results/jfan_merfish/260825_jfan_merfish_visium_pseudobulk/260825_jfan_merfish_visium_pseudobulk_result.json \
        --target-spacing-um 100 --min-cells 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Voronoi, cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data import load_dataset, raw_coordinates_from_standardized  # noqa: E402
from data.schemas import DataConfig  # noqa: E402


def _bin_area_um2(points: np.ndarray) -> float:
    """Convex-hull area; 0.0 until there are enough points to define one."""
    if points.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)  # `volume` is the 2D area for a 2D hull
    except Exception:
        return 0.0


def bin_accretion(
    S_um: np.ndarray,
    *,
    target_area_um2: float,
    min_cells: int,
    max_cells: int,
) -> np.ndarray:
    """Greedy nearest-neighbor region growing until each bin meets area+count targets.

    Mirrors steps (i)-(v) of Cappellari & Copin (2003) sec. 5.1, but the
    accretion stop rule is (bin_area >= target_area_um2 and n_pts >= min_cells)
    instead of a signal-to-noise threshold.
    """
    n = S_um.shape[0]
    unbinned = np.ones(n, dtype=bool)
    bin_id = np.full(n, -1, dtype=np.int64)
    tree_all = cKDTree(S_um)

    current_bin = 0
    while np.any(unbinned):
        remaining_idx = np.flatnonzero(unbinned)
        # Seed each new bin from the point closest to the centroid of what's left,
        # so bins are grown from the "middle" of remaining unclaimed territory.
        remaining_centroid = S_um[remaining_idx].mean(axis=0)
        seed_local = np.argmin(np.linalg.norm(S_um[remaining_idx] - remaining_centroid, axis=1))
        seed = int(remaining_idx[seed_local])

        members = [seed]
        unbinned[seed] = False
        member_tree = None
        while True:
            pts = S_um[members]
            area = _bin_area_um2(pts)
            reached_target = area >= target_area_um2 and len(members) >= min_cells
            if reached_target or len(members) >= max_cells:
                break
            remaining_idx = np.flatnonzero(unbinned)
            if remaining_idx.size == 0:
                break
            centroid = pts.mean(axis=0)
            # Nearest still-unbinned cell to the bin's running centroid.
            d = np.linalg.norm(S_um[remaining_idx] - centroid, axis=1)
            nxt = int(remaining_idx[np.argmin(d)])
            members.append(nxt)
            unbinned[nxt] = False

        bin_id[members] = current_bin
        current_bin += 1

    assert np.all(bin_id >= 0)
    return bin_id


def lloyd_relax(
    S_um: np.ndarray,
    bin_id: np.ndarray,
    *,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centroidal Voronoi Tessellation refinement: reassign to nearest centroid, repeat."""
    unique_ids = np.unique(bin_id)
    centers = np.array([S_um[bin_id == b].mean(axis=0) for b in unique_ids])
    assignment = bin_id.copy()
    for it in range(n_iter):
        tree = cKDTree(centers)
        _, nearest = tree.query(S_um, k=1)
        if it > 0 and np.array_equal(nearest, assignment):
            break
        # Drop any generator that lost all its points (rare, but keeps bins valid).
        occupied = np.unique(nearest)
        if occupied.size < centers.shape[0]:
            centers = centers[occupied]
            tree = cKDTree(centers)
            _, nearest = tree.query(S_um, k=1)
        assignment = nearest
        new_centers = np.array([S_um[assignment == i].mean(axis=0) for i in range(centers.shape[0])])
        centers = new_centers
    return assignment, centers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to a *_result.json from a visium_pseudobulk run")
    parser.add_argument("--target-spacing-um", type=float, default=100.0,
                         help="Reference footprint = area of a hex tile at this spacing (default matches the visium_pseudobulk hex pitch)")
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--max-cells", type=int, default=40, help="Safety cap so a single bin can't swallow a whole sparse region")
    parser.add_argument("--cvt-iters", type=int, default=15)
    parser.add_argument("--out", default=None, help="Output PNG path (default: alongside the result.json)")
    args = parser.parse_args()

    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    data_cfg_raw = dict(result["config"]["data"])
    cfg = DataConfig(
        source=data_cfg_raw["source"],
        h5ad=data_cfg_raw.get("h5ad"),
        spatial_key=data_cfg_raw.get("spatial_key", "spatial"),
        min_cells_per_gene=int(data_cfg_raw.get("min_cells_per_gene", 0)),
        standardize_expression=False,
        standardize_coordinates=True,
        normalize_total=False,
        log1p=False,
        use_raw=bool(data_cfg_raw.get("use_raw", False)),
        seed=int(data_cfg_raw.get("seed", 0)),
    ).validate()
    dataset = load_dataset(cfg)
    S_um = np.asarray(raw_coordinates_from_standardized(dataset.S, dataset.meta), dtype=np.float64)
    n_cells = S_um.shape[0]

    hex_radius_um = float(args.target_spacing_um) / np.sqrt(3.0)
    target_area_um2 = (3.0 * np.sqrt(3.0) / 2.0) * hex_radius_um**2  # regular hexagon area

    print(f"n_cells={n_cells}  target_spacing_um={args.target_spacing_um}  "
          f"target_area_um2={target_area_um2:.1f}  min_cells={args.min_cells}  max_cells={args.max_cells}")

    accreted = bin_accretion(
        S_um, target_area_um2=target_area_um2, min_cells=args.min_cells, max_cells=args.max_cells,
    )
    n_accretion_bins = int(np.unique(accreted).size)
    print(f"after accretion: {n_accretion_bins} bins, all {n_cells} cells assigned (0 dropped)")

    final_assignment, centers = lloyd_relax(S_um, accreted, n_iter=args.cvt_iters)
    n_final_bins = int(centers.shape[0])
    counts = np.bincount(final_assignment, minlength=n_final_bins)
    print(f"after CVT relaxation: {n_final_bins} bins")
    print(f"cells/bin: min={counts.min()} median={np.median(counts):.1f} "
          f"mean={counts.mean():.2f} max={counts.max()}")
    print(f"cells dropped: 0 / {n_cells} (by construction)")

    # Compare against the run's actual visium_pseudobulk summary, if present.
    vp_summary = (result.get("artifacts") or {}).get("visium_pseudobulk_summary")
    if vp_summary:
        print("\nvs. actual visium_pseudobulk run in this result.json:")
        print(f"  occupied spots: {vp_summary['n_pseudospots']}  "
              f"(vs {n_final_bins} adaptive bins)")
        print(f"  cells dropped in gaps: {vp_summary['n_cells_dropped_in_gaps']} "
              f"({vp_summary['fraction_cells_dropped_in_gaps']:.1%})  (vs 0 adaptive)")

    _plot_overlay(
        S_um, final_assignment, centers,
        out_path=Path(args.out) if args.out else result_path.parent / f"{result_path.parent.name}_adaptive_voronoi_dryrun.png",
        target_spacing_um=args.target_spacing_um,
        min_cells=args.min_cells,
    )


def _plot_overlay(
    S_um: np.ndarray,
    assignment: np.ndarray,
    centers: np.ndarray,
    *,
    out_path: Path,
    target_spacing_um: float,
    min_cells: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n_bins = centers.shape[0]
    counts = np.bincount(assignment, minlength=n_bins)

    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    cmap = plt.get_cmap("tab20")
    ax.scatter(
        S_um[:, 0], S_um[:, 1],
        c=[cmap(int(b) % 20) for b in assignment],
        s=10, linewidths=0.2, edgecolors="white", zorder=2,
    )
    ax.scatter(centers[:, 0], centers[:, 1], c="black", s=14, marker="+", linewidths=1.0, zorder=3, label="bin centroid")

    # Voronoi diagram of the final generators, clipped to the plotted region, for
    # a visual proxy of the adaptive bin boundaries (finite-region segments only).
    if n_bins >= 4:
        vor = Voronoi(centers)
        segments = []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in simplex:
                continue
            segments.append(vor.vertices[simplex])
        if segments:
            ax.add_collection(LineCollection(segments, colors="#333333", linewidths=0.6, alpha=0.8, zorder=1))

    ax.set_aspect("equal")
    ax.set_xlabel("x (\u00b5m)")
    ax.set_ylabel("y (\u00b5m)")
    pad = target_spacing_um
    ax.set_xlim(S_um[:, 0].min() - pad, S_um[:, 0].max() + pad)
    ax.set_ylim(S_um[:, 1].min() - pad, S_um[:, 1].max() + pad)
    ax.set_title(
        f"Adaptive Voronoi pseudobulk (dry run) \u2014 {n_bins} bins, 0 cells dropped\n"
        f"cells/bin min={counts.min()} median={np.median(counts):.1f} mean={counts.mean():.2f} max={counts.max()}\n"
        f"target footprint = hex area @ {target_spacing_um:g} \u00b5m spacing, min_cells={min_cells}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
