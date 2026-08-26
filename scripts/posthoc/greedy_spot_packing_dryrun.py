"""Standalone dry run: greedy maximal circle packing for Visium-style pseudobulking.

Not wired into the pipeline. Keeps the exact same capture mechanism as
`methods.visium_pseudobulk` (fixed-diameter circle, cells outside every circle are
dropped) but replaces the *placement* of spot centers: instead of one rigid global
hex lattice (only 2 degrees of freedom -- an x/y offset -- so entire rows/columns of
spots can miss nearby cell clusters), candidate centers are the cells themselves.
Candidates are ranked by how many cells they'd capture, then greedily accepted
(highest-scoring first) as long as they are >= `spot_distance_um` from every
already-accepted center -- i.e. non-max suppression / greedy maximum weighted
independent set on the disk-conflict graph. Every accepted spot is occupied by
construction (it's centered on a real cell), and neighboring circles can never
overlap because centers are always >= spot_distance_um apart while circles have
diameter < spot_distance_um.

Usage::

    python scripts/posthoc/greedy_spot_packing_dryrun.py \
        --result results/jfan_merfish/260825_jfan_merfish_visium_pseudobulk/260825_jfan_merfish_visium_pseudobulk_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data import load_dataset, raw_coordinates_from_standardized  # noqa: E402
from data.schemas import DataConfig  # noqa: E402


def _density_tissue_membership_fn(S_um: np.ndarray, radius_um: float):
    tree = cKDTree(S_um)

    def _membership(xy_um: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(np.asarray(xy_um, dtype=np.float64), k=1)
        return np.asarray(dist) <= float(radius_um)

    return _membership


def _circle_fully_in_tissue(
    center: np.ndarray, radius: float, membership_fn, n_samples: int = 16,
) -> bool:
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    pts = np.vstack([center[None, :], center[None, :] + offsets])
    return bool(np.all(membership_fn(pts)))


def greedy_pack_spots(
    S_um: np.ndarray,
    *,
    spot_radius_um: float,
    spot_distance_um: float,
    membership_fn,
) -> np.ndarray:
    """Greedy NMS-style maximal packing. Returns indices into `S_um` used as centers."""
    n = S_um.shape[0]
    tree = cKDTree(S_um)
    # Score each cell by local capture density: how many cells would land in its
    # circle if it were chosen as a spot center.
    scores = tree.query_ball_point(S_um, r=spot_radius_um, return_length=True)
    order = np.argsort(-scores, kind="stable")

    active = np.ones(n, dtype=bool)
    chosen: list[int] = []
    for idx in order:
        if not active[idx]:
            continue
        center = S_um[idx]
        if not _circle_fully_in_tissue(center, spot_radius_um, membership_fn):
            active[idx] = False
            continue
        chosen.append(int(idx))
        nearby = tree.query_ball_point(center, r=spot_distance_um)
        active[nearby] = False
    return np.asarray(chosen, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--spot-distance-um", type=float, default=100.0)
    parser.add_argument("--spot-diameter-um", type=float, default=55.0)
    parser.add_argument("--out", default=None)
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

    spot_radius_um = float(args.spot_diameter_um) / 2.0
    spot_distance_um = float(args.spot_distance_um)

    tree = cKDTree(S_um)
    nn_dist, _ = tree.query(S_um, k=2)
    median_nn = float(np.median(nn_dist[:, 1]))
    tissue_radius_um = 1.5 * median_nn
    membership_fn = _density_tissue_membership_fn(S_um, tissue_radius_um)

    print(f"n_cells={n_cells}  spot_distance_um={spot_distance_um}  spot_diameter_um={args.spot_diameter_um}  "
          f"tissue_mask_radius_um={tissue_radius_um:.1f}")

    chosen_idx = greedy_pack_spots(
        S_um, spot_radius_um=spot_radius_um, spot_distance_um=spot_distance_um, membership_fn=membership_fn,
    )
    centers = S_um[chosen_idx]
    n_spots = centers.shape[0]

    spot_tree = cKDTree(centers)
    dist, nearest = spot_tree.query(S_um, k=1)
    assigned = dist <= spot_radius_um
    counts = np.bincount(nearest[assigned], minlength=n_spots)
    n_occupied = int((counts > 0).sum())
    n_dropped = int((~assigned).sum())

    # Sanity: min center-to-center distance among chosen spots (must be >= spot_distance_um).
    if n_spots > 1:
        ctree = cKDTree(centers)
        d2, _ = ctree.query(centers, k=2)
        min_center_dist = float(d2[:, 1].min())
    else:
        min_center_dist = float("inf")

    print(f"\ngreedy packing result:")
    print(f"  spots placed: {n_spots}  (all occupied by construction; observed occupied={n_occupied})")
    print(f"  cells captured: {int(assigned.sum())}  cells dropped: {n_dropped} ({n_dropped/n_cells:.1%})")
    print(f"  cells/spot: min={counts[counts>0].min()} median={np.median(counts[counts>0]):.1f} "
          f"mean={counts[counts>0].mean():.2f} max={counts.max()}")
    print(f"  min observed center-to-center spacing: {min_center_dist:.1f} um (constraint: >= {spot_distance_um})")

    vp_summary = (result.get("artifacts") or {}).get("visium_pseudobulk_summary")
    if vp_summary:
        print("\nvs. actual visium_pseudobulk (rigid hex lattice) run in this result.json:")
        print(f"  occupied spots: {vp_summary['n_pseudospots']}  (vs {n_spots} greedy-packed)")
        print(f"  cells dropped in gaps: {vp_summary['n_cells_dropped_in_gaps']} "
              f"({vp_summary['fraction_cells_dropped_in_gaps']:.1%})  (vs {n_dropped} / {n_dropped/n_cells:.1%} greedy-packed)")

    _plot_overlay(
        S_um, centers, assigned, nearest,
        spot_radius_um=spot_radius_um,
        out_path=Path(args.out) if args.out else result_path.parent / f"{result_path.parent.name}_greedy_packing_dryrun.png",
    )


def _plot_overlay(
    S_um: np.ndarray,
    centers: np.ndarray,
    assigned: np.ndarray,
    nearest: np.ndarray,
    *,
    spot_radius_um: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    counts = np.bincount(nearest[assigned], minlength=centers.shape[0])
    occupied_mask = counts > 0

    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.scatter(
        S_um[~assigned, 0], S_um[~assigned, 1], c="#c8c8c8", s=6, linewidths=0, alpha=0.5,
        zorder=1, label="cells in gaps (dropped)",
    )
    ax.scatter(
        S_um[assigned, 0], S_um[assigned, 1], c="#4c78a8", s=8, linewidths=0, alpha=0.85,
        zorder=2, label="cells in spots",
    )
    for c in centers[occupied_mask]:
        ax.add_patch(Circle((c[0], c[1]), spot_radius_um, fill=False, edgecolor="#c0392b", linewidth=0.8, zorder=3))
    ax.scatter(centers[occupied_mask, 0], centers[occupied_mask, 1], c="black", s=8, marker="+", linewidths=0.8, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x (\u00b5m)")
    ax.set_ylabel("y (\u00b5m)")
    n_spots = int(occupied_mask.sum())
    n_dropped = int((~assigned).sum())
    ax.set_title(
        f"Greedy max-packing pseudobulk (dry run) \u2014 {n_spots} spots, all occupied\n"
        f"cells captured={int(assigned.sum())}  cells dropped={n_dropped} ({n_dropped/S_um.shape[0]:.1%})\n"
        f"cells/spot median={np.median(counts[occupied_mask]):.1f} max={counts.max()}"
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4c78a8", markersize=6, label="cells in spots"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c8c8c8", markersize=6, label="cells in gaps (dropped)"),
        Line2D([0], [0], color="#c0392b", lw=1.2, label="capture circle"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
