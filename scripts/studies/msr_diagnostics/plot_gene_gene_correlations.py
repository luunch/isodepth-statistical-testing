"""Plot gene-by-gene Pearson correlation heatmaps for all kernel-noise experiment sweeps.

For each experiment that has a datasets/ folder with .npz files, reads the
manifest to discover sweep parameters, computes 20×20 gene–gene Pearson
correlation matrices, and writes three figures per experiment into the
experiment's analysis/ directory:

  kernel_noise_gene_correlation_summary.png  — mean corr per condition
  kernel_noise_gene_correlation_all_seeds.png — all individual heatmaps
  kernel_noise_gene_correlation_std.png       — std dev across seeds

Usage:
    conda run -n isodepth_env python scripts/plot_gene_gene_correlations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Experiments to process (those that have a datasets/ directory with .npz files)
# ---------------------------------------------------------------------------
EXPERIMENTS_ROOT = Path(
    "/home/ajain71/scratchuchitra1/users/ajain71/isodepth-statistical-testing"
    "/results/experiments"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def delta_to_str(delta: float) -> str:
    """Convert float delta to filename fragment, e.g. 0.05 → '0p05'."""
    s = f"{delta:g}"          # '0.05', '0.1', '0.5', '1', '0.01'
    s = s.replace(".", "p")   # '0p05', '0p1', '0p5', '1', '0p01'
    return s


def dist_to_str(d: float) -> str:
    """Convert kernel distance float to filename prefix, e.g. 15.0 → 'd15'."""
    return f"d{int(d)}"


def load_corr(datasets_dir: Path, kernel_dist: float, delta: float, seed: int) -> np.ndarray:
    """Return (n_genes, n_genes) Pearson correlation matrix for one dataset."""
    key = f"{dist_to_str(kernel_dist)}_delta{delta_to_str(delta)}_seed{seed}"
    path = datasets_dir / f"{key}.npz"
    data = np.load(path)
    A = data["A"].astype(np.float64)   # (n_cells, n_genes)
    return np.corrcoef(A.T)            # (n_genes, n_genes)


def gene_order_from_grand_mean(mean_corrs: list[np.ndarray]) -> list[int]:
    """Hierarchical clustering order from the grand mean of a list of corr matrices."""
    grand_mean = np.mean(np.stack(mean_corrs), axis=0)
    dist_mat = 1.0 - np.abs(grand_mean)
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = (dist_mat + dist_mat.T) / 2.0          # enforce exact symmetry
    condensed = squareform(np.clip(dist_mat, 0.0, None))
    Z = linkage(condensed, method="average")
    return dendrogram(Z, no_plot=True)["leaves"]


def symmetric_vmax(mean_corrs_flat: np.ndarray, percentile: float = 99) -> float:
    """Colour scale limit from the off-diagonal absolute values."""
    abs_max = np.nanpercentile(np.abs(mean_corrs_flat), percentile)
    return max(float(abs_max) * 1.2, 0.05)


def draw_heatmap(ax, mat: np.ndarray, ordered_labels: list[str],
                 vmin: float, vmax: float, *, show_labels: bool = True,
                 cmap: str = "RdBu_r") -> plt.cm.ScalarMappable:
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    n = mat.shape[0]
    if show_labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(ordered_labels, fontsize=5, rotation=90)
        ax.set_yticklabels(ordered_labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return im


# ---------------------------------------------------------------------------
# Process each experiment
# ---------------------------------------------------------------------------

def process_experiment(exp_dir: Path) -> None:
    manifest_path = exp_dir / "manifest.json"
    datasets_dir  = exp_dir / "datasets"
    analysis_dir  = exp_dir / "analysis"

    if not manifest_path.exists() or not datasets_dir.exists():
        return
    if not list(datasets_dir.glob("*.npz")):
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    exp_name = exp_dir.name
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")

    deltas    = [float(d) for d in manifest.get("deltas", [])]
    seeds     = [int(s)   for s in manifest.get("seeds",  [])]
    kdists    = [float(d) for d in manifest.get("kernel_distances_um", [15.0])]

    if not deltas or not seeds or not kdists:
        print("  Skipping: missing deltas/seeds/kernel_distances_um in manifest.")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load correlations — shape: corrs[kdist][delta][seed_idx]
    # ------------------------------------------------------------------
    print(f"  kernel_distances={kdists}  deltas={deltas}  seeds={seeds}")
    corrs: dict[float, dict[float, list[np.ndarray]]] = {}
    for kd in kdists:
        corrs[kd] = {}
        for delta in deltas:
            seed_corrs = []
            for seed in seeds:
                try:
                    seed_corrs.append(load_corr(datasets_dir, kd, delta, seed))
                except FileNotFoundError:
                    pass  # some sweeps may skip certain combos
            corrs[kd][delta] = seed_corrs
            print(f"    kd={kd}, delta={delta}: {len(seed_corrs)} seeds loaded")

    # Mean / std per (kd, delta)
    mean_corrs: dict[float, dict[float, np.ndarray]] = {}
    std_corrs:  dict[float, dict[float, np.ndarray]] = {}
    for kd in kdists:
        mean_corrs[kd] = {}
        std_corrs[kd]  = {}
        for delta in deltas:
            stack = np.stack(corrs[kd][delta]) if corrs[kd][delta] else None
            if stack is not None:
                mean_corrs[kd][delta] = stack.mean(axis=0)
                std_corrs[kd][delta]  = stack.std(axis=0)

    # ------------------------------------------------------------------
    # Clustering order from grand mean across all conditions
    # ------------------------------------------------------------------
    all_means = [mean_corrs[kd][d] for kd in kdists for d in deltas
                 if d in mean_corrs.get(kd, {})]
    gene_order = gene_order_from_grand_mean(all_means)
    n_genes = all_means[0].shape[0]
    gene_labels   = [f"G{i}" for i in range(n_genes)]
    ordered_labels = [gene_labels[i] for i in gene_order]

    def reorder(mat: np.ndarray) -> np.ndarray:
        return mat[np.ix_(gene_order, gene_order)]

    # ------------------------------------------------------------------
    # Colour scale
    # ------------------------------------------------------------------
    off_diag = []
    for kd in kdists:
        for delta in deltas:
            if delta not in mean_corrs.get(kd, {}):
                continue
            m = mean_corrs[kd][delta].copy()
            np.fill_diagonal(m, np.nan)
            off_diag.append(m.flatten())
    vmax = symmetric_vmax(np.concatenate(off_diag))
    vmin = -vmax
    print(f"  Colour scale: [{vmin:.3f}, {vmax:.3f}]")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def delta_label(d: float) -> str:
        return rf"$\delta={d:g}$"

    def kdist_label(kd: float) -> str:
        return rf"$d={int(kd)}\,\mu$m"

    # ==================================================================
    # Figure 1: Summary — mean correlation, rows=kdists, cols=deltas
    # ==================================================================
    n_rows, n_cols = len(kdists), len(deltas)
    fig_w = max(4.5 * n_cols, 6)
    fig_h = max(4.0 * n_rows, 4)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    for ri, kd in enumerate(kdists):
        for ci, delta in enumerate(deltas):
            ax = axes[ri][ci]
            if delta not in mean_corrs.get(kd, {}):
                ax.axis("off")
                continue
            mat = reorder(mean_corrs[kd][delta])
            im = draw_heatmap(ax, mat, ordered_labels, vmin, vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Pearson r", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            title_parts = []
            if n_cols > 1:
                title_parts.append(delta_label(delta))
            if n_rows > 1:
                title_parts.append(kdist_label(kd))
            if not title_parts:
                title_parts.append(delta_label(delta))
            ax.set_title(",  ".join(title_parts), fontsize=9, pad=6)

    fig.suptitle(
        f"Gene–gene Pearson correlations — {exp_name}\n"
        f"(mean over {len(seeds)} seeds, {n_genes} genes)",
        fontsize=10, y=1.02,
    )
    out = analysis_dir / "kernel_noise_gene_correlation_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

    # ==================================================================
    # Figure 2: All individual seed heatmaps
    # ==================================================================
    conditions = [(kd, d) for kd in kdists for d in deltas
                  if d in mean_corrs.get(kd, {})]
    n_cond = len(conditions)
    max_seeds = max(len(corrs[kd][d]) for kd, d in conditions)

    fig2, axes2 = plt.subplots(
        n_cond, max_seeds,
        figsize=(max(2.0 * max_seeds, 8), max(2.0 * n_cond, 4)),
        squeeze=False,
        gridspec_kw={"hspace": 0.3, "wspace": 0.1},
    )

    for ri, (kd, delta) in enumerate(conditions):
        seed_list = corrs[kd][delta]
        for ci in range(max_seeds):
            ax = axes2[ri][ci]
            if ci < len(seed_list):
                mat = reorder(seed_list[ci])
                draw_heatmap(ax, mat, ordered_labels, vmin, vmax, show_labels=False)
                if ri == 0:
                    ax.set_title(f"seed {seeds[ci]}", fontsize=7, pad=2)
            else:
                ax.axis("off")
        # Row label
        row_parts = []
        if n_rows > 1:
            row_parts.append(kdist_label(kd))
        row_parts.append(delta_label(delta))
        axes2[ri][0].set_ylabel(",  ".join(row_parts), fontsize=8, labelpad=4)

    # Shared colorbar
    cbar_ax2 = fig2.add_axes([0.92, 0.12, 0.013, 0.76])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb2 = fig2.colorbar(sm, cax=cbar_ax2)
    cb2.set_label("Pearson r", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig2.suptitle(
        f"Gene–gene correlations — all seeds — {exp_name}",
        fontsize=10, y=0.99,
    )
    out2 = analysis_dir / "kernel_noise_gene_correlation_all_seeds.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2.name}")

    # ==================================================================
    # Figure 3: Std dev across seeds
    # ==================================================================
    std_vals = np.concatenate([
        std_corrs[kd][d].flatten()
        for kd in kdists for d in deltas
        if d in std_corrs.get(kd, {})
    ])
    std_vmax = max(float(np.percentile(std_vals, 99)), 0.02)

    fig3, axes3 = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )

    for ri, kd in enumerate(kdists):
        for ci, delta in enumerate(deltas):
            ax = axes3[ri][ci]
            if delta not in std_corrs.get(kd, {}):
                ax.axis("off")
                continue
            mat = reorder(std_corrs[kd][delta])
            im = draw_heatmap(ax, mat, ordered_labels, 0, std_vmax,
                              cmap="Oranges")
            cbar = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("std(r)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            title_parts = []
            if n_cols > 1:
                title_parts.append(delta_label(delta))
            if n_rows > 1:
                title_parts.append(kdist_label(kd))
            if not title_parts:
                title_parts.append(delta_label(delta))
            ax.set_title(",  ".join(title_parts), fontsize=9, pad=6)

    fig3.suptitle(
        f"Gene–gene correlation std across seeds — {exp_name}",
        fontsize=10, y=1.02,
    )
    out3 = analysis_dir / "kernel_noise_gene_correlation_std.png"
    fig3.savefig(out3, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out3.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exp_dirs = sorted(EXPERIMENTS_ROOT.iterdir())
    for exp_dir in exp_dirs:
        if exp_dir.is_dir():
            process_experiment(exp_dir)
    print("\nAll done.")
