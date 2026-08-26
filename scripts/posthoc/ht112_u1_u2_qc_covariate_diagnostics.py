"""QC-covariate-vs-isodepth diagnostic plots for HT112C1 U1 and U2 cropx runs.

Produces the same layout as ``loss_diff_clone2_linear_qc_covariate_diagnostics``:
scatter of ``pct_mt`` / ``total_counts`` vs ``true_isodepth`` (with Spearman rho/p),
plus spatial maps of isodepth, pct_mt, and total_counts.

Target runs (tumor-proportion-only whitening, matching the reference figure's
confounding check):

- U1: ``results/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_gt0p7/loss_diff_gt0p7_cropx``
  (obs filter gt0.7 + spatial_crop x>-3; no denoise)
- U2: ``results/calicost/HT112C1_U2_loss_difference_gt0p7/loss_diff_gt0p7_cropx``
  (obs filter gt0.7 + spatial_denoise 300 um; no crop — despite the run name)

Saved result JSONs omit some of these keys from ``config.data``, so this script
reloads via the sibling config JSON with ``data`` patches taken from each run's
``artifacts.dataset_meta`` (and the denoise/crop overrides above), then verifies
per-clone cell counts and per-clone re-standardized ``S`` against the NPZ before
trusting the alignment.

Usage (from repo root, isodepth_env):
    python -m scripts.posthoc.ht112_u1_u2_qc_covariate_diagnostics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments.core.paths import repo_root

REPO = repo_root(__file__)
sys.path.insert(0, str(REPO))

from data import load_dataset  # noqa: E402
from data.schemas import run_config_from_mapping  # noqa: E402
from experiments.configuration import load_json_config  # noqa: E402

COVARIATES = [
    ("pct_mt", "pct_mt (%)"),
    ("total_counts", "total_counts"),
]

# (label, config_path, result_dir, spatial_crop override, denoise_um override)
RUNS: list[dict[str, Any]] = [
    dict(
        label="HT112C1_U1",
        config="configs/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_cropx.json",
        result_dir="results/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_gt0p7/loss_diff_gt0p7_cropx",
        spatial_crop={"x": {"gt": -3}},
        spatial_denoise_radius_um=None,
    ),
    dict(
        label="HT112C1_U2",
        config="configs/calicost/HT112C1_U2_loss_difference_cropx.json",
        result_dir="results/calicost/HT112C1_U2_loss_difference_gt0p7/loss_diff_gt0p7_cropx",
        spatial_crop=None,
        spatial_denoise_radius_um=300.0,
    ),
]


def _whitening_title(cw: Any) -> str:
    if isinstance(cw, dict):
        key = cw.get("obs_key", cw.get("obs_keys"))
    else:
        key = cw
    if isinstance(key, (list, tuple)):
        key_str = ", ".join(str(k) for k in key)
    else:
        key_str = str(key)
    if key_str == "calicost_tumor_proportion":
        return "covariate whitening: calicost_tumor_proportion only (not total_counts)"
    return f"covariate whitening: {key_str}"


def _qc_from_raw_counts(A: np.ndarray, var_names: list[str]) -> dict[str, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    total_counts = A.sum(axis=1)
    mt_mask = np.array([str(v).upper().startswith("MT-") for v in var_names], dtype=bool)
    mt_counts = A[:, mt_mask].sum(axis=1) if mt_mask.any() else np.zeros(A.shape[0])
    pct_mt = np.where(total_counts > 0, 100.0 * mt_counts / total_counts, 0.0)
    return {
        "total_counts": total_counts,
        "pct_mt": pct_mt,
    }


def _plot_clone(
    *,
    out_path: Path,
    title: str,
    S: np.ndarray,
    true_isodepth: np.ndarray,
    covariate_values: dict[str, np.ndarray],
) -> None:
    n_cov = len(COVARIATES)
    n_spatial = 1 + n_cov
    ncols = 2 * n_spatial
    fig = plt.figure(figsize=(3.0 * ncols, 8.5))
    gs = fig.add_gridspec(2, ncols)

    scatter_span = ncols // n_cov
    for col, (name, xlabel) in enumerate(COVARIATES):
        x = covariate_values[name]
        ax = fig.add_subplot(gs[0, col * scatter_span : (col + 1) * scatter_span])
        ax.scatter(x, true_isodepth, s=14, alpha=0.6)
        rho, p = spearmanr(x, true_isodepth)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("true_isodepth")
        ax.set_title(f"{name}: rho={rho:.2f}, p={p:.2g}")

    spatial_span = ncols // n_spatial
    spatial_panels = [("isodepth", true_isodepth)] + [
        (name, covariate_values[name]) for name, _ in COVARIATES
    ]
    for col, (name, values) in enumerate(spatial_panels):
        ax = fig.add_subplot(gs[1, col * spatial_span : (col + 1) * spatial_span])
        sc = ax.scatter(S[:, 0], S[:, 1], c=values, s=14, cmap="viridis")
        ax.set_title(name)
        fig.colorbar(sc, ax=ax)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _process_run(spec: dict[str, Any]) -> list[Path]:
    config_path = REPO / spec["config"]
    result_dir = REPO / spec["result_dir"]
    result_json = next(result_dir.glob("*_result.json"))
    with result_json.open("r", encoding="utf-8") as f:
        result = json.load(f)
    meta = result["artifacts"]["dataset_meta"]
    run_name = result_dir.name
    cw = result.get("config", {}).get("data", {}).get("covariate_whitening")
    whitening_note = _whitening_title(cw)
    per_type = result.get("artifacts", {}).get("per_type_summaries") or {}

    cfg = load_json_config(str(config_path))
    run_cfg = run_config_from_mapping(cfg)
    run_cfg.data.obs_numeric_filters = meta.get("obs_numeric_filters")
    run_cfg.data.spatial_crop = spec["spatial_crop"]
    run_cfg.data.spatial_denoise_radius_um = spec["spatial_denoise_radius_um"]
    um = meta.get("coordinate_um_per_unit") or getattr(run_cfg.test, "coordinate_um_per_unit", None)
    run_cfg.data.coordinate_um_per_unit = um

    print(f"[load] {spec['label']} <- {config_path.name} ({run_name})", flush=True)
    dataset = load_dataset(run_cfg.data)
    if int(dataset.S.shape[0]) != int(result["n_cells"]):
        raise RuntimeError(
            f"{spec['label']}: reloaded n_cells={dataset.S.shape[0]} != result n_cells={result['n_cells']}"
        )

    labels = np.asarray(dataset.meta["cell_type_labels"], dtype=np.int64)
    names = [str(n) for n in dataset.meta["cell_type_names"]]
    var_names = dataset.meta.get("var_names")
    if var_names is None:
        var_names = [f"gene_{i}" for i in range(dataset.A.shape[1])]
    var_names = [str(v) for v in var_names]
    qc_all = _qc_from_raw_counts(dataset.A, var_names)

    written: list[Path] = []
    for type_index, type_name in enumerate(names):
        mask = labels == type_index
        n_c = int(mask.sum())
        npz_path = result_dir / type_name / f"{type_name}_isodepths.npz"
        if not npz_path.exists():
            print(f"  [skip] missing NPZ for clone {type_name}", flush=True)
            continue
        npz = np.load(npz_path)
        true_isodepth = np.asarray(npz["true_isodepth"], dtype=np.float64)
        S_npz = np.asarray(npz["S"], dtype=np.float64)
        if true_isodepth.shape[0] != n_c:
            raise RuntimeError(
                f"{spec['label']} clone {type_name}: isodepth n={true_isodepth.shape[0]} != reload n={n_c}"
            )
        S_c = np.asarray(dataset.S[mask], dtype=np.float64)
        S_rez = (S_c - S_c.mean(axis=0)) / np.maximum(S_c.std(axis=0), 1e-8)
        if not np.allclose(S_npz, S_rez, atol=1e-3):
            raise RuntimeError(
                f"{spec['label']} clone {type_name}: per-clone S alignment failed "
                f"(max |diff|={np.max(np.abs(S_npz - S_rez)):.4g})"
            )

        covariate_values = {name: np.asarray(qc_all[name][mask], dtype=np.float64) for name, _ in COVARIATES}
        summary = per_type.get(type_name) or per_type.get(str(type_name)) or {}
        p_value = summary.get("p_value", result.get("p_value"))
        try:
            p_str = f"{float(p_value):.3g}"
        except (TypeError, ValueError):
            p_str = str(p_value)

        out_path = result_dir / type_name / f"{type_name}_pct_mt_total_counts_diagnostics.png"
        title = (
            f"{run_name} | clone {type_name} (n={n_c}, p={p_str}) | {whitening_note}"
        )
        _plot_clone(
            out_path=out_path,
            title=title,
            S=S_npz,
            true_isodepth=true_isodepth,
            covariate_values=covariate_values,
        )
        rho_mt, p_mt = spearmanr(covariate_values["pct_mt"], true_isodepth)
        rho_tc, p_tc = spearmanr(covariate_values["total_counts"], true_isodepth)
        print(
            f"  [write] {out_path.name}  "
            f"rho(pct_mt)={rho_mt:.2f} (p={p_mt:.2g})  "
            f"rho(total_counts)={rho_tc:.2f} (p={p_tc:.2g})",
            flush=True,
        )
        written.append(out_path)
    return written


def main() -> None:
    all_written: list[Path] = []
    for spec in RUNS:
        all_written.extend(_process_run(spec))
    print(f"[done] wrote {len(all_written)} figure(s)")


if __name__ == "__main__":
    main()
