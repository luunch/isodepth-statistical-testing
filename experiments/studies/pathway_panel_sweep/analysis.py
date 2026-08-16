from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analysis.experiment_plots import save_value_histogram
from analysis.plots import _plot_spatial_isodepth
from experiments.core.study_io import scan_result_json_paths, write_csv
from experiments.studies.pathway_panel_sweep.lib import (
    REPO_ROOT,
    analysis_dir_for_spec,
    extract_pathway_result_payload,
    load_manifest_entries,
    load_pathway_panel_sweep_spec,
    manifest_path_for_spec,
)
from scripts.posthoc.postprocess_gsea_isodepth import _bh_qvalues

PER_PATHWAY_FIELDS = [
    "pathway_index",
    "pathway_name",
    "run_name",
    "gene_list_requested_count",
    "n_genes_surviving",
    "p_value",
    "q_value",
    "significant",
    "stat_true",
    "stat_true_per_gene",
    "null_mean",
    "null_std",
    "null_min",
    "null_max",
    "n_cells",
    "n_perms",
    "n_reruns",
    "runtime_sec",
    "spearman_vs_full",
    "result_json_path",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "pathway_name",
    "message",
]


def _load_isodepth_and_S(result_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_candidates = sorted(result_json_path.parent.rglob("*_isodepths.npz"))
    if not npz_candidates:
        raise FileNotFoundError(f"No *_isodepths.npz under {result_json_path.parent}")
    npz = np.load(npz_candidates[0], allow_pickle=False)
    if "true_isodepth" not in npz or "S" not in npz:
        raise KeyError(f"Missing true_isodepth/S in {npz_candidates[0]}")
    iso = np.asarray(npz["true_isodepth"], dtype=np.float64).reshape(-1)
    spatial = np.asarray(npz["S"], dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[0] != iso.shape[0]:
        raise ValueError(
            f"Bad S/isodepth shapes in {npz_candidates[0]}: S={spatial.shape}, iso={iso.shape}"
        )
    return iso, spatial


def _orient_to_reference(iso: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    rho, _ = spearmanr(iso, reference)
    rho = float(rho)
    if np.isfinite(rho) and rho < 0.0:
        return -iso, rho
    return iso, rho


def _save_spearman_matrix(
    labels: list[str],
    matrix: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> Path:
    n = len(labels)
    short = [lab.replace("HALLMARK_", "") for lab in labels]
    fig_w = max(10.0, 0.28 * n + 4.0)
    fig_h = max(8.0, 0.28 * n + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_pathway_isodepth_grid(
    *,
    spatial: np.ndarray,
    panel_specs: list[tuple[np.ndarray, str]],
    out_path: Path,
) -> Path:
    n_panels = len(panel_specs)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.8 * n_rows), squeeze=False)
    for axis, (depth_values, title) in zip(axes.flat, panel_specs):
        _plot_spatial_isodepth(axis, spatial, depth_values, title)
    for axis in axes.flat[n_panels:]:
        axis.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolve_reference_result(
    spec_path: Path,
    payload: Mapping[str, Any],
) -> Optional[Path]:
    raw = payload.get("reference_full_result_json")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def analyze_pathway_panel_sweep_results(
    spec_path: str,
    *,
    reference_full_result_json: str | Path | None = None,
) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(spec_path).resolve(), "r", encoding="utf-8") as handle:
        spec_payload = json.load(handle)

    reference_path: Optional[Path]
    if reference_full_result_json is not None:
        reference_path = Path(reference_full_result_json).resolve()
    else:
        reference_path = _resolve_reference_result(Path(spec_path).resolve(), spec_payload)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[str, dict[str, object] | None] = {
        str(path): entry for path, entry in manifest_entries.items()
    }
    for path in scan_result_json_paths(spec.output_root / "runs"):
        candidate_paths.setdefault(str(path), None)

    collected_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_pathway_result_payload(
            result_json_path,
            manifest_entry=manifest_entry,
            alpha=float(spec.alpha),
        )
        warning_rows.extend(warnings)
        if record is not None:
            n_genes = max(int(record["n_genes_surviving"]), 1)
            record["stat_true_per_gene"] = float(record["stat_true"]) / float(n_genes)
            record["q_value"] = float("nan")
            record["spearman_vs_full"] = float("nan")
            collected_rows.append(record)

    # BH q-values across pathways; significance uses q < alpha.
    if collected_rows:
        p_values = np.asarray([float(row["p_value"]) for row in collected_rows], dtype=np.float64)
        q_values = _bh_qvalues(p_values)
        for row, q in zip(collected_rows, q_values.tolist()):
            row["q_value"] = float(q)
            row["significant"] = bool(float(q) < float(spec.alpha))

    collected_rows = sorted(
        collected_rows,
        key=lambda row: (float(row["q_value"]), float(row["p_value"]), str(row["pathway_name"])),
    )

    # Optional: load matched full 3000-HVG isodepth and correlate / plot.
    spearman_matrix_path = ""
    spearman_csv_path = ""
    pathway_grid_path = ""
    reference_used = ""
    if reference_path is not None and reference_path.exists() and collected_rows:
        try:
            full_iso, full_S = _load_isodepth_and_S(reference_path)
            reference_used = str(reference_path)

            labels = ["FULL_3000HVG"]
            oriented: list[np.ndarray] = [full_iso]
            spatial_ref = full_S

            for row in collected_rows:
                iso, spatial = _load_isodepth_and_S(Path(str(row["result_json_path"])))
                if iso.shape[0] != full_iso.shape[0]:
                    warning_rows.append(
                        {
                            "warning_type": "cell_mismatch_vs_full",
                            "result_json_path": str(row["result_json_path"]),
                            "pathway_name": str(row["pathway_name"]),
                            "message": (
                                f"Skipping isodepth compare: pathway n={iso.shape[0]} "
                                f"vs full n={full_iso.shape[0]}"
                            ),
                        }
                    )
                    continue
                if not np.allclose(spatial, spatial_ref, atol=1e-5):
                    warning_rows.append(
                        {
                            "warning_type": "coordinate_mismatch_vs_full",
                            "result_json_path": str(row["result_json_path"]),
                            "pathway_name": str(row["pathway_name"]),
                            "message": "Pathway S does not match full-run S; refusing silent mismatch.",
                        }
                    )
                    continue
                oriented_iso, raw_rho = _orient_to_reference(iso, full_iso)
                # Store signed Spearman before orientation (model's native sign).
                row["spearman_vs_full"] = float(raw_rho)
                row["_oriented_abs_rho"] = float(abs(raw_rho)) if np.isfinite(raw_rho) else float("nan")
                labels.append(str(row["pathway_name"]))
                oriented.append(oriented_iso)

            # Matrix/plot: FULL reference + significant pathways only.
            significant_names = {
                str(row["pathway_name"])
                for row in collected_rows
                if bool(row.get("significant"))
            }
            matrix_keep_idx = [0] + [
                i for i, lab in enumerate(labels) if i > 0 and lab in significant_names
            ]
            matrix_labels = [labels[i] for i in matrix_keep_idx]
            matrix_oriented = [oriented[i] for i in matrix_keep_idx]
            n_matrix = len(matrix_labels)
            matrix = np.eye(n_matrix, dtype=np.float64)
            for i in range(n_matrix):
                for j in range(i + 1, n_matrix):
                    rho, _ = spearmanr(matrix_oriented[i], matrix_oriented[j])
                    matrix[i, j] = float(rho)
                    matrix[j, i] = float(rho)

            spearman_csv_path_obj = analysis_dir / "isodepth_spearman_matrix.csv"
            with spearman_csv_path_obj.open("w", encoding="utf-8") as handle:
                handle.write("," + ",".join(matrix_labels) + "\n")
                for i, lab in enumerate(matrix_labels):
                    vals = ",".join(f"{matrix[i, j]:.6g}" for j in range(n_matrix))
                    handle.write(f"{lab},{vals}\n")
            spearman_csv_path = str(spearman_csv_path_obj)

            spearman_matrix_path = str(
                _save_spearman_matrix(
                    matrix_labels,
                    matrix,
                    analysis_dir / "isodepth_spearman_matrix.png",
                    title=(
                        "Isodepth Spearman correlations "
                        f"(significant pathways only, q<{float(spec.alpha):g}; "
                        "oriented to FULL_3000HVG)"
                    ),
                )
            )

            # Spatial grid: full first, then all pathways (oriented), labeled like rerun grid.
            panel_specs: list[tuple[np.ndarray, str]] = [
                (
                    oriented[0],
                    "FULL_3000HVG\n(reference)",
                )
            ]
            for row in collected_rows:
                name = str(row["pathway_name"])
                if name not in labels:
                    continue
                idx = labels.index(name)
                short = name.replace("HALLMARK_", "")
                title = (
                    f"{short}\n"
                    f"p={float(row['p_value']):.3g}, q={float(row['q_value']):.3g}\n"
                    f"|ρ(full)|={float(row.get('_oriented_abs_rho', abs(float(row['spearman_vs_full'])))):.3f}"
                )
                panel_specs.append((oriented[idx], title))

            pathway_grid_path = str(
                _save_pathway_isodepth_grid(
                    spatial=spatial_ref,
                    panel_specs=panel_specs,
                    out_path=analysis_dir / "pathway_isodepths_grid.png",
                )
            )
        except Exception as exc:
            warning_rows.append(
                {
                    "warning_type": "reference_isodepth_analysis_failed",
                    "result_json_path": str(reference_path),
                    "pathway_name": "",
                    "message": str(exc),
                }
            )
    elif reference_path is not None and not reference_path.exists():
        warning_rows.append(
            {
                "warning_type": "missing_reference_full_result",
                "result_json_path": str(reference_path),
                "pathway_name": "",
                "message": "reference_full_result_json not found; skipping isodepth matrix/grid",
            }
        )

    per_pathway_rows = [{key: row.get(key) for key in PER_PATHWAY_FIELDS} for row in collected_rows]
    significant_rows = [row for row in per_pathway_rows if bool(row["significant"])]

    per_pathway_path = analysis_dir / "per_pathway_results.csv"
    significant_path = analysis_dir / "significant_pathways.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_pathway_path, per_pathway_rows, fieldnames=PER_PATHWAY_FIELDS)
    write_csv(significant_path, significant_rows, fieldnames=PER_PATHWAY_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    q_values_arr = np.asarray([float(row["q_value"]) for row in collected_rows], dtype=np.float64)
    stat_per_gene = np.asarray(
        [float(row["stat_true_per_gene"]) for row in collected_rows], dtype=np.float64
    )

    qvalue_plot_path = save_value_histogram(
        q_values_arr.tolist(),
        analysis_dir / "hallmark_qvalue_distribution.png",
        title=f"Hallmark Pathway BH q-values (alpha={spec.alpha})",
        x_label="q-value",
    )
    stat_plot_path = save_value_histogram(
        stat_per_gene.tolist(),
        analysis_dir / "hallmark_stat_true_per_gene_distribution.png",
        title="Hallmark Pathway stat_true / n_genes",
        x_label="stat_true_per_gene",
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "alpha": float(spec.alpha),
        "significance_rule": "bh_q_value < alpha",
        "n_result_files_scanned": len(candidate_paths),
        "n_pathways_analyzed": len(collected_rows),
        "n_pathways_significant": len(significant_rows),
        "fraction_significant": float(len(significant_rows) / len(collected_rows))
        if collected_rows
        else None,
        "reference_full_result_json": reference_used,
        "significant_pathway_names": [str(row["pathway_name"]) for row in significant_rows],
        "per_pathway_results_csv": str(per_pathway_path),
        "significant_pathways_csv": str(significant_path),
        "analysis_warnings_csv": str(warnings_path),
        "hallmark_qvalue_distribution_plot": "" if qvalue_plot_path is None else str(qvalue_plot_path),
        "hallmark_stat_true_per_gene_distribution_plot": ""
        if stat_plot_path is None
        else str(stat_plot_path),
        "isodepth_spearman_matrix_csv": spearman_csv_path,
        "isodepth_spearman_matrix_plot": spearman_matrix_path,
        "pathway_isodepths_grid_plot": pathway_grid_path,
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Hallmark pathway-panel sweep.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    parser.add_argument(
        "--reference-full-result-json",
        default=None,
        help="Matched full-transcriptome (e.g. 3000 HVG) result JSON for isodepth comparisons",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_pathway_panel_sweep_results(
        args.spec,
        reference_full_result_json=args.reference_full_result_json,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
