from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.experiment_plots import (
    save_repeat_null_boxplot,
    save_repeat_null_density_overlay,
    save_repeat_pvalue_plot,
    save_repeat_true_loss_plot,
    save_spearman_histogram,
    save_spearman_matrix_heatmap,
)
from experiments.real_data_existence_consistency import (
    analysis_dir_for_spec,
    build_matrix_rows,
    build_null_rows,
    build_pairwise_isodepth_rows,
    extract_repeat_payload,
    load_manifest_entries,
    load_real_data_existence_consistency_spec,
    manifest_path_for_spec,
    scan_result_json_paths,
    write_csv,
)


PER_RUN_FIELDS = [
    "repeat_index",
    "seed",
    "p_value",
    "stat_true",
    "null_mean",
    "null_std",
    "null_min",
    "null_max",
    "runtime_sec",
    "result_json_path",
]

NULL_FIELDS = [
    "repeat_index",
    "seed",
    "null_sample_index",
    "null_loss",
]

PAIRWISE_FIELDS = [
    "repeat_i",
    "repeat_j",
    "seed_i",
    "seed_j",
    "spearman_true_isodepth",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def _off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.asarray([], dtype=np.float64)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.asarray(matrix[mask], dtype=np.float64)


def analyze_real_data_existence_consistency_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_real_data_existence_consistency_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {path: entry for path, entry in manifest_entries.items()}
    for root in spec.reuse_result_roots:
        for path in scan_result_json_paths(root):
            candidate_paths.setdefault(path, None)

    collected_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_repeat_payload(result_json_path, manifest_entry=manifest_entry)
        warning_rows.extend(warnings)
        if record is not None:
            collected_rows.append(record)

    collected_rows = sorted(
        collected_rows,
        key=lambda row: (
            int(row["repeat_index"]) if int(row["repeat_index"]) >= 0 else 10**9,
            int(row["seed"]),
            str(row["run_name"]),
        ),
    )
    for fallback_index, row in enumerate(collected_rows):
        if int(row["repeat_index"]) < 0:
            row["repeat_index"] = int(fallback_index)

    per_run_rows = [
        {key: row[key] for key in PER_RUN_FIELDS}
        for row in collected_rows
    ]
    null_rows = build_null_rows(collected_rows)
    pairwise_rows, matrix = build_pairwise_isodepth_rows(collected_rows) if collected_rows else ([], np.zeros((0, 0)))
    matrix_rows = build_matrix_rows(collected_rows, matrix) if collected_rows else []
    matrix_fieldnames = ["repeat_index"] + [str(int(row["repeat_index"])) for row in collected_rows]

    p_values = np.asarray([float(row["p_value"]) for row in collected_rows], dtype=np.float64)
    true_losses = np.asarray([float(row["stat_true"]) for row in collected_rows], dtype=np.float64)
    off_diagonal = _off_diagonal_values(matrix)

    per_run_path = analysis_dir / "per_run_results.csv"
    null_path = analysis_dir / "null_losses_long.csv"
    pairwise_path = analysis_dir / "pairwise_isodepth_spearman.csv"
    matrix_path = analysis_dir / "true_isodepth_spearman_matrix.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"

    write_csv(per_run_path, per_run_rows, fieldnames=PER_RUN_FIELDS)
    write_csv(null_path, null_rows, fieldnames=NULL_FIELDS)
    write_csv(pairwise_path, pairwise_rows, fieldnames=PAIRWISE_FIELDS)
    write_csv(matrix_path, matrix_rows, fieldnames=matrix_fieldnames)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    pvalue_plot_path = save_repeat_pvalue_plot(per_run_rows, analysis_dir / "pvalue_by_repeat.png")
    true_loss_plot_path = save_repeat_true_loss_plot(per_run_rows, analysis_dir / "true_loss_by_repeat.png")
    null_boxplot_path = save_repeat_null_boxplot(
        null_rows,
        per_run_rows,
        analysis_dir / "null_loss_distributions_by_repeat.png",
    )
    null_density_path = save_repeat_null_density_overlay(
        null_rows,
        per_run_rows,
        analysis_dir / "null_loss_density_overlay.png",
    )
    spearman_matrix_path = save_spearman_matrix_heatmap(
        matrix,
        analysis_dir / "true_isodepth_spearman_matrix.png",
    )
    spearman_hist_path = save_spearman_histogram(
        off_diagonal.tolist(),
        analysis_dir / "isodepth_spearman_histogram.png",
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_runs_analyzed": len(collected_rows),
        "n_warnings": len(warning_rows),
        "seeds_analyzed": [int(row["seed"]) for row in collected_rows],
        "per_run_results_csv": str(per_run_path),
        "null_losses_long_csv": str(null_path),
        "pairwise_isodepth_spearman_csv": str(pairwise_path),
        "true_isodepth_spearman_matrix_csv": str(matrix_path),
        "analysis_warnings_csv": str(warnings_path),
        "pvalue_mean": float(np.mean(p_values)) if p_values.size else None,
        "pvalue_std": float(np.std(p_values)) if p_values.size else None,
        "pvalue_min": float(np.min(p_values)) if p_values.size else None,
        "pvalue_max": float(np.max(p_values)) if p_values.size else None,
        "true_loss_mean": float(np.mean(true_losses)) if true_losses.size else None,
        "true_loss_std": float(np.std(true_losses)) if true_losses.size else None,
        "true_loss_min": float(np.min(true_losses)) if true_losses.size else None,
        "true_loss_max": float(np.max(true_losses)) if true_losses.size else None,
        "off_diagonal_spearman_mean": float(np.mean(off_diagonal)) if off_diagonal.size else None,
        "off_diagonal_spearman_std": float(np.std(off_diagonal)) if off_diagonal.size else None,
        "off_diagonal_spearman_min": float(np.min(off_diagonal)) if off_diagonal.size else None,
        "off_diagonal_spearman_max": float(np.max(off_diagonal)) if off_diagonal.size else None,
        "pvalue_by_repeat_plot": "" if pvalue_plot_path is None else str(pvalue_plot_path),
        "true_loss_by_repeat_plot": "" if true_loss_plot_path is None else str(true_loss_plot_path),
        "null_loss_distributions_by_repeat_plot": "" if null_boxplot_path is None else str(null_boxplot_path),
        "null_loss_density_overlay_plot": "" if null_density_path is None else str(null_density_path),
        "true_isodepth_spearman_matrix_plot": "" if spearman_matrix_path is None else str(spearman_matrix_path),
        "isodepth_spearman_histogram_plot": "" if spearman_hist_path is None else str(spearman_hist_path),
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a real-data existence consistency study.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_real_data_existence_consistency_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
