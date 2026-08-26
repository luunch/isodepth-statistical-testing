from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.experiment_plots import save_target_vs_null_histogram
from experiments.core.study_io import scan_result_json_paths, write_csv
from experiments.studies.random_gene_panel_null.lib import (
    analysis_dir_for_spec,
    compute_target_rank,
    extract_panel_result_payload,
    load_manifest_entries,
    load_random_gene_panel_null_spec,
    manifest_path_for_spec,
)

PER_RUN_FIELDS = [
    "condition_type",
    "panel_index",
    "panel_seed",
    "run_name",
    "gene_list_requested_count",
    "n_genes_surviving",
    "p_value",
    "stat_true",
    "null_mean",
    "null_std",
    "null_min",
    "null_max",
    "n_cells",
    "n_perms",
    "n_reruns",
    "runtime_sec",
    "result_json_path",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def analyze_random_gene_panel_null_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_random_gene_panel_null_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {
        path: entry for path, entry in manifest_entries.items()
    }
    for path in scan_result_json_paths(spec.output_root / "runs"):
        candidate_paths.setdefault(path, None)

    collected_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_panel_result_payload(
            result_json_path,
            manifest_entry=manifest_entry,
        )
        warning_rows.extend(warnings)
        if record is not None:
            collected_rows.append(record)

    collected_rows = sorted(
        collected_rows,
        key=lambda row: (
            0 if row["condition_type"] == "target" else 1,
            int(row["panel_index"]) if int(row["panel_index"]) >= 0 else -1,
            int(row["panel_seed"]),
            str(row["run_name"]),
        ),
    )

    per_run_rows = [{key: row[key] for key in PER_RUN_FIELDS} for row in collected_rows]
    target_rows = [row for row in collected_rows if row["condition_type"] == "target"]
    random_rows = [row for row in collected_rows if row["condition_type"] == "random_panel"]

    target_p_value = float(target_rows[0]["p_value"]) if target_rows else float("nan")
    target_stat_true = float(target_rows[0]["stat_true"]) if target_rows else float("nan")
    random_p_values = np.asarray([float(row["p_value"]) for row in random_rows], dtype=np.float64)
    random_stat_true = np.asarray([float(row["stat_true"]) for row in random_rows], dtype=np.float64)

    p_value_rank = compute_target_rank(target_p_value, random_p_values, lower_is_better=True)
    stat_true_rank = compute_target_rank(target_stat_true, random_stat_true, lower_is_better=True)

    per_run_path = analysis_dir / "per_run_results.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_run_path, per_run_rows, fieldnames=PER_RUN_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    pvalue_plot_path = save_target_vs_null_histogram(
        random_p_values.tolist(),
        target_p_value,
        analysis_dir / "target_vs_random_pvalue_histogram.png",
        title="Hypoxia Target p-value vs Random 200-Gene Panels",
        x_label="p-value",
        target_label="Hypoxia target",
    )
    stat_true_plot_path = save_target_vs_null_histogram(
        random_stat_true.tolist(),
        target_stat_true,
        analysis_dir / "target_vs_random_stat_true_histogram.png",
        title="Hypoxia Target stat_true vs Random 200-Gene Panels",
        x_label="stat_true (lower is better)",
        target_label="Hypoxia target",
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_runs_analyzed": len(collected_rows),
        "n_random_panels_analyzed": len(random_rows),
        "n_warnings": len(warning_rows),
        "target_p_value": target_p_value,
        "target_stat_true": target_stat_true,
        "random_p_value_mean": float(np.mean(random_p_values)) if random_p_values.size else None,
        "random_p_value_std": float(np.std(random_p_values)) if random_p_values.size else None,
        "random_p_value_min": float(np.min(random_p_values)) if random_p_values.size else None,
        "random_p_value_max": float(np.max(random_p_values)) if random_p_values.size else None,
        "random_stat_true_mean": float(np.mean(random_stat_true)) if random_stat_true.size else None,
        "random_stat_true_std": float(np.std(random_stat_true)) if random_stat_true.size else None,
        "random_stat_true_min": float(np.min(random_stat_true)) if random_stat_true.size else None,
        "random_stat_true_max": float(np.max(random_stat_true)) if random_stat_true.size else None,
        "target_p_value_rank_among_random": int(p_value_rank["rank"]),
        "target_p_value_percentile_among_random": float(p_value_rank["percentile"]),
        "target_stat_true_rank_among_random": int(stat_true_rank["rank"]),
        "target_stat_true_percentile_among_random": float(stat_true_rank["percentile"]),
        "per_run_results_csv": str(per_run_path),
        "analysis_warnings_csv": str(warnings_path),
        "target_vs_random_pvalue_histogram": "" if pvalue_plot_path is None else str(pvalue_plot_path),
        "target_vs_random_stat_true_histogram": "" if stat_true_plot_path is None else str(stat_true_plot_path),
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a random gene-panel null specificity study.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_random_gene_panel_null_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
