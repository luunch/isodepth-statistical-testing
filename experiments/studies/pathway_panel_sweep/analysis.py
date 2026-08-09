from __future__ import annotations

import argparse
import json

import numpy as np

from analysis.experiment_plots import save_target_vs_null_histogram
from experiments.core.study_io import scan_result_json_paths, write_csv
from experiments.studies.pathway_panel_sweep.lib import (
    analysis_dir_for_spec,
    extract_pathway_result_payload,
    load_manifest_entries,
    load_pathway_panel_sweep_spec,
    manifest_path_for_spec,
)

PER_PATHWAY_FIELDS = [
    "pathway_index",
    "pathway_name",
    "run_name",
    "gene_list_requested_count",
    "n_genes_surviving",
    "p_value",
    "significant",
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
    "pathway_name",
    "message",
]


def analyze_pathway_panel_sweep_results(spec_path: str) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

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
            collected_rows.append(record)

    collected_rows = sorted(
        collected_rows,
        key=lambda row: (float(row["p_value"]), str(row["pathway_name"])),
    )

    per_pathway_rows = [{key: row[key] for key in PER_PATHWAY_FIELDS} for row in collected_rows]
    significant_rows = [row for row in per_pathway_rows if bool(row["significant"])]

    per_pathway_path = analysis_dir / "per_pathway_results.csv"
    significant_path = analysis_dir / "significant_pathways.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_pathway_path, per_pathway_rows, fieldnames=PER_PATHWAY_FIELDS)
    write_csv(significant_path, significant_rows, fieldnames=PER_PATHWAY_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    p_values = np.asarray([float(row["p_value"]) for row in collected_rows], dtype=np.float64)
    stat_true = np.asarray([float(row["stat_true"]) for row in collected_rows], dtype=np.float64)

    hypoxia_rows = [
        row for row in collected_rows if str(row["pathway_name"]) == "HALLMARK_HYPOXIA"
    ]
    hypoxia_p = float(hypoxia_rows[0]["p_value"]) if hypoxia_rows else float("nan")
    hypoxia_stat = float(hypoxia_rows[0]["stat_true"]) if hypoxia_rows else float("nan")

    pvalue_plot_path = save_target_vs_null_histogram(
        p_values.tolist(),
        hypoxia_p,
        analysis_dir / "hallmark_pvalue_distribution_with_hypoxia_marked.png",
        title=f"Hallmark Pathway p-values (alpha={spec.alpha})",
        x_label="p-value",
        target_label="HALLMARK_HYPOXIA",
    )
    stat_plot_path = save_target_vs_null_histogram(
        stat_true.tolist(),
        hypoxia_stat,
        analysis_dir / "hallmark_stat_true_distribution_with_hypoxia_marked.png",
        title="Hallmark Pathway stat_true (lower is better)",
        x_label="stat_true",
        target_label="HALLMARK_HYPOXIA",
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "alpha": float(spec.alpha),
        "n_result_files_scanned": len(candidate_paths),
        "n_pathways_analyzed": len(collected_rows),
        "n_pathways_significant": len(significant_rows),
        "fraction_significant": float(len(significant_rows) / len(collected_rows))
        if collected_rows
        else None,
        "hypoxia_p_value": hypoxia_p,
        "hypoxia_stat_true": hypoxia_stat,
        "hypoxia_significant": bool(hypoxia_rows and hypoxia_rows[0]["significant"]),
        "significant_pathway_names": [str(row["pathway_name"]) for row in significant_rows],
        "per_pathway_results_csv": str(per_pathway_path),
        "significant_pathways_csv": str(significant_path),
        "analysis_warnings_csv": str(warnings_path),
        "hallmark_pvalue_distribution_plot": "" if pvalue_plot_path is None else str(pvalue_plot_path),
        "hallmark_stat_true_distribution_plot": "" if stat_plot_path is None else str(stat_plot_path),
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a Hallmark pathway-panel sweep.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_pathway_panel_sweep_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
