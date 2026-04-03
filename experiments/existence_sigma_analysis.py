from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.experiment_plots import (
    save_fourier_heatmap,
    save_null_pvalue_histograms,
    save_rate_vs_sigma_plot,
)
from experiments.existence_sigma import (
    analysis_dir_for_spec,
    extract_result_record,
    load_existence_sigma_spec,
    load_manifest_entries,
    manifest_path_for_spec,
    scan_result_json_paths,
    summarize_condition_rows,
    write_csv,
)


PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "method_name",
    "metric",
    "mode",
    "k",
    "sigma",
    "seed",
    "truth_label",
    "null_family",
    "family_label",
    "p_value",
    "reject",
    "stat_true",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "summary_metric",
    "truth_label",
    "mode",
    "family_label",
    "sigma",
    "k",
    "alpha",
    "n_runs",
    "n_reject",
    "rate",
    "standard_error",
    "ci_lower",
    "ci_upper",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def analyze_existence_sigma_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_existence_sigma_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {path: entry for path, entry in manifest_entries.items()}
    for root in spec.reuse_result_roots:
        for path in scan_result_json_paths(root):
            candidate_paths.setdefault(path, None)

    per_run_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_result_record(result_json_path, manifest_entry=manifest_entry)
        warning_rows.extend(warnings)
        if record is not None:
            record["reject"] = int(float(record["p_value"]) < float(spec.alpha))
            per_run_rows.append(record)

    summary_rows = summarize_condition_rows(per_run_rows, alpha=spec.alpha)

    per_run_path = analysis_dir / "per_run_results.csv"
    summary_path = analysis_dir / "summary_by_condition.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_run_path, per_run_rows, fieldnames=PER_RUN_FIELDS)
    write_csv(summary_path, summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    save_rate_vs_sigma_plot(
        summary_rows,
        analysis_dir / "power_vs_sigma.png",
        truth_label="alternative",
        title="Existence Power vs Sigma",
        y_label="Power",
    )
    save_rate_vs_sigma_plot(
        summary_rows,
        analysis_dir / "null_rejection_vs_sigma.png",
        truth_label="null",
        title="Null Rejection Rate vs Sigma",
        y_label="Null Rejection Rate",
    )
    save_fourier_heatmap(
        summary_rows,
        analysis_dir / "fourier_power_heatmap.png",
        truth_label="alternative",
        title="Fourier Power Heatmap",
    )
    save_fourier_heatmap(
        summary_rows,
        analysis_dir / "fourier_null_rejection_heatmap.png",
        truth_label="null",
        title="Fourier Null Rejection Heatmap",
    )
    histogram_paths = save_null_pvalue_histograms(per_run_rows, analysis_dir / "pvalue_histograms")

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_rows_analyzed": len(per_run_rows),
        "n_warnings": len(warning_rows),
        "per_run_results_csv": str(per_run_path),
        "summary_by_condition_csv": str(summary_path),
        "analysis_warnings_csv": str(warnings_path),
        "histogram_paths": [str(path) for path in histogram_paths],
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze existence sigma sweep results.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_existence_sigma_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
