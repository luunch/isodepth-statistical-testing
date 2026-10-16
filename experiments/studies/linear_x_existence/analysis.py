from __future__ import annotations

import json
from pathlib import Path

from analysis.experiment_plots import save_pvalue_vs_signal_fraction_plot
from experiments.configuration import build_run_config
from experiments.core.study_io import scan_result_json_paths, write_csv
from experiments.core.study_spec import analysis_arg_parser
from experiments.studies.linear_x_existence.lib import (
    analysis_dir_for_spec,
    extract_linear_x_record,
    load_linear_x_existence_spec,
    load_manifest_entries,
    manifest_path_for_spec,
    summarize_signal_fraction_rows,
)


PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "method_name",
    "metric",
    "signal_fraction",
    "sigma",
    "seed",
    "p_value",
    "stat_true",
    "runtime_sec",
    "truth_source",
]

SUMMARY_FIELDS = [
    "signal_fraction",
    "n_runs",
    "p_value_mean",
    "p_value_std",
    "p_value_min",
    "p_value_max",
    "n_p_below_alpha",
    "alpha",
    "stat_true_mean",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def analyze_linear_x_existence_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_linear_x_existence_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_run_config = build_run_config(str(spec.base_config), {})
    expected_sigma = float(base_run_config.data.sigma)
    alpha = float(base_run_config.test.alpha)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {path: entry for path, entry in manifest_entries.items()}
    for root in spec.reuse_result_roots:
        for path in scan_result_json_paths(root):
            candidate_paths.setdefault(path, None)

    per_run_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_linear_x_record(
            result_json_path,
            expected_sigma=expected_sigma,
            allowed_fractions=list(spec.signal_fractions),
            manifest_entry=manifest_entry,
        )
        warning_rows.extend(warnings)
        if record is not None:
            per_run_rows.append(record)

    summary_rows = summarize_signal_fraction_rows(per_run_rows, alpha=alpha)

    per_run_path = analysis_dir / "per_run_results.csv"
    summary_path = analysis_dir / "summary_by_signal_fraction.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_run_path, per_run_rows, fieldnames=PER_RUN_FIELDS)
    write_csv(summary_path, summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    pvalue_plot_path = save_pvalue_vs_signal_fraction_plot(
        per_run_rows,
        summary_rows,
        analysis_dir / "pvalue_vs_signal_fraction.png",
        title="Linear-x existence p-value vs signal fraction",
        alpha=alpha,
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_rows_analyzed": len(per_run_rows),
        "n_warnings": len(warning_rows),
        "per_run_results_csv": str(per_run_path),
        "summary_by_signal_fraction_csv": str(summary_path),
        "analysis_warnings_csv": str(warnings_path),
        "pvalue_vs_signal_fraction_plot": "" if pvalue_plot_path is None else str(pvalue_plot_path),
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser():
    return analysis_arg_parser("Analyze linear-x existence sweep p-values.")


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_linear_x_existence_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
