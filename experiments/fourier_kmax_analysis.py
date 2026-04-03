from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.experiment_plots import save_pvalue_vs_kmax_plot
from experiments.configuration import build_run_config
from experiments.fourier_kmax import (
    analysis_dir_for_spec,
    extract_kmax_record,
    load_fourier_kmax_spec,
    load_manifest_entries,
    manifest_path_for_spec,
    scan_result_json_paths,
    summarize_kmax_rows,
    write_csv,
)


PER_RUN_FIELDS = [
    "result_json_path",
    "run_name",
    "method_name",
    "metric",
    "k_min",
    "k_max",
    "sigma",
    "seed",
    "dependent_xy",
    "poly_degree",
    "truth_source",
    "p_value",
    "stat_true",
    "runtime_sec",
]

SUMMARY_FIELDS = [
    "k_max",
    "n_runs",
    "p_value_mean",
    "p_value_std",
    "p_value_min",
    "p_value_max",
    "stat_true_mean",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def analyze_fourier_kmax_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_fourier_kmax_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    base_run_config = build_run_config(str(spec.base_config), {})
    expected_sigma = float(base_run_config.data.sigma)
    expected_dependent_xy = bool(base_run_config.data.dependent_xy)
    expected_poly_degree = int(base_run_config.data.poly_degree)

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {path: entry for path, entry in manifest_entries.items()}
    for root in spec.reuse_result_roots:
        for path in scan_result_json_paths(root):
            candidate_paths.setdefault(path, None)

    per_run_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        record, warnings = extract_kmax_record(
            result_json_path,
            expected_sigma=expected_sigma,
            expected_dependent_xy=expected_dependent_xy,
            expected_poly_degree=expected_poly_degree,
            manifest_entry=manifest_entry,
        )
        warning_rows.extend(warnings)
        if record is not None:
            per_run_rows.append(record)

    summary_rows = summarize_kmax_rows(per_run_rows)

    per_run_path = analysis_dir / "per_run_results.csv"
    summary_path = analysis_dir / "summary_by_kmax.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"
    write_csv(per_run_path, per_run_rows, fieldnames=PER_RUN_FIELDS)
    write_csv(summary_path, summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    pvalue_plot_path = save_pvalue_vs_kmax_plot(
        per_run_rows,
        summary_rows,
        analysis_dir / "pvalue_vs_kmax.png",
        title="Fourier Existence p-value vs k_max",
    )

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_rows_analyzed": len(per_run_rows),
        "n_warnings": len(warning_rows),
        "per_run_results_csv": str(per_run_path),
        "summary_by_kmax_csv": str(summary_path),
        "analysis_warnings_csv": str(warnings_path),
        "pvalue_vs_kmax_plot": "" if pvalue_plot_path is None else str(pvalue_plot_path),
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Fourier k_max existence sweep results.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_fourier_kmax_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
