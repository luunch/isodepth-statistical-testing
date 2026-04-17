from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.experiment_plots import (
    save_boxplots_by_kmax,
    save_grouped_pvalue_vs_kmax_plot,
    save_pvalue_vs_kmax_plot,
)
from experiments.configuration import build_run_config
from experiments.fourier_kmax_existence_perturbation import (
    analysis_dir_for_spec,
    build_boxplot_rows,
    extract_existence_record,
    extract_perturbation_records,
    load_fourier_kmax_existence_perturbation_spec,
    load_manifest_entries,
    manifest_path_for_spec,
    scan_result_json_paths,
    summarize_existence_rows,
    summarize_perturbation_rows,
    write_csv,
)


EXISTENCE_FIELDS = [
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

PERTURBATION_FIELDS = [
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
    "delta",
    "p_value",
    "stat_true",
    "runtime_sec",
]

EXISTENCE_SUMMARY_FIELDS = [
    "k_max",
    "n_runs",
    "p_value_mean",
    "p_value_std",
    "p_value_min",
    "p_value_max",
    "stat_true_mean",
]

PERTURBATION_SUMMARY_FIELDS = [
    "k_max",
    "delta",
    "n_runs",
    "p_value_mean",
    "p_value_std",
    "p_value_min",
    "p_value_max",
    "stat_true_mean",
]

BOXPLOT_FIELDS = [
    "k_max",
    "test_label",
    "test_kind",
    "delta",
    "seed",
    "p_value",
]

WARNING_FIELDS = [
    "warning_type",
    "result_json_path",
    "run_name",
    "message",
]


def analyze_fourier_kmax_existence_perturbation_results(spec_path: str | Path) -> dict[str, object]:
    spec = load_fourier_kmax_existence_perturbation_spec(spec_path)
    analysis_dir = analysis_dir_for_spec(spec)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    existence_base_run_config = build_run_config(str(spec.existence_base_config), {})
    perturbation_base_run_config = build_run_config(str(spec.perturbation_base_config), {})

    manifest_entries = load_manifest_entries(manifest_path_for_spec(spec))
    candidate_paths: dict[Path, dict[str, object] | None] = {path: entry for path, entry in manifest_entries.items()}
    for root in spec.reuse_result_roots:
        for path in scan_result_json_paths(root):
            candidate_paths.setdefault(path, None)

    existence_rows: list[dict[str, object]] = []
    perturbation_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for result_json_path, manifest_entry in sorted(candidate_paths.items()):
        test_kind = str((manifest_entry or {}).get("test_kind", ""))
        if test_kind in {"", "existence"}:
            record, warnings = extract_existence_record(
                result_json_path,
                expected_sigma=float(existence_base_run_config.data.sigma),
                expected_dependent_xy=bool(existence_base_run_config.data.dependent_xy),
                expected_poly_degree=int(existence_base_run_config.data.poly_degree),
                manifest_entry=manifest_entry,
            )
            warning_rows.extend(warnings)
            if record is not None:
                existence_rows.append(record)
        if test_kind in {"", "perturbation"}:
            records, warnings = extract_perturbation_records(
                result_json_path,
                expected_sigma=float(perturbation_base_run_config.data.sigma),
                expected_dependent_xy=bool(perturbation_base_run_config.data.dependent_xy),
                expected_poly_degree=int(perturbation_base_run_config.data.poly_degree),
                manifest_entry=manifest_entry,
            )
            warning_rows.extend(warnings)
            perturbation_rows.extend(records)

    existence_summary_rows = summarize_existence_rows(existence_rows)
    perturbation_summary_rows = summarize_perturbation_rows(perturbation_rows)
    boxplot_rows = build_boxplot_rows(existence_rows, perturbation_rows)

    existence_path = analysis_dir / "existence_per_run_results.csv"
    existence_summary_path = analysis_dir / "existence_summary_by_kmax.csv"
    perturbation_path = analysis_dir / "perturbation_per_run_results.csv"
    perturbation_summary_path = analysis_dir / "perturbation_summary_by_kmax_delta.csv"
    boxplot_path = analysis_dir / "boxplot_values_by_kmax.csv"
    warnings_path = analysis_dir / "analysis_warnings.csv"

    write_csv(existence_path, existence_rows, fieldnames=EXISTENCE_FIELDS)
    write_csv(existence_summary_path, existence_summary_rows, fieldnames=EXISTENCE_SUMMARY_FIELDS)
    write_csv(perturbation_path, perturbation_rows, fieldnames=PERTURBATION_FIELDS)
    write_csv(perturbation_summary_path, perturbation_summary_rows, fieldnames=PERTURBATION_SUMMARY_FIELDS)
    write_csv(boxplot_path, boxplot_rows, fieldnames=BOXPLOT_FIELDS)
    write_csv(warnings_path, warning_rows, fieldnames=WARNING_FIELDS)

    existence_plot_path = save_pvalue_vs_kmax_plot(
        existence_rows,
        existence_summary_rows,
        analysis_dir / "existence_pvalue_vs_kmax.png",
        title="Fourier Existence p-value vs k_max",
    )
    perturbation_plot_path = save_grouped_pvalue_vs_kmax_plot(
        perturbation_rows,
        perturbation_summary_rows,
        analysis_dir / "perturbation_pvalue_vs_kmax.png",
        group_field="delta",
        title="Fourier Perturbation p-value vs k_max",
        group_label_prefix="delta=",
    )
    boxplot_paths = save_boxplots_by_kmax(boxplot_rows, analysis_dir)

    payload = {
        "experiment_name": spec.experiment_name,
        "analysis_dir": str(analysis_dir),
        "n_result_files_scanned": len(candidate_paths),
        "n_existence_rows_analyzed": len(existence_rows),
        "n_perturbation_rows_analyzed": len(perturbation_rows),
        "n_warnings": len(warning_rows),
        "existence_per_run_results_csv": str(existence_path),
        "existence_summary_by_kmax_csv": str(existence_summary_path),
        "perturbation_per_run_results_csv": str(perturbation_path),
        "perturbation_summary_by_kmax_delta_csv": str(perturbation_summary_path),
        "boxplot_values_by_kmax_csv": str(boxplot_path),
        "analysis_warnings_csv": str(warnings_path),
        "existence_pvalue_vs_kmax_plot": "" if existence_plot_path is None else str(existence_plot_path),
        "perturbation_pvalue_vs_kmax_plot": "" if perturbation_plot_path is None else str(perturbation_plot_path),
        "boxplot_paths": [str(path) for path in boxplot_paths],
    }
    summary_json_path = analysis_dir / "analysis_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["analysis_summary_json"] = str(summary_json_path)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the Fourier k_max existence and perturbation study.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = analyze_fourier_kmax_existence_perturbation_results(args.spec)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
