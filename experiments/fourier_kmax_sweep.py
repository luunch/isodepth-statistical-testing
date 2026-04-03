from __future__ import annotations

import argparse
import json

from data import load_dataset
from experiments.configuration import build_run_config, save_standardized_outputs
from experiments.fourier_kmax import (
    FourierKmaxStudySpec,
    analysis_dir_for_spec,
    build_fourier_kmax_run_config,
    expand_fourier_kmax_conditions,
    load_fourier_kmax_spec,
    manifest_path_for_spec,
)


def run_fourier_kmax_sweep(
    spec: FourierKmaxStudySpec,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, object]:
    base_run_config = build_run_config(str(spec.base_config), {})
    conditions = expand_fourier_kmax_conditions(spec)
    if max_runs is not None:
        conditions = conditions[:max_runs]

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "output_root": str(spec.output_root),
        "analysis_dir": str(analysis_dir_for_spec(spec)),
        "k_min": int(spec.k_min),
        "runs": [],
    }

    if dry_run:
        manifest_payload["planned_run_count"] = len(conditions)
        manifest_payload["planned_runs_preview"] = [
            {
                "run_name": condition.run_name,
                "seed": int(condition.seed),
                "k_min": int(condition.k_min),
                "k_max": int(condition.k_max),
            }
            for condition in conditions[:10]
        ]
        return manifest_payload

    spec.output_root.mkdir(parents=True, exist_ok=True)
    (spec.output_root / "runs").mkdir(parents=True, exist_ok=True)
    analysis_dir_for_spec(spec).mkdir(parents=True, exist_ok=True)

    from methods.permutation import run_permutation_method

    for index, condition in enumerate(conditions, start=1):
        print(f"[{index}/{len(conditions)}] {condition.run_name}")
        run_config = build_fourier_kmax_run_config(base_run_config, spec, condition)
        dataset = load_dataset(run_config.data)
        result = run_permutation_method(dataset, run_config.test)
        _, result_path = save_standardized_outputs(dataset, result, run_config)
        manifest_payload["runs"].append(
            {
                "run_name": condition.run_name,
                "seed": int(condition.seed),
                "k_min": int(condition.k_min),
                "k_max": int(condition.k_max),
                "result_json_path": str(result_path.resolve()),
            }
        )

    manifest_path = manifest_path_for_spec(spec)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    print(f"Saved manifest to: {manifest_path}")
    return manifest_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Fourier k_max existence sweep experiment.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without executing them")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on the number of planned runs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec = load_fourier_kmax_spec(args.spec)
    payload = run_fourier_kmax_sweep(spec, dry_run=args.dry_run, max_runs=args.max_runs)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
