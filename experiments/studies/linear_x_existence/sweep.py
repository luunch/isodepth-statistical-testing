from __future__ import annotations

import json

from data import load_dataset
from experiments.configuration import build_manifest_config_snapshot, build_run_config, save_standardized_outputs
from experiments.core.study_spec import sweep_arg_parser
from experiments.studies.linear_x_existence.lib import (
    LinearXExistenceStudySpec,
    analysis_dir_for_spec,
    build_linear_x_existence_run_config,
    expand_linear_x_existence_conditions,
    load_linear_x_existence_spec,
    manifest_path_for_spec,
)


def run_linear_x_existence_sweep(
    spec: LinearXExistenceStudySpec,
    *,
    spec_path: str | None = None,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, object]:
    base_run_config = build_run_config(str(spec.base_config), {})
    conditions = expand_linear_x_existence_conditions(spec)
    if max_runs is not None:
        conditions = conditions[:max_runs]

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "output_root": str(spec.output_root),
        "analysis_dir": str(analysis_dir_for_spec(spec)),
        "signal_fractions": [float(value) for value in spec.signal_fractions],
        "seeds": [int(value) for value in spec.seeds],
        "runs": [],
    }
    if spec_path is not None:
        manifest_payload["config_snapshot"] = build_manifest_config_snapshot(
            spec_path, {"base_config": spec.base_config},
        )

    if dry_run:
        manifest_payload["planned_run_count"] = len(conditions)
        manifest_payload["planned_runs_preview"] = [
            {
                "run_name": condition.run_name,
                "seed": int(condition.seed),
                "signal_fraction": float(condition.signal_fraction),
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
        run_config = build_linear_x_existence_run_config(base_run_config, spec, condition)
        dataset = load_dataset(run_config.data)
        result = run_permutation_method(dataset, run_config.test)
        _, result_path = save_standardized_outputs(dataset, result, run_config)
        manifest_payload["runs"].append(
            {
                "run_name": condition.run_name,
                "seed": int(condition.seed),
                "signal_fraction": float(condition.signal_fraction),
                "result_json_path": str(result_path.resolve()),
            }
        )

    manifest_path = manifest_path_for_spec(spec)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    print(f"Saved manifest to: {manifest_path}")
    return manifest_payload


def _build_arg_parser():
    return sweep_arg_parser("Run a linear-x existence sweep over signal fraction and seed.")


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec = load_linear_x_existence_spec(args.spec)
    payload = run_linear_x_existence_sweep(
        spec, spec_path=args.spec, dry_run=args.dry_run, max_runs=args.max_runs,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
