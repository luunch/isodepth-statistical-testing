from __future__ import annotations

import argparse
import json

from data import load_dataset
from experiments.configuration import build_manifest_config_snapshot, build_run_config, save_standardized_outputs
from experiments.studies.random_gene_panel_null.lib import (
    build_condition_run_config,
    build_eligible_gene_universe,
    eligible_universe_path_for_spec,
    expand_conditions,
    load_random_gene_panel_null_spec,
    manifest_path_for_spec,
)


def run_random_gene_panel_null_sweep(
    spec_path: str,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, object]:
    spec = load_random_gene_panel_null_spec(spec_path)
    base_run_config = build_run_config(str(spec.base_config), {})
    target_gene_list = list(base_run_config.data.gene_list or [])
    eligible_genes = build_eligible_gene_universe(spec)
    conditions = expand_conditions(
        spec,
        eligible_genes=eligible_genes,
        target_gene_list=target_gene_list,
    )
    if max_runs is not None:
        conditions = conditions[:max_runs]

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "output_root": str(spec.output_root),
        "n_panels": int(spec.n_panels),
        "panel_size": int(spec.panel_size),
        "panel_seeds": [int(seed) for seed in spec.panel_seeds],
        "n_perms": int(spec.n_perms),
        "n_reruns": int(spec.n_reruns),
        "universe_min_cells_per_gene": int(spec.universe_min_cells_per_gene),
        "include_target_run": bool(spec.include_target_run),
        "eligible_gene_universe_count": len(eligible_genes),
        "eligible_gene_universe_path": str(eligible_universe_path_for_spec(spec)),
        "config_snapshot": build_manifest_config_snapshot(
            spec_path,
            {"base_config": spec.base_config},
        ),
        "runs": [],
    }

    spec.output_root.mkdir(parents=True, exist_ok=True)
    with open(eligible_universe_path_for_spec(spec), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "universe_min_cells_per_gene": int(spec.universe_min_cells_per_gene),
                "eligible_gene_count": len(eligible_genes),
                "eligible_genes": eligible_genes,
            },
            handle,
            indent=2,
        )

    if dry_run:
        manifest_payload["planned_run_count"] = len(conditions)
        manifest_payload["planned_runs_preview"] = [
            {
                "condition_type": condition.condition_type,
                "panel_index": int(condition.panel_index),
                "panel_seed": int(condition.panel_seed),
                "run_name": condition.run_name,
                "gene_list_requested_count": len(condition.gene_list),
            }
            for condition in conditions[:10]
        ]
        return manifest_payload

    (spec.output_root / "runs").mkdir(parents=True, exist_ok=True)

    from methods.permutation import run_permutation_method

    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index}/{len(conditions)}] {condition.run_name} "
            f"({condition.condition_type}, genes={len(condition.gene_list)})",
            flush=True,
        )
        run_config = build_condition_run_config(base_run_config, spec, condition)
        dataset = load_dataset(run_config.data)
        result = run_permutation_method(dataset, run_config.test)
        _, result_path = save_standardized_outputs(dataset, result, run_config)
        manifest_payload["runs"].append(
            {
                "condition_type": condition.condition_type,
                "panel_index": int(condition.panel_index),
                "panel_seed": int(condition.panel_seed),
                "run_name": condition.run_name,
                "gene_list_requested_count": len(condition.gene_list),
                "result_json_path": str(result_path.resolve()),
            }
        )

    manifest_path = manifest_path_for_spec(spec)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    print(f"Saved manifest to: {manifest_path}", flush=True)
    return manifest_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a random gene-panel null specificity sweep for isodepth existence tests.",
    )
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of runs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = run_random_gene_panel_null_sweep(
        args.spec,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
