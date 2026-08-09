from __future__ import annotations

import argparse
import json

from data import load_dataset
from experiments.configuration import build_manifest_config_snapshot, build_run_config, save_standardized_outputs
from experiments.studies.pathway_panel_sweep.lib import (
    build_pathway_run_config,
    expand_pathway_conditions,
    load_gmt_gene_sets,
    load_pathway_panel_sweep_spec,
    manifest_path_for_spec,
)


def run_pathway_panel_sweep(
    spec_path: str,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    base_run_config = build_run_config(str(spec.base_config), {})
    gene_sets = load_gmt_gene_sets(spec.gmt_path)
    conditions = expand_pathway_conditions(spec, gene_sets=gene_sets)
    if max_runs is not None:
        conditions = conditions[:max_runs]

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "gmt_path": str(spec.gmt_path),
        "output_root": str(spec.output_root),
        "n_perms": int(spec.n_perms),
        "n_reruns": int(spec.n_reruns),
        "alpha": float(spec.alpha),
        "min_requested_genes": int(spec.min_requested_genes),
        "pathway_count_in_gmt": len(gene_sets),
        "pathway_count_planned": len(conditions),
        "config_snapshot": build_manifest_config_snapshot(
            spec_path,
            {"base_config": spec.base_config},
        ),
        "runs": [],
    }

    spec.output_root.mkdir(parents=True, exist_ok=True)

    if dry_run:
        manifest_payload["planned_run_count"] = len(conditions)
        manifest_payload["planned_runs_preview"] = [
            {
                "pathway_index": int(condition.pathway_index),
                "pathway_name": condition.pathway_name,
                "run_name": condition.run_name,
                "gene_list_requested_count": len(condition.gene_list),
            }
            for condition in conditions[:15]
        ]
        return manifest_payload

    (spec.output_root / "runs").mkdir(parents=True, exist_ok=True)

    from methods.permutation import run_permutation_method

    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index}/{len(conditions)}] {condition.pathway_name} "
            f"(genes={len(condition.gene_list)})",
            flush=True,
        )
        run_config = build_pathway_run_config(base_run_config, spec, condition)
        dataset = load_dataset(run_config.data)
        result = run_permutation_method(dataset, run_config.test)
        _, result_path = save_standardized_outputs(dataset, result, run_config)
        manifest_payload["runs"].append(
            {
                "pathway_index": int(condition.pathway_index),
                "pathway_name": condition.pathway_name,
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
        description="Run isodepth existence tests for every pathway in a GMT file.",
    )
    parser.add_argument("--spec", required=True, help="Path to the experiment spec JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of runs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = run_pathway_panel_sweep(
        args.spec,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
