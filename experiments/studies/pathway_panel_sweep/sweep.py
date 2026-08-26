from __future__ import annotations

import argparse
import json

from data import load_dataset
from experiments.configuration import build_manifest_config_snapshot, build_run_config, save_standardized_outputs
from experiments.studies.pathway_panel_sweep.lib import (
    build_pathway_run_config,
    expand_pathway_conditions,
    hvg3000_result_json_for_spec,
    inject_reference_isodepth_covariate,
    load_gmt_gene_sets,
    load_pathway_panel_sweep_spec,
    load_reference_isodepth,
    manifest_path_for_spec,
    run_hvg3000_reference,
)


def run_pathway_panel_sweep(
    spec_path: str,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
    hvg3000_only: bool = False,
    skip_hvg3000: bool = False,
) -> dict[str, object]:
    spec = load_pathway_panel_sweep_spec(spec_path)
    base_run_config = build_run_config(str(spec.base_config), {})
    gene_sets = load_gmt_gene_sets(spec.gmt_path)
    conditions = expand_pathway_conditions(spec, gene_sets=gene_sets)
    if max_runs is not None:
        conditions = conditions[:max_runs]

    hvg3000_result_path = None
    if spec.covariate_from_hvg3000 or spec.hvg3000_config is not None:
        hvg3000_result_path = str(hvg3000_result_json_for_spec(spec))

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "base_config_path": str(spec.base_config),
        "gmt_path": str(spec.gmt_path),
        "output_root": str(spec.output_root),
        "n_perms": int(spec.n_perms),
        "n_reruns": int(spec.n_reruns),
        "alpha": float(spec.alpha),
        "min_requested_genes": int(spec.min_requested_genes),
        "hvg3000_config": str(spec.hvg3000_config) if spec.hvg3000_config else None,
        "hvg3000_result_json": hvg3000_result_path,
        "covariate_from_hvg3000": bool(spec.covariate_from_hvg3000),
        "covariate_obs_key": spec.covariate_obs_key,
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

    reference_iso = None
    reference_S = None
    if spec.covariate_from_hvg3000 or spec.hvg3000_config is not None:
        if skip_hvg3000:
            result_path = hvg3000_result_json_for_spec(spec)
            if not result_path.exists():
                raise FileNotFoundError(
                    f"--skip-hvg3000 was set but HVG3000 result is missing: {result_path}"
                )
        else:
            result_path = run_hvg3000_reference(spec)
        manifest_payload["hvg3000_result_json"] = str(result_path)
        if hvg3000_only:
            return manifest_payload
        if spec.covariate_from_hvg3000:
            reference_iso, reference_S = load_reference_isodepth(result_path)
            print(
                f"Using HVG3000 isodepth covariate from {result_path} "
                f"(n_cells={reference_iso.shape[0]})",
                flush=True,
            )

    for index, condition in enumerate(conditions, start=1):
        print(
            f"[{index}/{len(conditions)}] {condition.pathway_name} "
            f"(genes={len(condition.gene_list)})",
            flush=True,
        )
        run_config = build_pathway_run_config(base_run_config, spec, condition)
        dataset = load_dataset(run_config.data)
        if spec.covariate_from_hvg3000:
            if reference_iso is None or reference_S is None:
                raise RuntimeError("HVG3000 isodepth was not loaded before pathway runs")
            inject_reference_isodepth_covariate(
                dataset,
                reference_iso,
                reference_S,
                obs_key=spec.covariate_obs_key,
            )
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
    parser.add_argument(
        "--hvg3000-only",
        action="store_true",
        help="Train/save the 3000-HVG isodepth and exit before the pathway sweep",
    )
    parser.add_argument(
        "--skip-hvg3000",
        action="store_true",
        help="Reuse an existing HVG3000 result instead of training it",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = run_pathway_panel_sweep(
        args.spec,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
        hvg3000_only=args.hvg3000_only,
        skip_hvg3000=args.skip_hvg3000,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
