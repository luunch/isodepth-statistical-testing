from __future__ import annotations

import argparse
import json

from data import load_dataset
from experiments.configuration import build_run_config, save_standardized_outputs
from methods.permutation import run_permutation_method


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configured isodepth statistical test from a config-defined dataset."
    )

    parser.add_argument("--config", default=None, help="Path to JSON config file")

    parser.add_argument("--data-source", dest="data_source", default=argparse.SUPPRESS)
    parser.add_argument("--h5ad", default=argparse.SUPPRESS, help="Path to input .h5ad file")
    parser.add_argument("--spatial-key", default=argparse.SUPPRESS)
    parser.add_argument("--obs-x-col", default=argparse.SUPPRESS)
    parser.add_argument("--obs-y-col", default=argparse.SUPPRESS)
    parser.add_argument("--layer", default=argparse.SUPPRESS)
    parser.add_argument("--use-raw", dest="use_raw", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-use-raw", dest="use_raw", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--min-cells-per-gene", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--standardize", dest="standardize", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--max-cells", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode", default=argparse.SUPPRESS)
    parser.add_argument("--n-cells", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-genes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sigma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--method", default=argparse.SUPPRESS)
    parser.add_argument("--metric", default=argparse.SUPPRESS)
    parser.add_argument("--n-perms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-nulls", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-experts", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--delta", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--perturb-target", default=argparse.SUPPRESS)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=argparse.SUPPRESS)

    parser.add_argument("--out-dir", dest="out_dir", default=argparse.SUPPRESS, help="Output directory")
    parser.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS, help="Run name used in result naming")
    parser.add_argument("--save-preds", dest="save_preds", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-save-preds", dest="save_preds", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument(
        "--save-perm-stats",
        dest="save_perm_stats",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-save-perm-stats",
        dest="save_perm_stats",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    return parser


def _build_cli_overrides(args: argparse.Namespace) -> dict:
    data_overrides = {}
    test_overrides = {}
    output_overrides = {}

    for arg_name, config_key in {
        "data_source": "source",
        "h5ad": "h5ad",
        "spatial_key": "spatial_key",
        "obs_x_col": "obs_x_col",
        "obs_y_col": "obs_y_col",
        "layer": "layer",
        "use_raw": "use_raw",
        "min_cells_per_gene": "min_cells_per_gene",
        "standardize": "standardize",
        "max_cells": "max_cells",
        "mode": "mode",
        "n_cells": "n_cells",
        "n_genes": "n_genes",
        "sigma": "sigma",
    }.items():
        if hasattr(args, arg_name):
            data_overrides[config_key] = getattr(args, arg_name)

    for arg_name, config_key in {
        "method": "method",
        "metric": "metric",
        "n_perms": "n_perms",
        "n_nulls": "n_nulls",
        "epochs": "epochs",
        "lr": "lr",
        "patience": "patience",
        "device": "device",
        "batch_size": "batch_size",
        "n_experts": "n_experts",
        "delta": "delta",
        "perturb_target": "perturb_target",
        "verbose": "verbose",
    }.items():
        if hasattr(args, arg_name):
            test_overrides[config_key] = getattr(args, arg_name)

    for arg_name, config_key in {
        "out_dir": "out_dir",
        "run_name": "run_name",
        "save_preds": "save_preds",
        "save_perm_stats": "save_perm_stats",
    }.items():
        if hasattr(args, arg_name):
            output_overrides[config_key] = getattr(args, arg_name)

    if hasattr(args, "seed"):
        data_overrides["seed"] = args.seed
        test_overrides["seed"] = args.seed

    overrides = {}
    if data_overrides:
        overrides["data"] = data_overrides
    if test_overrides:
        overrides["test"] = test_overrides
    if output_overrides:
        overrides["output"] = output_overrides
    return overrides


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cli_overrides = _build_cli_overrides(args)
    run_config = build_run_config(args.config, cli_overrides)

    dataset = load_dataset(run_config.data)
    result = run_permutation_method(dataset, run_config.test)
    payload, result_path = save_standardized_outputs(dataset, result, run_config)

    print(json.dumps(payload, indent=2))
    print(f"Saved outputs to: {result_path.parent}")


if __name__ == "__main__":
    main()
