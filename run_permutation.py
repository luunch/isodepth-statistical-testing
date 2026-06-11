from __future__ import annotations

import argparse

from data import load_dataset
from experiments.configuration import build_run_config, save_standardized_outputs
from experiments.recursive_svg import run_recursive_svg
from methods.permutation import run_permutation_method


def _parse_csv_floats(value: str) -> list[float]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    try:
        return [float(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats") from exc


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
    parser.add_argument(
        "--top-var-genes",
        dest="top_var_genes",
        type=int,
        default=argparse.SUPPRESS,
        help="Keep only the top-N scanpy highly variable genes (0 = use all genes; h5ad only)",
    )
    parser.add_argument(
        "--normalize-total",
        dest="normalize_total",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Per-cell CPM normalization (target sum 1e6) before log1p (removes depth confound; h5ad only).",
    )
    parser.add_argument(
        "--no-normalize-total",
        dest="normalize_total",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--log1p", dest="log1p", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-log1p", dest="log1p", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument(
        "--standardize-expression",
        dest="standardize_expression",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-standardize-expression",
        dest="standardize_expression",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--standardize-coordinates",
        dest="standardize_coordinates",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-standardize-coordinates",
        dest="standardize_coordinates",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--q", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-cells", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--mode", default=argparse.SUPPRESS)
    parser.add_argument("--n-cells", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n-genes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sigma", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--expression-distribution",
        dest="expression_distribution",
        default=argparse.SUPPRESS,
        choices=("gaussian", "poisson"),
        help="Synthetic expression sampling distribution (default: gaussian).",
    )
    parser.add_argument("--mean-count", dest="mean_count", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--k-min", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--k-max", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--method", default=argparse.SUPPRESS)
    parser.add_argument("--metric", default=argparse.SUPPRESS)
    parser.add_argument("--n-perms", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--n-folds",
        dest="n_folds",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of cross-validation folds for cross_validation method.",
    )
    parser.add_argument("--n-reruns", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--n-nulls", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--decoder", default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sgd-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--sgd-cosine-lr-decay",
        dest="sgd_cosine_lr_decay",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-sgd-cosine-lr-decay",
        dest="sgd_cosine_lr_decay",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--sgd-cosine-eta-min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--sgd-cosine-t-max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--delta", type=_parse_csv_floats, default=argparse.SUPPRESS)
    parser.add_argument("--perturb-target", default=argparse.SUPPRESS)
    parser.add_argument("--subset-fractions", type=_parse_csv_floats, default=argparse.SUPPRESS)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--quiet", dest="verbose", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable recursive SVG gradient peeling (requires linear or quadratic decoder).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-gradients",
        dest="max_gradients",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum number of spatial gradients to discover in recursive mode.",
    )

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
    parser.add_argument(
        "--covariate-type",
        dest="covariate_type",
        default=argparse.SUPPRESS,
        help=(
            "Fixed bottleneck covariate type.  Use 'midline' for d(x,y)=|x-median(x)| "
            "(computed from coordinates), or any obs column name to read per-cell latent "
            "values from adata.obs[<key>] in the h5ad file."
        ),
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
        "top_var_genes": "top_var_genes",
        "normalize_total": "normalize_total",
        "log1p": "log1p",
        "standardize_expression": "standardize_expression",
        "standardize_coordinates": "standardize_coordinates",
        "q": "q",
        "max_cells": "max_cells",
        "mode": "mode",
        "n_cells": "n_cells",
        "n_genes": "n_genes",
        "sigma": "sigma",
        "expression_distribution": "expression_distribution",
        "mean_count": "mean_count",
        "k": "k",
        "k_min": "k_min",
        "k_max": "k_max",
    }.items():
        if hasattr(args, arg_name):
            data_overrides[config_key] = getattr(args, arg_name)

    for arg_name, config_key in {
        "method": "method",
        "metric": "metric",
        "n_perms": "n_perms",
        "n_folds": "n_folds",
        "n_reruns": "n_reruns",
        "alpha": "alpha",
        "n_nulls": "n_nulls",
        "epochs": "epochs",
        "lr": "lr",
        "patience": "patience",
        "device": "device",
        "decoder": "decoder",
        "batch_size": "batch_size",
        "sgd_batch_size": "sgd_batch_size",
        "sgd_cosine_lr_decay": "sgd_cosine_lr_decay",
        "sgd_cosine_eta_min": "sgd_cosine_eta_min",
        "sgd_cosine_t_max_steps": "sgd_cosine_t_max_steps",
        "delta": "delta",
        "perturb_target": "perturb_target",
        "subset_fractions": "subset_fractions",
        "verbose": "verbose",
        "recursive": "recursive",
        "max_gradients": "max_gradients",
    }.items():
        if hasattr(args, arg_name):
            test_overrides[config_key] = getattr(args, arg_name)

    if hasattr(args, "covariate_type"):
        test_overrides["covariate"] = {"type": args.covariate_type}

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

    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)

    if run_config.test.recursive:
        payload, result_path = run_recursive_svg(dataset, run_config)
        print(f"Saved recursive outputs to: {result_path.parent}")
    else:
        result = run_permutation_method(dataset, run_config.test)
        payload, result_path = save_standardized_outputs(dataset, result, run_config)
        print(f"Saved outputs to: {result_path.parent}")


if __name__ == "__main__":
    main()
