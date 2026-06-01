"""Executable experiment: how many reruns does an isodepth fit actually need?

A ``parallel_permutation`` fit trains ``test.n_reruns`` independent random
initializations of the network and keeps the one with the lowest training loss.
This experiment measures the value of that best-of-N selection directly:

1. Train a large number of reruns (default 100) *concurrently on the same,
   unpermuted dataset* -- no coordinate permutations are used (``n_perms = 0``), so
   every parallel slot is an independent random init fit to the true layout.  The
   existing parallel architecture trains all reruns in one batched pass.
2. Take the per-rerun final training losses and, for each ``k`` in ``1..R``,
   randomly select ``k`` reruns ``n_subsamples`` times (default 100), take the min
   loss of each subset, and average -> the expected best-of-``k`` loss.
3. Output a single graph of expected min loss vs ``k``.

The input is a config file; the experiment uses its data/optimization settings but
**overrides** ``n_reruns`` (this is the variable under study), forces ``n_perms = 0``
(no extra permutations are needed), and forces ``patience = 0`` (no early stopping --
every rerun trains for the full ``epochs``).

Usage
-----
    python -m experiments.rerun_convergence --config configs/mouse_hippocampus_existence.json
    python -m experiments.rerun_convergence --config configs/radial.json --n-reruns 100 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.rerun_convergence import (
    expected_min_loss_curve,
    render_expected_min_loss_figure,
)
from data import load_dataset
from experiments.configuration import build_run_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a parallel_permutation JSON config")
    parser.add_argument("--n-reruns", type=int, default=100, help="Number R of reruns to train concurrently")
    parser.add_argument("--n-subsamples", type=int, default=100, help="Random size-k subsets drawn per k")
    parser.add_argument("--device", default=None, help="Override test.device (e.g. cuda, cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override test.epochs")
    parser.add_argument("--max-cells", type=int, default=None, help="Override data.max_cells (subsample cells)")
    parser.add_argument("--q", type=int, default=None, help="Override data.q")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--analysis-seed", type=int, default=0, help="Seed for the subsampling RNG")
    parser.add_argument("--out-dir", default=None, help="Output directory (default results/<run>_rerun_convergence)")
    parser.add_argument("--run-name", default=None, help="Override output.run_name")
    return parser


def _train_rerun_losses(args: argparse.Namespace) -> dict:
    """Train ``R`` reruns on the unpermuted dataset; return their training losses."""
    from methods.trainers import (
        get_training_metadata,
        resolve_device,
        run_with_cuda_oom_retry,
        train_parallel_isodepth_model,
    )

    data_overrides: dict = {}
    output_overrides: dict = {}
    if args.max_cells is not None:
        data_overrides["max_cells"] = args.max_cells
    if args.q is not None:
        data_overrides["q"] = args.q
    if args.seed is not None:
        data_overrides["seed"] = args.seed
    if args.run_name is not None:
        output_overrides["run_name"] = args.run_name

    overrides: dict = {}
    if data_overrides:
        overrides["data"] = data_overrides
    if output_overrides:
        overrides["output"] = output_overrides

    run_config = build_run_config(args.config, overrides)

    # This experiment owns n_reruns / n_perms / patience; everything else (epochs,
    # lr, decoder, q, device, ...) comes from the config.  No permutations, no early
    # stopping, learned encoder (covariate=None).
    train_config = replace(
        run_config.test,
        n_reruns=int(args.n_reruns),
        n_perms=0,
        patience=0,
        covariate=None,
    )
    if args.device is not None:
        train_config = replace(train_config, device=args.device)
    if args.epochs is not None:
        train_config = replace(train_config, epochs=int(args.epochs))
    if args.seed is not None:
        train_config = replace(train_config, seed=int(args.seed))

    dataset = load_dataset(run_config.data, covariate=None)
    device = resolve_device(train_config.device)
    print(
        f"device: {device} | n_cells={dataset.n_cells} n_genes={dataset.n_genes} "
        f"| training {train_config.n_reruns} reruns on the unpermuted dataset "
        f"(n_perms=0, patience=0, epochs={train_config.epochs})",
        flush=True,
    )

    def _train(resolved_device):
        return train_parallel_isodepth_model(
            dataset.S,
            dataset.A,
            train_config,
            device=resolved_device,
            model_label=f"{train_config.n_reruns} concurrent reruns (true layout)",
        )

    model, _outputs, _s_batched = run_with_cuda_oom_retry(
        _train, device, label="rerun_convergence reruns"
    )
    metadata = get_training_metadata(model)
    # n_perms=0 -> one model, so train_loss_per_rerun has shape (1, R); row 0 is the
    # per-rerun loss vector for the (only) true-layout model.
    rerun_losses = np.asarray(metadata["train_loss_per_rerun"], dtype=np.float64)[0]
    return {
        "rerun_losses": rerun_losses,
        "run_name": run_config.output.run_name,
        "metric": run_config.test.metric,
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    captured = _train_rerun_losses(args)

    run_name = captured["run_name"]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"{run_name}_rerun_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = expected_min_loss_curve(
        captured["rerun_losses"],
        n_subsamples=args.n_subsamples,
        seed=args.analysis_seed,
    )

    fig_path = render_expected_min_loss_figure(
        curve,
        out_dir / f"{run_name}_rerun_convergence.png",
        title=f"Expected best-of-k rerun loss: {run_name}",
        n_subsamples=args.n_subsamples,
    )
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
