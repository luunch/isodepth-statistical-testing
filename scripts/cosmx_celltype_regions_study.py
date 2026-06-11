"""Sequential CosMx cell-type region existence-test experiment.

Runs every config in ``configs/cosmx_celltype_regions/`` (or rows from the
manifest), writes full per-region output folders after each run, updates the
aggregate summary continuously, and clears GPU/CPU memory between regions.

Resumable: skips regions whose ``{run_name}_result.json`` already exists unless
``--rerun-all`` or ``run.skip_finished: false`` in the spec.

Usage:
  mamba run -n isodepth_env python scripts/cosmx_celltype_regions_study.py
  mamba run -n isodepth_env python scripts/cosmx_celltype_regions_study.py \\
      --spec configs/experiments/cosmx_celltype_regions.json --limit 3
  mamba run -n isodepth_env python scripts/cosmx_celltype_regions_study.py --summarize
  sbatch run_cosmx_celltype_regions_study.sh
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import load_dataset
from experiments.configuration import build_run_config, save_standardized_outputs
from methods.permutation import run_permutation_method
from scripts.liver_lobule_sweep import load_spec

DEFAULT_SPEC = REPO / "configs/experiments/cosmx_celltype_regions.json"
GAUSSIAN_OUT_DIR = "results/cosmx_celltype_regions_gaussian"


def _gaussian_run_name(poisson_run_name: str) -> str:
    if poisson_run_name.endswith("_poisson"):
        return poisson_run_name[: -len("_poisson")] + "_gaussian"
    return f"{poisson_run_name}_gaussian"


def _cli_overrides_for_variant(variant: str | None, poisson_run_name: str) -> dict:
    if variant != "gaussian":
        return {}
    return {
        "data": {
            "normalize_total": True,
            "standardize_expression": True,
            "log1p": True,
        },
        "test": {"metric": "nll_gaussian_mse"},
        "output": {
            "out_dir": GAUSSIAN_OUT_DIR,
            "run_name": _gaussian_run_name(poisson_run_name),
        },
    }


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _result_json(out_dir: Path, run_name: str) -> Path:
    return out_dir / run_name / f"{run_name}_result.json"


def _is_done(out_dir: Path, run_name: str) -> bool:
    return _result_json(out_dir, run_name).exists()


def _load_queue(spec: dict) -> pd.DataFrame:
    manifest_path = REPO / spec["manifest"]
    config_dir = REPO / spec.get("config_dir", "configs/cosmx_celltype_regions")

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if "config" not in df.columns:
            raise ValueError(f"manifest missing 'config' column: {manifest_path}")
        df["config_path"] = df["config"].map(lambda p: REPO / p)
    else:
        configs = sorted(config_dir.glob("*.json"))
        if not configs:
            raise SystemExit(f"no configs in {config_dir} and no manifest at {manifest_path}")
        rows = []
        for cfg in configs:
            raw = json.loads(cfg.read_text())
            out = raw.get("output", {})
            rows.append({
                "region_name": cfg.stem,
                "run_name": out.get("run_name", cfg.stem),
                "config": str(cfg.relative_to(REPO)),
                "config_path": cfg,
            })
        df = pd.DataFrame(rows)

    sort_by = spec.get("run", {}).get("sort_by", "sample_then_size")
    if sort_by == "sample_then_size" and {"sample", "n_cells"}.issubset(df.columns):
        df = df.sort_values(["sample", "n_cells"], ascending=[True, False])
    elif "run_name" in df.columns:
        df = df.sort_values("run_name")

    limit = int(spec.get("run", {}).get("limit", 0))
    if limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def _write_progress(
    out_dir: Path,
    *,
    spec: dict,
    queue: pd.DataFrame,
    completed: list[dict],
    failed: list[dict],
    status: str,
) -> None:
    payload = {
        "experiment_name": spec.get("experiment_name", "cosmx_celltype_regions"),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_queued": int(len(queue)),
        "completed": len(completed),
        "failed": len(failed),
        "pending": int(len(queue) - len(completed) - len(failed)),
        "completed_runs": completed,
        "failed_runs": failed,
    }
    (out_dir / "experiment_progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def _refresh_summary(
    out_dir: Path,
    manifest: Path,
    alpha: float,
    *,
    write_plot: bool = False,
    run_name_suffix: str | None = None,
) -> None:
    from scripts.cosmx_celltype_regions_summary import write_summary

    df = write_summary(
        manifest, out_dir, alpha=alpha, write_plot=write_plot,
        run_name_suffix=run_name_suffix,
    )
    ok = df[df["status"] == "ok"]
    if len(ok):
        rate = float((ok["p_value"] < alpha).mean())
        print(f"  [summary] {len(ok)}/{len(df)} finished; sig rate (p<{alpha}) = {rate:.3f}")


def _run_one(cfg_path: Path, *, variant: str | None, poisson_run_name: str) -> dict:
    overrides = _cli_overrides_for_variant(variant, poisson_run_name)
    run_config = build_run_config(str(cfg_path), overrides)
    run_name = run_config.output.run_name
    out_root = Path(run_config.output.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"  [load] {cfg_path.name} (n_cells from manifest if listed)", flush=True)
    t0 = time.perf_counter()
    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    n_cells = int(dataset.S.shape[0])
    print(f"  [train] {run_name}: n={n_cells:,} cells", flush=True)

    result = run_permutation_method(dataset, run_config.test)
    payload, result_path = save_standardized_outputs(dataset, result, run_config)

    elapsed = time.perf_counter() - t0
    row = {
        "run_name": run_name,
        "config": str(cfg_path.relative_to(REPO)),
        "n_cells": n_cells,
        "p_value": payload.get("p_value"),
        "stat_true": payload.get("stat_true"),
        "elapsed_s": round(elapsed, 1),
        "result_dir": str(result_path.parent),
    }
    print(f"  [done] {run_name}: p={row['p_value']:.4f} ({elapsed:.1f}s) -> {result_path.parent.name}/",
          flush=True)

    del dataset, result, payload, run_config
    _release_memory()
    return row


def main(
    spec_path: str | Path = DEFAULT_SPEC,
    *,
    rerun_all: bool = False,
    limit: int | None = None,
    variant: str | None = None,
    out_dir_override: str | None = None,
) -> None:
    spec = load_spec(spec_path)
    out_dir = REPO / (out_dir_override or spec["output"]["out_dir"])
    if variant == "gaussian":
        out_dir = REPO / GAUSSIAN_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / spec["manifest"]

    skip_finished = bool(spec.get("run", {}).get("skip_finished", True)) and not rerun_all
    save_outputs = bool(spec.get("output", {}).get("save_per_region_outputs", True))
    write_summary_each = bool(spec.get("output", {}).get("write_summary_after_each", True))
    alpha = float(spec.get("summary", {}).get("alpha", 0.05))
    run_name_suffix = "_gaussian" if variant == "gaussian" else None

    if limit is not None:
        spec = dict(spec)
        spec["run"] = dict(spec.get("run", {}))
        spec["run"]["limit"] = int(limit)
    if variant:
        spec = dict(spec)
        spec["variant"] = variant
        spec["output"] = dict(spec.get("output", {}))
        spec["output"]["out_dir"] = str(out_dir.relative_to(REPO))

    queue = _load_queue(spec)
    exp_name = spec.get("experiment_name", "cosmx_celltype_regions")
    if variant:
        exp_name = f"{exp_name}_{variant}"
    print(f"[experiment] {exp_name} — {len(queue)} regions queued")
    print(f"[output]     {out_dir}")
    if variant == "gaussian":
        print("[variant]    gaussian (normalize_total+log1p, nll_gaussian_mse)")

    (out_dir / "experiment_spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    completed: list[dict] = []
    failed: list[dict] = []

    for i, row in queue.iterrows():
        cfg_path = Path(row["config_path"]) if "config_path" in row else REPO / row["config"]
        poisson_run_name = str(row.get("run_name", cfg_path.stem))
        run_name = (
            _gaussian_run_name(poisson_run_name) if variant == "gaussian" else poisson_run_name
        )

        if skip_finished and _is_done(out_dir, run_name):
            print(f"[{i + 1}/{len(queue)}] skip (done): {run_name}", flush=True)
            completed.append({"run_name": run_name, "status": "skipped_existing"})
            continue

        print(f"[{i + 1}/{len(queue)}] {run_name}", flush=True)
        _write_progress(out_dir, spec=spec, queue=queue, completed=completed,
                        failed=failed, status="running")

        try:
            if not save_outputs:
                raise RuntimeError("save_per_region_outputs=false is not supported for this study")
            result_row = _run_one(cfg_path, variant=variant, poisson_run_name=poisson_run_name)
            completed.append(result_row)
        except Exception as exc:
            print(f"  [FAIL] {run_name}: {exc}", flush=True)
            failed.append({"run_name": run_name, "error": str(exc)})
            _release_memory()

        if write_summary_each:
            _refresh_summary(
                out_dir, manifest_path, alpha, write_plot=False,
                run_name_suffix=run_name_suffix,
            )
        _write_progress(out_dir, spec=spec, queue=queue, completed=completed,
                        failed=failed, status="running")

    from scripts.cosmx_celltype_regions_summary import write_summary
    write_summary(
        manifest_path, out_dir, alpha=alpha, write_plot=True,
        run_name_suffix=run_name_suffix,
    )

    status = "failed" if failed else "completed"
    _write_progress(out_dir, spec=spec, queue=queue, completed=completed,
                    failed=failed, status=status)
    print(f"\n[experiment done] completed={len(completed)} failed={len(failed)} -> {out_dir}")
    if failed:
        raise SystemExit(1)


def summarize_only(spec_path: str | Path = DEFAULT_SPEC, *, variant: str | None = None) -> None:
    spec = load_spec(spec_path)
    out_dir = REPO / (GAUSSIAN_OUT_DIR if variant == "gaussian" else spec["output"]["out_dir"])
    alpha = float(spec.get("summary", {}).get("alpha", 0.05))
    from scripts.cosmx_celltype_regions_summary import write_summary
    write_summary(
        REPO / spec["manifest"], out_dir, alpha=alpha, write_plot=True,
        run_name_suffix="_gaussian" if variant == "gaussian" else None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--limit", type=int, default=None, help="Run at most N regions (overrides spec).")
    ap.add_argument("--rerun-all", action="store_true", help="Re-run even if result JSON exists.")
    ap.add_argument("--summarize", action="store_true", help="Only rebuild summary.csv/png from existing runs.")
    ap.add_argument("--gaussian", action="store_true",
                    help="Gaussian path (normalize_total+log1p, nll_gaussian_mse) -> results/cosmx_celltype_regions_gaussian/")
    args = ap.parse_args()
    variant = "gaussian" if args.gaussian else None
    if args.summarize:
        summarize_only(args.spec, variant=variant)
    else:
        main(args.spec, rerun_all=args.rerun_all, limit=args.limit, variant=variant)
