"""Run all square grid-aligned Moran-null block-permutation configs sequentially.

Generates histograms via ``test.moran: true`` alongside the usual block-permutation
artifacts (metric distribution, isodepth triptych, block overlay, etc.).

Usage:
  python scripts/generate_moran_null_square_configs.py
  python scripts/moran_null_square_study.py
  python scripts/moran_null_square_study.py --spec configs/experiments/moran_null_square_study.json
  python scripts/moran_null_square_study.py --limit 1
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data import load_dataset
from experiments.configuration import build_run_config, save_standardized_outputs
from methods.permutation import run_permutation_method

DEFAULT_SPEC = REPO / "configs" / "experiments" / "moran_null_square_study.json"


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _result_json(out_dir: Path, run_name: str) -> Path:
    return out_dir / run_name / f"{run_name}_result.json"


def _is_done(out_dir: Path, run_name: str) -> bool:
    return _result_json(out_dir, run_name).exists()


def _load_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_paths(spec: dict) -> list[Path]:
    if spec.get("configs"):
        return [REPO / p for p in spec["configs"]]
    config_dir = REPO / spec.get("config_dir", "configs/synthetic/moran_null_square")
    return sorted(config_dir.glob("*.json"))


def run_one(config_path: Path) -> dict:
    run_config = build_run_config(str(config_path), {})
    out_dir = REPO / run_config.output.out_dir
    run_name = run_config.output.run_name

    if _is_done(out_dir, run_name):
        return {"run_name": run_name, "status": "skipped", "config": str(config_path)}

    t0 = time.time()
    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    result = run_permutation_method(dataset, run_config.test)
    save_standardized_outputs(dataset, result, run_config)
    elapsed = time.time() - t0
    _release_memory()

    moran_plot = out_dir / run_name / f"{run_name}_moran_distribution.png"
    return {
        "run_name": run_name,
        "status": "ok",
        "config": str(config_path),
        "elapsed_sec": round(elapsed, 1),
        "p_value": result.p_value,
        "moran_true_mean": result.artifacts.get("moran_true_mean"),
        "moran_p_value": result.artifacts.get("moran_p_value"),
        "moran_plot": str(moran_plot) if moran_plot.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerun-all", action="store_true")
    args = parser.parse_args()

    spec = _load_spec(Path(args.spec))
    output_root = REPO / spec.get("output_root", "results/moran_null_square")
    output_root.mkdir(parents=True, exist_ok=True)
    skip_finished = bool(spec.get("run", {}).get("skip_finished", True)) and not args.rerun_all

    config_paths = _config_paths(spec)
    if args.limit > 0:
        config_paths = config_paths[: args.limit]

    rows: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()
    print(f"Moran null square study: {len(config_paths)} configs", flush=True)

    for i, cfg_path in enumerate(config_paths, start=1):
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        run_name = raw["output"]["run_name"]
        out_dir = REPO / raw["output"]["out_dir"]
        if skip_finished and _is_done(out_dir, run_name):
            print(f"[{i}/{len(config_paths)}] skip (done): {run_name}", flush=True)
            rows.append({"run_name": run_name, "status": "skipped", "config": str(cfg_path)})
            continue

        print(f"[{i}/{len(config_paths)}] running: {run_name}", flush=True)
        try:
            row = run_one(cfg_path)
        except Exception as exc:
            print(f"  FAILED {run_name}: {exc}", flush=True)
            row = {"run_name": run_name, "status": "failed", "error": str(exc)}
        rows.append(row)
        print(f"  -> {row.get('status')}  moran_plot={row.get('moran_plot')}", flush=True)

    summary = {
        "experiment_name": spec.get("experiment_name", "moran_null_square"),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
    }
    summary_path = output_root / "study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
