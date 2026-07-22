"""Generate matched block-perm Moran null plots for kernel_noise_square_study runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from methods.moran import DEFAULT_MORAN_NEIGHBOR_RADIUS_UM

REPO = Path(__file__).resolve().parent.parent


def _run_name_from_result_path(path: Path) -> str:
    stem = path.stem
    return stem[:-7] if stem.endswith("_result") else stem


def _dataset_key(run_name: str) -> str | None:
    match = re.match(r"^(d\d+_delta[^_]+_seed\d+)_block_r\d+$", run_name)
    return match.group(1) if match else None


def _moran_out_paths(
    run_dir: Path,
    run_name: str,
    *,
    block_radius_um: float,
    neighbor_radius_um: float,
    n_perms: int,
) -> tuple[Path, Path]:
    slug_br = int(round(block_radius_um))
    slug_nr = int(round(neighbor_radius_um))
    out_png = run_dir / (
        f"{run_name}_block_perm_moran_i_br{slug_br}_nr{slug_nr}_n{n_perms}.png"
    )
    return out_png, out_png.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=REPO / "results/experiments/kernel_noise_square_study",
    )
    parser.add_argument(
        "--neighbor-radius-um",
        type=float,
        default=DEFAULT_MORAN_NEIGHBOR_RADIUS_UM,
        help="Fixed Moran W radius for all runs (default: 30 µm).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for kernel_noise_block_perm_autocorr_length.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    runs_root = study_root / "runs"
    datasets_root = study_root / "datasets"
    script = REPO / "scripts/kernel_noise_block_perm_autocorr_length.py"
    neighbor_radius_um = float(args.neighbor_radius_um)

    result_paths = sorted(runs_root.glob("*_block_r*/*_result.json"))
    if not result_paths:
        raise SystemExit(f"No block run result JSON files under {runs_root}")

    ok = skip = fail = 0
    for idx, result_json in enumerate(result_paths, start=1):
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        run_name = _run_name_from_result_path(result_json)
        dataset_key = _dataset_key(run_name)
        if dataset_key is None:
            print(f"[{idx}/{len(result_paths)}] FAIL parse dataset key from {run_name}")
            fail += 1
            continue

        cache_path = datasets_root / f"{dataset_key}.npz"
        if not cache_path.exists():
            print(f"[{idx}/{len(result_paths)}] FAIL missing dataset cache {cache_path}")
            fail += 1
            continue

        cfg = (payload.get("config") or {}).get("test") or {}
        block_radius = float(cfg.get("block_radius", 60.0))
        block_shape = str(cfg.get("block_shape", "square"))
        block_jitter = bool(cfg.get("block_jitter", True))
        coord_um = float(cfg.get("coordinate_um_per_unit", 960.0))
        n_perms = int(len(payload.get("stat_perm") or []))
        if n_perms <= 0:
            n_perms = int(cfg.get("n_perms", 99))

        out_png, out_json = _moran_out_paths(
            result_json.parent,
            run_name,
            block_radius_um=block_radius,
            neighbor_radius_um=neighbor_radius_um,
            n_perms=n_perms,
        )

        cmd = [
            args.python,
            str(script),
            "--cache",
            str(cache_path),
            "--n-perms",
            str(n_perms),
            "--block-radius",
            str(block_radius),
            "--neighbor-radius",
            str(neighbor_radius_um),
            "--block-shape",
            block_shape,
            "--coordinate-um-per-unit",
            str(coord_um),
            "--out",
            str(out_png),
        ]
        cmd.append("--block-jitter" if block_jitter else "--no-block-jitter")

        print(
            f"[{idx}/{len(result_paths)}] RUN {run_name}  "
            f"br={block_radius:g}  nr={neighbor_radius_um:g}  n={n_perms}"
        )
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode == 0 and out_png.exists() and out_json.exists():
            ok += 1
        else:
            fail += 1
            print(f"[{idx}/{len(result_paths)}] FAIL {run_name} rc={proc.returncode}")

    print("--- SUMMARY ---")
    print(f"ok={ok} skip={skip} fail={fail} total={len(result_paths)}")
    if fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
