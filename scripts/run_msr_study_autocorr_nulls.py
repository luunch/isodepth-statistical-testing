"""Generate MSR-surrogate autocorrelation null artifacts for kernel-noise MSR study runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.kernel_noise_msr_moran_null import DEFAULT_MSR_CALIBRATION_UM


def _run_name_from_result_path(path: Path) -> str:
    stem = path.stem
    return stem[:-7] if stem.endswith("_result") else stem


def _dataset_key_from_run_name(run_name: str) -> str | None:
    match = re.match(r"^(d\d+(?:p\d+)?_delta\d+(?:p\d+)?_seed\d+)_", run_name)
    return match.group(1) if match else None


def _autocorr_out_paths(
    run_dir: Path,
    run_name: str,
    *,
    truncate_um: float,
    n_perms: int,
) -> tuple[Path, Path]:
    slug_t = int(round(truncate_um))
    out_png = run_dir / f"{run_name}_msr_autocorr_t{slug_t}_n{n_perms}.png"
    return out_png, out_png.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate even if JSON exists")
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    runs_root = study_root / "runs"
    datasets_root = study_root / "datasets"
    script = REPO / "scripts/kernel_noise_msr_autocorr_null.py"

    result_paths = sorted(runs_root.glob("*/*_result.json"))
    if not result_paths:
        raise SystemExit(f"No MSR run result JSON files under {runs_root}")

    ok = skip = fail = 0
    for idx, result_json in enumerate(result_paths, start=1):
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        run_name = _run_name_from_result_path(result_json)
        dataset_key = _dataset_key_from_run_name(run_name)
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
        truncate_um = float(cfg.get("msr_truncate_um", 30.0))
        msr_radius = float(cfg.get("msr_neighbor_radius_um", 30.0))
        coord_um = float(cfg.get("coordinate_um_per_unit", 960.0))
        calibration_um = cfg.get("msr_calibration_um", None)
        run_seed = int(cfg.get("seed", 42))
        n_perms = int(len(payload.get("stat_perm") or []))
        if n_perms <= 0:
            n_perms = int(cfg.get("n_perms", 99))

        out_png, out_json = _autocorr_out_paths(
            result_json.parent, run_name, truncate_um=truncate_um, n_perms=n_perms,
        )
        if out_json.exists() and not args.force:
            print(f"[{idx}/{len(result_paths)}] SKIP existing {run_name}")
            skip += 1
            continue

        cal_used = float(calibration_um) if calibration_um is not None else DEFAULT_MSR_CALIBRATION_UM
        cmd = [
            args.python,
            str(script),
            "--cache", str(cache_path),
            "--n-perms", str(n_perms),
            "--truncate-um", str(truncate_um),
            "--msr-radius", str(msr_radius),
            "--coordinate-um-per-unit", str(coord_um),
            "--seed", str(run_seed),
            "--calibration-um", str(cal_used),
            "--out", str(out_png),
        ]
        print(
            f"[{idx}/{len(result_paths)}] RUN {run_name}  "
            f"trunc={truncate_um:g}  cal={cal_used:g}  seed={run_seed}  n={n_perms}"
        )
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode == 0 and out_json.exists():
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
