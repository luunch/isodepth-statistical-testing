"""Quick loader smoke test: verify obs_indices match expected cell counts."""
from __future__ import annotations

from experiments.core.paths import repo_root

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.schemas import DataConfig
from data.h5ad_loader import load_h5ad_dataset
from experiments.configuration import _resolve_config_relative_paths

CHECKS = [
    ("configs/cosmx_celltype_regions/cosmx_luad_9_r1_neutrophil_c35.json", 1587),
    ("configs/cosmx_celltype_regions/cosmx_luad_9_r1_tumor_9_c0.json", 6916),
]


def main() -> None:
    ok = True
    for cfg_rel, expected in CHECKS:
        cfg_path = ROOT / cfg_rel
        raw = json.loads(cfg_path.read_text())
        raw = _resolve_config_relative_paths(raw, str(cfg_path))
        dc = DataConfig(**raw["data"]).validate()
        bundle = load_h5ad_dataset(
            h5ad_path=dc.h5ad,
            spatial_key=dc.spatial_key,
            layer=dc.layer,
            obs_indices=dc.obs_indices,
            normalize_total=dc.normalize_total,
            log1p=dc.log1p,
            standardize_expression=dc.standardize_expression,
            seed=dc.seed,
        )
        n = bundle.S.shape[0]
        status = "ok" if n == expected else "MISMATCH"
        ok &= status == "ok"
        print(f"[{status}] {cfg_rel}: n={n} (expected {expected})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
