from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp

from data.schemas import RunConfig, SUPPORTED_EXISTENCE_METHODS, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.core.paths import repo_root
from experiments.core.study_io import load_result_payload, scan_result_json_paths, write_csv

REPO_ROOT = repo_root(__file__)


@dataclass
class RandomGenePanelNullStudySpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    n_panels: int = 30
    panel_size: int = 200
    panel_seeds: list[int] = field(default_factory=list)
    n_perms: int = 249
    n_reruns: int = 30
    universe_min_cells_per_gene: int = 3
    include_target_run: bool = True

    def validate(self) -> "RandomGenePanelNullStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if self.n_panels <= 0:
            raise ValueError("n_panels must be > 0")
        if self.panel_size <= 0:
            raise ValueError("panel_size must be > 0")
        self.n_perms = int(self.n_perms)
        self.n_reruns = int(self.n_reruns)
        if self.n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if self.n_reruns <= 0:
            raise ValueError("n_reruns must be > 0")
        if self.universe_min_cells_per_gene <= 0:
            raise ValueError("universe_min_cells_per_gene must be > 0")

        self.panel_seeds = [int(value) for value in self.panel_seeds]
        if not self.panel_seeds:
            self.panel_seeds = list(range(int(self.n_panels)))
        if len(self.panel_seeds) != int(self.n_panels):
            raise ValueError("panel_seeds length must match n_panels")

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        return self


@dataclass(frozen=True)
class RandomGenePanelCondition:
    condition_type: str
    panel_index: int
    panel_seed: int
    run_name: str
    gene_list: list[str]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_random_gene_panel_null_spec(path: str | Path) -> RandomGenePanelNullStudySpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = RandomGenePanelNullStudySpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        n_panels=int(payload.get("n_panels", 30)),
        panel_size=int(payload.get("panel_size", 200)),
        panel_seeds=list(payload.get("panel_seeds", [])),
        n_perms=int(payload.get("n_perms", 249)),
        n_reruns=int(payload.get("n_reruns", 30)),
        universe_min_cells_per_gene=int(payload.get("universe_min_cells_per_gene", 3)),
        include_target_run=bool(payload.get("include_target_run", True)),
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "h5ad":
        raise ValueError("base_config data.source must be 'h5ad' for random gene-panel null studies")
    if base_run_config.test.method not in SUPPORTED_EXISTENCE_METHODS:
        raise ValueError(
            "base_config must use an existence test.method in "
            f"{sorted(SUPPORTED_EXISTENCE_METHODS)}"
        )
    target_gene_list = base_run_config.data.gene_list
    if not target_gene_list:
        raise ValueError("base_config data.gene_list must be set for the target hypoxia panel")
    return spec


def sample_random_gene_panel(
    eligible_genes: list[str],
    *,
    panel_size: int,
    seed: int,
) -> list[str]:
    if panel_size > len(eligible_genes):
        raise ValueError(
            f"panel_size={panel_size} exceeds eligible gene universe size {len(eligible_genes)}"
        )
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(eligible_genes), size=int(panel_size), replace=False)
    return [eligible_genes[int(index)] for index in sorted(indices)]


def build_eligible_gene_universe(spec: RandomGenePanelNullStudySpec) -> list[str]:
    """Return genes passing expression support within the base config's obs subset (pre-crop)."""
    base_run_config = build_run_config(str(spec.base_config), {})
    data_cfg = base_run_config.data

    h5ad_path = _resolve_repo_path(str(data_cfg.h5ad))
    adata = ad.read_h5ad(h5ad_path, backed="r")
    from data.h5ad_loader import _apply_obs_subset

    adata = _apply_obs_subset(
        adata,
        obs_filters=data_cfg.obs_filters,
        obs_numeric_filters=data_cfg.obs_numeric_filters,
        obs_indices=data_cfg.obs_indices,
        obs_drop_na=data_cfg.obs_drop_na,
    )
    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()

    layer = data_cfg.layer
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        counts = adata.layers[layer]
    elif data_cfg.use_raw:
        if adata.raw is None:
            raise ValueError("use_raw requested but adata.raw is None")
        counts = adata.raw.X
    else:
        counts = adata.X

    if sp.issparse(counts):
        n_expr_cells = np.asarray((counts > 0).sum(axis=0)).reshape(-1)
    else:
        n_expr_cells = np.asarray(counts > 0, dtype=np.int64).sum(axis=0)

    var_names = np.asarray([str(name) for name in adata.var_names], dtype=object)
    keep_mask = n_expr_cells >= int(spec.universe_min_cells_per_gene)
    eligible = sorted(var_names[keep_mask].tolist())
    if len(eligible) < int(spec.panel_size):
        raise ValueError(
            f"Eligible gene universe size {len(eligible)} is smaller than panel_size={spec.panel_size}"
        )
    return eligible


def expand_conditions(
    spec: RandomGenePanelNullStudySpec,
    *,
    eligible_genes: list[str],
    target_gene_list: list[str],
) -> list[RandomGenePanelCondition]:
    conditions: list[RandomGenePanelCondition] = []
    if spec.include_target_run:
        conditions.append(
            RandomGenePanelCondition(
                condition_type="target",
                panel_index=-1,
                panel_seed=-1,
                run_name=f"{spec.experiment_name}__target",
                gene_list=[str(gene) for gene in target_gene_list],
            )
        )

    for panel_index, panel_seed in enumerate(spec.panel_seeds):
        gene_list = sample_random_gene_panel(
            eligible_genes,
            panel_size=int(spec.panel_size),
            seed=int(panel_seed),
        )
        conditions.append(
            RandomGenePanelCondition(
                condition_type="random_panel",
                panel_index=int(panel_index),
                panel_seed=int(panel_seed),
                run_name=(
                    f"{spec.experiment_name}__panel-{int(panel_index):03d}"
                    f"__seed-{int(panel_seed):03d}"
                ),
                gene_list=gene_list,
            )
        )
    return conditions


def build_condition_run_config(
    base_run_config: RunConfig,
    spec: RandomGenePanelNullStudySpec,
    condition: RandomGenePanelCondition,
) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("data", {})
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["data"]["gene_list"] = list(condition.gene_list)
    mapping["test"]["n_perms"] = int(spec.n_perms)
    mapping["test"]["n_reruns"] = int(spec.n_reruns)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: RandomGenePanelNullStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: RandomGenePanelNullStudySpec) -> Path:
    return spec.output_root / "analysis"


def eligible_universe_path_for_spec(spec: RandomGenePanelNullStudySpec) -> Path:
    return spec.output_root / "eligible_gene_universe.json"


def _surviving_gene_count_from_result(payload: Mapping[str, Any]) -> Optional[int]:
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return None

    per_type = artifacts.get("per_type_summaries")
    if isinstance(per_type, Mapping):
        for type_summary in per_type.values():
            if not isinstance(type_summary, Mapping):
                continue
            artifact_paths = type_summary.get("artifact_paths", {})
            if not isinstance(artifact_paths, Mapping):
                continue
            npz_path = artifact_paths.get("isodepths_npz")
            if npz_path is None:
                continue
            npz_file = Path(str(npz_path))
            if not npz_file.exists():
                continue
            with np.load(npz_file, allow_pickle=False) as npz:
                if "A" in npz:
                    return int(np.asarray(npz["A"]).shape[1])
    return None


def extract_panel_result_payload(
    result_json_path: str | Path,
    *,
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(result_json_path).resolve()
    warnings: list[dict[str, Any]] = []
    try:
        payload = load_result_payload(path)
    except Exception as exc:
        return None, [
            {
                "warning_type": "unreadable_result",
                "result_json_path": str(path),
                "run_name": "",
                "message": str(exc),
            }
        ]

    if payload.get("method_name") not in SUPPORTED_EXISTENCE_METHODS:
        warnings.append(
            {
                "warning_type": "unexpected_method",
                "result_json_path": str(path),
                "run_name": str((manifest_entry or {}).get("run_name", "")),
                "message": f"Skipping result because method_name={payload.get('method_name')!r}",
            }
        )
        return None, warnings

    stat_perm = np.asarray(payload.get("stat_perm", []), dtype=np.float64)
    if stat_perm.ndim != 1 or stat_perm.size == 0:
        warnings.append(
            {
                "warning_type": "missing_stat_perm",
                "result_json_path": str(path),
                "run_name": str((manifest_entry or {}).get("run_name", "")),
                "message": "Skipping result because stat_perm is missing or invalid",
            }
        )
        return None, warnings

    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    test_cfg = config.get("test", {})
    output_cfg = config.get("output", {})
    data_cfg = config.get("data", {})
    if not isinstance(test_cfg, Mapping):
        test_cfg = {}
    if not isinstance(output_cfg, Mapping):
        output_cfg = {}
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}

    gene_list = data_cfg.get("gene_list") or []
    if not isinstance(gene_list, list):
        gene_list = []

    surviving_gene_count = _surviving_gene_count_from_result(payload)
    if surviving_gene_count is None:
        surviving_gene_count = int(payload.get("n_genes", 0))

    entry = manifest_entry or {}
    record = {
        "result_json_path": str(path),
        "run_name": str(output_cfg.get("run_name", path.stem.replace("_result", ""))),
        "condition_type": str(entry.get("condition_type", "unknown")),
        "panel_index": int(entry.get("panel_index", -1)),
        "panel_seed": int(entry.get("panel_seed", -1)),
        "gene_list_requested_count": int(len(gene_list)),
        "n_genes_surviving": int(surviving_gene_count),
        "method_name": str(payload.get("method_name")),
        "metric": str(payload.get("metric")),
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
        "n_cells": int(payload.get("n_cells")),
        "n_perms": int(test_cfg.get("n_perms", stat_perm.size)),
        "n_reruns": int(test_cfg.get("n_reruns", -1)),
        "null_mean": float(np.mean(stat_perm)),
        "null_std": float(np.std(stat_perm)),
        "null_min": float(np.min(stat_perm)),
        "null_max": float(np.max(stat_perm)),
        "stat_perm": stat_perm,
    }
    return record, warnings


def load_manifest_entries(path: str | Path) -> dict[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries: dict[Path, dict[str, Any]] = {}
    for entry in payload.get("runs", []):
        entries[Path(entry["result_json_path"]).resolve()] = dict(entry)
    return entries


def compute_target_rank(
    target_value: float,
    null_values: np.ndarray,
    *,
    lower_is_better: bool,
) -> dict[str, float | int]:
    null_values = np.asarray(null_values, dtype=np.float64).reshape(-1)
    if null_values.size == 0:
        return {
            "rank": 0,
            "n_null": 0,
            "percentile": float("nan"),
            "fraction_better_or_equal": float("nan"),
        }

    if lower_is_better:
        better_or_equal = null_values <= target_value
    else:
        better_or_equal = null_values >= target_value

    rank = int(better_or_equal.sum()) + 1
    percentile = 100.0 * float((null_values < target_value).sum()) / float(null_values.size)
    if not lower_is_better:
        percentile = 100.0 - percentile
    return {
        "rank": rank,
        "n_null": int(null_values.size),
        "percentile": float(percentile),
        "fraction_better_or_equal": float(better_or_equal.mean()),
    }
