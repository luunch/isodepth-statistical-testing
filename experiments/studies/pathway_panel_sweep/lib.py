from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.schemas import RunConfig, SUPPORTED_EXISTENCE_METHODS, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.core.paths import repo_root
from experiments.core.study_io import load_result_payload, scan_result_json_paths, write_csv
from experiments.studies.random_gene_panel_null.lib import _surviving_gene_count_from_result

REPO_ROOT = repo_root(__file__)


@dataclass
class PathwayPanelSweepSpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    gmt_path: Path
    n_perms: int = 249
    n_reruns: int = 30
    alpha: float = 0.05
    min_requested_genes: int = 15

    def validate(self) -> "PathwayPanelSweepSpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if not self.gmt_path.exists():
            raise ValueError(f"gmt_path does not exist: {self.gmt_path}")
        self.n_perms = int(self.n_perms)
        self.n_reruns = int(self.n_reruns)
        if self.n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if self.n_reruns <= 0:
            raise ValueError("n_reruns must be > 0")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.min_requested_genes = int(self.min_requested_genes)
        if self.min_requested_genes <= 0:
            raise ValueError("min_requested_genes must be > 0")

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.gmt_path = self.gmt_path.resolve()
        return self


@dataclass(frozen=True)
class PathwayPanelCondition:
    pathway_index: int
    pathway_name: str
    run_name: str
    gene_list: list[str]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_gmt_gene_sets(gmt_path: str | Path) -> dict[str, list[str]]:
    """Load GMT file; returns pathway -> sorted gene symbol list."""
    path = Path(gmt_path)
    gene_sets: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            pathway = parts[0].strip()
            genes = sorted({g.strip() for g in parts[2:] if g.strip()})
            if pathway and genes:
                gene_sets[pathway] = genes
    if not gene_sets:
        raise ValueError(f"No gene sets found in GMT: {path}")
    return gene_sets


def load_pathway_panel_sweep_spec(path: str | Path) -> PathwayPanelSweepSpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = PathwayPanelSweepSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        gmt_path=_resolve_repo_path(payload["gmt_path"]),
        n_perms=int(payload.get("n_perms", 249)),
        n_reruns=int(payload.get("n_reruns", 30)),
        alpha=float(payload.get("alpha", 0.05)),
        min_requested_genes=int(payload.get("min_requested_genes", 15)),
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "h5ad":
        raise ValueError("base_config data.source must be 'h5ad' for pathway panel sweeps")
    if base_run_config.test.method not in SUPPORTED_EXISTENCE_METHODS:
        raise ValueError(
            "base_config must use an existence test.method in "
            f"{sorted(SUPPORTED_EXISTENCE_METHODS)}"
        )
    return spec


def expand_pathway_conditions(
    spec: PathwayPanelSweepSpec,
    *,
    gene_sets: dict[str, list[str]],
) -> list[PathwayPanelCondition]:
    conditions: list[PathwayPanelCondition] = []
    for pathway_index, pathway_name in enumerate(sorted(gene_sets.keys())):
        gene_list = list(gene_sets[pathway_name])
        if len(gene_list) < int(spec.min_requested_genes):
            continue
        safe_suffix = pathway_name.replace("HALLMARK_", "").lower()
        conditions.append(
            PathwayPanelCondition(
                pathway_index=int(pathway_index),
                pathway_name=pathway_name,
                run_name=f"{spec.experiment_name}__{safe_suffix}",
                gene_list=gene_list,
            )
        )
    if not conditions:
        raise ValueError("No pathways met min_requested_genes after loading GMT")
    return conditions


def build_pathway_run_config(
    base_run_config: RunConfig,
    spec: PathwayPanelSweepSpec,
    condition: PathwayPanelCondition,
) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("data", {})
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["data"]["gene_list"] = list(condition.gene_list)
    mapping["data"]["top_var_genes"] = 0
    mapping["test"]["n_perms"] = int(spec.n_perms)
    mapping["test"]["n_reruns"] = int(spec.n_reruns)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: PathwayPanelSweepSpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: PathwayPanelSweepSpec) -> Path:
    return spec.output_root / "analysis"


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


def extract_pathway_result_payload(
    result_json_path: str | Path,
    *,
    manifest_entry: Optional[Mapping[str, Any]] = None,
    alpha: float = 0.05,
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
                "pathway_name": "",
                "message": str(exc),
            }
        ]

    if payload.get("method_name") not in SUPPORTED_EXISTENCE_METHODS:
        warnings.append(
            {
                "warning_type": "unexpected_method",
                "result_json_path": str(path),
                "pathway_name": str((manifest_entry or {}).get("pathway_name", "")),
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
                "pathway_name": str((manifest_entry or {}).get("pathway_name", "")),
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
    p_value = float(payload.get("p_value"))
    record = {
        "result_json_path": str(path),
        "run_name": str(output_cfg.get("run_name", path.stem.replace("_result", ""))),
        "pathway_index": int(entry.get("pathway_index", -1)),
        "pathway_name": str(entry.get("pathway_name", "")),
        "gene_list_requested_count": int(len(gene_list)),
        "n_genes_surviving": int(surviving_gene_count),
        "p_value": p_value,
        "significant": bool(p_value < float(alpha)),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
        "n_cells": int(payload.get("n_cells")),
        "n_perms": int(test_cfg.get("n_perms", stat_perm.size)),
        "n_reruns": int(test_cfg.get("n_reruns", -1)),
        "null_mean": float(np.mean(stat_perm)),
        "null_std": float(np.std(stat_perm)),
        "null_min": float(np.min(stat_perm)),
        "null_max": float(np.max(stat_perm)),
    }
    return record, warnings
