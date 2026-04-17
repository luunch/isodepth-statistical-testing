from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.schemas import RunConfig, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.existence_sigma import load_result_payload, scan_result_json_paths, write_csv
from methods.metrics import compute_metric


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RealDataExistenceConsistencyStudySpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    n_repeats: int = 20
    repeat_seeds: list[int] = field(default_factory=list)
    n_perms: int = 100
    reuse_result_roots: list[Path] = field(default_factory=list)

    def validate(self) -> "RealDataExistenceConsistencyStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")
        self.n_perms = int(self.n_perms)
        if self.n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        self.repeat_seeds = [int(value) for value in self.repeat_seeds]
        if not self.repeat_seeds:
            self.repeat_seeds = list(range(int(self.n_repeats)))
        if not self.repeat_seeds:
            raise ValueError("repeat_seeds must contain at least one value")
        if len(self.repeat_seeds) != int(self.n_repeats):
            raise ValueError("repeat_seeds length must match n_repeats")

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.reuse_result_roots = [path.resolve() for path in self.reuse_result_roots]
        return self


@dataclass(frozen=True)
class RealDataExistenceRepeatCondition:
    repeat_index: int
    seed: int
    run_name: str


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_real_data_existence_consistency_spec(path: str | Path) -> RealDataExistenceConsistencyStudySpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = RealDataExistenceConsistencyStudySpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        n_repeats=int(payload.get("n_repeats", 20)),
        repeat_seeds=list(payload.get("repeat_seeds", [])),
        n_perms=int(payload.get("n_perms", 100)),
        reuse_result_roots=[_resolve_repo_path(value) for value in payload.get("reuse_result_roots", [])],
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "h5ad":
        raise ValueError("base_config must use data.source='h5ad'")
    if base_run_config.test.method != "parallel_permutation":
        raise ValueError("base_config must use test.method='parallel_permutation'")
    return spec


def expand_real_data_existence_repeat_conditions(
    spec: RealDataExistenceConsistencyStudySpec,
) -> list[RealDataExistenceRepeatCondition]:
    conditions: list[RealDataExistenceRepeatCondition] = []
    for repeat_index, seed in enumerate(spec.repeat_seeds):
        conditions.append(
            RealDataExistenceRepeatCondition(
                repeat_index=int(repeat_index),
                seed=int(seed),
                run_name=(
                    f"{spec.experiment_name}__repeat-{int(repeat_index):03d}"
                    f"__seed-{int(seed):03d}"
                ),
            )
        )
    return conditions


def build_repeat_run_config(
    base_run_config: RunConfig,
    spec: RealDataExistenceConsistencyStudySpec,
    condition: RealDataExistenceRepeatCondition,
) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["test"]["seed"] = int(condition.seed)
    mapping["test"]["n_perms"] = int(spec.n_perms)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: RealDataExistenceConsistencyStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: RealDataExistenceConsistencyStudySpec) -> Path:
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


def extract_repeat_payload(
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

    if payload.get("method_name") != "parallel_permutation":
        warnings.append(
            {
                "warning_type": "unexpected_method",
                "result_json_path": str(path),
                "run_name": str((manifest_entry or {}).get("run_name", "")),
                "message": f"Skipping result because method_name={payload.get('method_name')!r}",
            }
        )
        return None, warnings

    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    test_cfg = config.get("test", {})
    output_cfg = config.get("output", {})
    data_cfg = config.get("data", {})
    artifacts = payload.get("artifacts", {})
    if not isinstance(test_cfg, Mapping):
        test_cfg = {}
    if not isinstance(output_cfg, Mapping):
        output_cfg = {}
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}
    if not isinstance(artifacts, Mapping):
        artifacts = {}

    stat_perm = np.asarray(payload.get("stat_perm", []), dtype=np.float64)
    if stat_perm.ndim != 1 or stat_perm.size == 0:
        warnings.append(
            {
                "warning_type": "missing_stat_perm",
                "result_json_path": str(path),
                "run_name": str(output_cfg.get("run_name", "")),
                "message": "Skipping result because stat_perm is missing or invalid",
            }
        )
        return None, warnings

    true_isodepth = artifacts.get("true_isodepth")
    if true_isodepth is None:
        warnings.append(
            {
                "warning_type": "missing_true_isodepth",
                "result_json_path": str(path),
                "run_name": str(output_cfg.get("run_name", "")),
                "message": "Skipping result because artifacts.true_isodepth is missing",
            }
        )
        return None, warnings

    true_isodepth_array = np.asarray(true_isodepth, dtype=np.float64).reshape(-1)
    if true_isodepth_array.size == 0:
        warnings.append(
            {
                "warning_type": "empty_true_isodepth",
                "result_json_path": str(path),
                "run_name": str(output_cfg.get("run_name", "")),
                "message": "Skipping result because artifacts.true_isodepth is empty",
            }
        )
        return None, warnings

    record = {
        "result_json_path": str(path),
        "run_name": str(output_cfg.get("run_name", path.stem.replace("_result", ""))),
        "method_name": str(payload.get("method_name")),
        "metric": str(payload.get("metric")),
        "seed": int(test_cfg.get("seed", data_cfg.get("seed", -1))),
        "repeat_index": int((manifest_entry or {}).get("repeat_index", -1)),
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
        "n_perms": int(test_cfg.get("n_perms", stat_perm.size)),
        "null_mean": float(np.mean(stat_perm)),
        "null_std": float(np.std(stat_perm)),
        "null_min": float(np.min(stat_perm)),
        "null_max": float(np.max(stat_perm)),
        "stat_perm": stat_perm,
        "true_isodepth": true_isodepth_array,
    }
    return record, warnings


def build_null_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for row in rows:
        stat_perm = np.asarray(row["stat_perm"], dtype=np.float64)
        for null_index, null_loss in enumerate(stat_perm):
            expanded.append(
                {
                    "repeat_index": int(row["repeat_index"]),
                    "seed": int(row["seed"]),
                    "null_sample_index": int(null_index),
                    "null_loss": float(null_loss),
                }
            )
    return expanded


def build_pairwise_isodepth_rows(rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, object]], np.ndarray]:
    n_runs = len(rows)
    matrix = np.eye(n_runs, dtype=np.float64)
    pairwise_rows: list[dict[str, object]] = []

    for i, row_i in enumerate(rows):
        depth_i = np.asarray(row_i["true_isodepth"], dtype=np.float64).reshape(-1, 1)
        for j, row_j in enumerate(rows):
            depth_j = np.asarray(row_j["true_isodepth"], dtype=np.float64).reshape(-1, 1)
            if depth_i.shape[0] != depth_j.shape[0]:
                raise ValueError("All true_isodepth arrays must have the same length")
            spearman = float(compute_metric("spearman_corr_mean", depth_i, depth_j))
            matrix[i, j] = spearman
            pairwise_rows.append(
                {
                    "repeat_i": int(row_i["repeat_index"]),
                    "repeat_j": int(row_j["repeat_index"]),
                    "seed_i": int(row_i["seed"]),
                    "seed_j": int(row_j["seed"]),
                    "spearman_true_isodepth": spearman,
                }
            )

    return pairwise_rows, matrix


def build_matrix_rows(rows: list[Mapping[str, Any]], matrix: np.ndarray) -> list[dict[str, object]]:
    header = ["repeat_index"] + [str(int(row["repeat_index"])) for row in rows]
    matrix_rows: list[dict[str, object]] = []
    for row_index, run_row in enumerate(rows):
        current = {"repeat_index": int(run_row["repeat_index"])}
        for col_index, column_name in enumerate(header[1:]):
            current[column_name] = float(matrix[row_index, col_index])
        matrix_rows.append(current)
    return matrix_rows
