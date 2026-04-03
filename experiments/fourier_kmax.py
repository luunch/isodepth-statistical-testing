from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from data.schemas import RunConfig, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.existence_sigma import load_result_payload, scan_result_json_paths, write_csv


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FourierKmaxStudySpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    k_min: int
    k_max_values: list[int]
    seeds: list[int]
    reuse_result_roots: list[Path] = field(default_factory=list)

    def validate(self) -> "FourierKmaxStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if self.k_min <= 0:
            raise ValueError("k_min must be > 0")

        self.k_max_values = [int(value) for value in self.k_max_values]
        if not self.k_max_values:
            raise ValueError("k_max_values must contain at least one value")
        if any(value < self.k_min for value in self.k_max_values):
            raise ValueError("k_max_values entries must be >= k_min")

        self.seeds = [int(value) for value in self.seeds]
        if not self.seeds:
            raise ValueError("seeds must contain at least one value")

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.reuse_result_roots = [path.resolve() for path in self.reuse_result_roots]
        return self


@dataclass(frozen=True)
class FourierKmaxCondition:
    seed: int
    k_min: int
    k_max: int
    run_name: str


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_fourier_kmax_spec(path: str | Path) -> FourierKmaxStudySpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = FourierKmaxStudySpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        k_min=int(payload["k_min"]),
        k_max_values=list(payload.get("k_max_values", [])),
        seeds=list(payload.get("seeds", [])),
        reuse_result_roots=[_resolve_repo_path(value) for value in payload.get("reuse_result_roots", [])],
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "synthetic":
        raise ValueError("base_config must use data.source='synthetic'")
    if base_run_config.data.mode != "fourier":
        raise ValueError("base_config must use data.mode='fourier'")
    if base_run_config.test.method != "parallel_permutation":
        raise ValueError("base_config must use test.method='parallel_permutation'")
    return spec


def expand_fourier_kmax_conditions(spec: FourierKmaxStudySpec) -> list[FourierKmaxCondition]:
    conditions: list[FourierKmaxCondition] = []
    for k_max in spec.k_max_values:
        for seed in spec.seeds:
            run_name = (
                f"{spec.experiment_name}__kmin-{int(spec.k_min):02d}__kmax-{int(k_max):02d}__seed-{int(seed):03d}"
            )
            conditions.append(
                FourierKmaxCondition(
                    seed=int(seed),
                    k_min=int(spec.k_min),
                    k_max=int(k_max),
                    run_name=run_name,
                )
            )
    return conditions


def build_fourier_kmax_run_config(
    base_run_config: RunConfig,
    spec: FourierKmaxStudySpec,
    condition: FourierKmaxCondition,
) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("data", {})
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["data"]["source"] = "synthetic"
    mapping["data"]["mode"] = "fourier"
    mapping["data"]["seed"] = int(condition.seed)
    mapping["data"]["k_min"] = int(condition.k_min)
    mapping["data"]["k_max"] = int(condition.k_max)
    mapping["test"]["seed"] = int(condition.seed)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: FourierKmaxStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: FourierKmaxStudySpec) -> Path:
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


def extract_kmax_record(
    result_json_path: str | Path,
    *,
    expected_sigma: float,
    expected_dependent_xy: bool,
    expected_poly_degree: int,
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(result_json_path).resolve()
    payload = load_result_payload(path)
    warnings: list[dict[str, Any]] = []

    if payload.get("method_name") != "parallel_permutation":
        return None, warnings

    config = payload.get("config", {})
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    if not isinstance(data_cfg, Mapping):
        return None, warnings
    artifacts = payload.get("artifacts", {})
    dataset_meta = artifacts.get("dataset_meta", {}) if isinstance(artifacts, Mapping) else {}
    if not isinstance(dataset_meta, Mapping):
        dataset_meta = {}

    source = data_cfg.get("source", dataset_meta.get("source"))
    mode = data_cfg.get("mode", dataset_meta.get("mode"))
    if source != "synthetic" or mode != "fourier":
        return None, warnings

    sigma_value = data_cfg.get("sigma", dataset_meta.get("sigma"))
    dependent_xy_value = data_cfg.get("dependent_xy", dataset_meta.get("dependent_xy", True))
    poly_degree_value = data_cfg.get("poly_degree", dataset_meta.get("poly_degree", 3))
    k_min_value = data_cfg.get("k_min", dataset_meta.get("k_min"))
    k_max_value = data_cfg.get("k_max", dataset_meta.get("k_max"))
    seed_value = data_cfg.get("seed", dataset_meta.get("seed"))

    if sigma_value is None or k_min_value is None or k_max_value is None:
        warnings.append(
            {
                "warning_type": "missing_metadata",
                "result_json_path": str(path),
                "run_name": str(config.get("output", {}).get("run_name", path.stem.replace("_result", ""))),
                "message": "Skipping result because sigma, k_min, or k_max could not be inferred",
            }
        )
        return None, warnings

    sigma = float(sigma_value)
    dependent_xy = bool(dependent_xy_value)
    poly_degree = int(poly_degree_value)
    if (
        sigma != float(expected_sigma)
        or dependent_xy != bool(expected_dependent_xy)
        or poly_degree != int(expected_poly_degree)
    ):
        warnings.append(
            {
                "warning_type": "config_mismatch",
                "result_json_path": str(path),
                "run_name": str(config.get("output", {}).get("run_name", path.stem.replace("_result", ""))),
                "message": (
                    "Skipping result because sigma/dependent_xy/poly_degree do not match the sweep base config"
                ),
            }
        )
        return None, warnings

    k_min = int(k_min_value)
    k_max = int(k_max_value)
    run_name = str(config.get("output", {}).get("run_name", path.stem.replace("_result", "")))
    seed = int(seed_value) if seed_value is not None else -1

    record = {
        "result_json_path": str(path),
        "run_name": run_name,
        "method_name": str(payload.get("method_name")),
        "metric": str(payload.get("metric")),
        "k_min": k_min,
        "k_max": k_max,
        "sigma": sigma,
        "seed": seed,
        "dependent_xy": int(dependent_xy),
        "poly_degree": poly_degree,
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
        "truth_source": "manifest" if manifest_entry is not None else "metadata",
    }
    return record, warnings


def summarize_kmax_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k_max"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for k_max, group in sorted(grouped.items()):
        p_values = [float(item["p_value"]) for item in group]
        stat_true = [float(item["stat_true"]) for item in group]
        n_runs = len(group)
        p_mean = float(sum(p_values) / n_runs)
        p_std = float(math.sqrt(sum((value - p_mean) ** 2 for value in p_values) / n_runs)) if n_runs else math.nan
        summaries.append(
            {
                "k_max": int(k_max),
                "n_runs": int(n_runs),
                "p_value_mean": p_mean,
                "p_value_std": p_std,
                "p_value_min": float(min(p_values)),
                "p_value_max": float(max(p_values)),
                "stat_true_mean": float(sum(stat_true) / n_runs),
            }
        )
    return summaries


__all__ = [
    "FourierKmaxCondition",
    "FourierKmaxStudySpec",
    "analysis_dir_for_spec",
    "build_fourier_kmax_run_config",
    "expand_fourier_kmax_conditions",
    "extract_kmax_record",
    "load_fourier_kmax_spec",
    "load_manifest_entries",
    "manifest_path_for_spec",
    "scan_result_json_paths",
    "summarize_kmax_rows",
    "write_csv",
]
