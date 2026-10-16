from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.schemas import RunConfig, SUPPORTED_EXISTENCE_METHODS, linear_x_n_signal_genes, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.core.study_io import load_result_payload
from experiments.core.study_spec import (
    analysis_dir_for_output,
    common_study_fields,
    load_spec,
    manifest_path_for_output,
    resolve_repo_path as _resolve_repo_path,
)


def _fraction_key(value: float) -> float:
    return round(float(value), 8)


def _fraction_token(value: float) -> str:
    return f"{float(value):.4g}".replace(".", "p")


@dataclass
class LinearXExistenceStudySpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    signal_fractions: list[float]
    seeds: list[int]
    reuse_result_roots: list[Path] = field(default_factory=list)

    def validate(self) -> "LinearXExistenceStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")

        self.signal_fractions = [_fraction_key(value) for value in self.signal_fractions]
        if not self.signal_fractions:
            raise ValueError("signal_fractions must contain at least one value")
        if len(set(self.signal_fractions)) != len(self.signal_fractions):
            raise ValueError("signal_fractions must not contain duplicates")
        if any(not (0.0 <= value <= 1.0) for value in self.signal_fractions):
            raise ValueError("signal_fractions entries must lie in [0, 1]")

        self.seeds = [int(value) for value in self.seeds]
        if not self.seeds:
            raise ValueError("seeds must contain at least one value")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.reuse_result_roots = [path.resolve() for path in self.reuse_result_roots]
        return self


@dataclass(frozen=True)
class LinearXExistenceCondition:
    seed: int
    signal_fraction: float
    run_name: str


def load_linear_x_existence_spec(path: str | Path) -> LinearXExistenceStudySpec:
    payload = load_spec(path)
    experiment_name, base_config, output_root = common_study_fields(payload)
    spec = LinearXExistenceStudySpec(
        experiment_name=experiment_name,
        base_config=base_config,
        output_root=output_root,
        signal_fractions=list(payload.get("signal_fractions", [])),
        seeds=list(payload.get("seeds", [])),
        reuse_result_roots=[_resolve_repo_path(value) for value in payload.get("reuse_result_roots", [])],
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "synthetic":
        raise ValueError("base_config must use data.source='synthetic'")
    if base_run_config.data.mode != "linear_x":
        raise ValueError("base_config must use data.mode='linear_x'")
    if base_run_config.test.method not in SUPPORTED_EXISTENCE_METHODS:
        raise ValueError(
            "base_config must use an existence test.method in "
            f"{sorted(SUPPORTED_EXISTENCE_METHODS)}"
        )
    for fraction in spec.signal_fractions:
        linear_x_n_signal_genes(base_run_config.data.n_genes, fraction)
    return spec


def expand_linear_x_existence_conditions(spec: LinearXExistenceStudySpec) -> list[LinearXExistenceCondition]:
    conditions: list[LinearXExistenceCondition] = []
    for fraction in spec.signal_fractions:
        for seed in spec.seeds:
            conditions.append(
                LinearXExistenceCondition(
                    seed=int(seed),
                    signal_fraction=float(fraction),
                    run_name=(
                        f"{spec.experiment_name}__frac-{_fraction_token(fraction)}__seed-{int(seed):03d}"
                    ),
                )
            )
    return conditions


def build_linear_x_existence_run_config(
    base_run_config: RunConfig,
    spec: LinearXExistenceStudySpec,
    condition: LinearXExistenceCondition,
) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("data", {})
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})
    mapping["data"]["source"] = "synthetic"
    mapping["data"]["mode"] = "linear_x"
    mapping["data"]["signal_fraction"] = float(condition.signal_fraction)
    mapping["data"]["seed"] = int(condition.seed)
    mapping["test"]["seed"] = int(condition.seed)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: LinearXExistenceStudySpec) -> Path:
    return manifest_path_for_output(spec.output_root)


def analysis_dir_for_spec(spec: LinearXExistenceStudySpec) -> Path:
    return analysis_dir_for_output(spec.output_root)


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


def extract_linear_x_record(
    result_json_path: str | Path,
    *,
    expected_sigma: float,
    allowed_fractions: list[float],
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(result_json_path).resolve()
    payload = load_result_payload(path)
    warnings: list[dict[str, Any]] = []
    if payload.get("method_name") not in SUPPORTED_EXISTENCE_METHODS:
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
    if source != "synthetic" or mode != "linear_x":
        return None, warnings

    run_name = str(config.get("output", {}).get("run_name", path.stem.replace("_result", "")))
    sigma_value = data_cfg.get("sigma", dataset_meta.get("sigma"))
    fraction_value = data_cfg.get("signal_fraction", dataset_meta.get("signal_fraction"))
    seed_value = data_cfg.get("seed", dataset_meta.get("seed"))
    if sigma_value is None or fraction_value is None or payload.get("p_value") is None:
        warnings.append(
            {
                "warning_type": "missing_metadata",
                "result_json_path": str(path),
                "run_name": run_name,
                "message": "Skipping result because sigma, signal_fraction, or p_value could not be inferred",
            }
        )
        return None, warnings

    sigma = float(sigma_value)
    if sigma != float(expected_sigma):
        warnings.append(
            {
                "warning_type": "config_mismatch",
                "result_json_path": str(path),
                "run_name": run_name,
                "message": "Skipping result because sigma does not match the sweep base config",
            }
        )
        return None, warnings

    fraction = _fraction_key(float(fraction_value))
    allowed = {_fraction_key(value) for value in allowed_fractions}
    if fraction not in allowed:
        warnings.append(
            {
                "warning_type": "fraction_not_in_spec",
                "result_json_path": str(path),
                "run_name": run_name,
                "message": f"Skipping result because signal_fraction={fraction} is not in the spec",
            }
        )
        return None, warnings

    record = {
        "result_json_path": str(path),
        "run_name": run_name,
        "method_name": str(payload.get("method_name")),
        "metric": str(payload.get("metric")),
        "signal_fraction": fraction,
        "sigma": sigma,
        "seed": int(seed_value) if seed_value is not None else -1,
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
        "truth_source": "manifest" if manifest_entry is not None else "metadata",
    }
    return record, warnings


def summarize_signal_fraction_rows(
    rows: list[Mapping[str, object]],
    *,
    alpha: float,
) -> list[dict[str, object]]:
    grouped: dict[float, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_fraction_key(float(row["signal_fraction"])), []).append(row)

    summaries: list[dict[str, object]] = []
    for fraction in sorted(grouped):
        group_rows = grouped[fraction]
        p_values = np.asarray([float(row["p_value"]) for row in group_rows], dtype=np.float64)
        stat_true = np.asarray([float(row["stat_true"]) for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "signal_fraction": fraction,
                "n_runs": int(len(group_rows)),
                "p_value_mean": float(np.mean(p_values)),
                "p_value_std": float(np.std(p_values)),
                "p_value_min": float(np.min(p_values)),
                "p_value_max": float(np.max(p_values)),
                "n_p_below_alpha": int(np.sum(p_values < float(alpha))),
                "alpha": float(alpha),
                "stat_true_mean": float(np.mean(stat_true)),
            }
        )
    return summaries
