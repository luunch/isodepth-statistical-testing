from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.schemas import RunConfig, SUPPORTED_EXISTENCE_METHODS, run_config_from_mapping
from experiments.configuration import build_run_config
from experiments.existence_sigma import load_result_payload, scan_result_json_paths, write_csv


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FourierKmaxExistencePerturbationStudySpec:
    experiment_name: str
    existence_base_config: Path
    perturbation_base_config: Path
    output_root: Path
    k_min: int
    k_max_values: list[int]
    seeds: list[int]
    delta_values: list[float]
    reuse_result_roots: list[Path] = field(default_factory=list)

    def validate(self) -> "FourierKmaxExistencePerturbationStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.existence_base_config.exists():
            raise ValueError(f"existence_base_config does not exist: {self.existence_base_config}")
        if not self.perturbation_base_config.exists():
            raise ValueError(f"perturbation_base_config does not exist: {self.perturbation_base_config}")
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

        self.delta_values = [float(value) for value in self.delta_values]
        if not self.delta_values:
            raise ValueError("delta_values must contain at least one value")
        if any(value <= 0.0 for value in self.delta_values):
            raise ValueError("delta_values entries must be > 0")

        self.existence_base_config = self.existence_base_config.resolve()
        self.perturbation_base_config = self.perturbation_base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.reuse_result_roots = [path.resolve() for path in self.reuse_result_roots]
        return self


@dataclass(frozen=True)
class FourierKmaxStudyCondition:
    test_kind: str
    seed: int
    k_min: int
    k_max: int
    run_name: str


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_fourier_kmax_existence_perturbation_spec(
    path: str | Path,
) -> FourierKmaxExistencePerturbationStudySpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = FourierKmaxExistencePerturbationStudySpec(
        experiment_name=str(payload["experiment_name"]),
        existence_base_config=_resolve_repo_path(payload["existence_base_config"]),
        perturbation_base_config=_resolve_repo_path(payload["perturbation_base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        k_min=int(payload["k_min"]),
        k_max_values=list(payload.get("k_max_values", [])),
        seeds=list(payload.get("seeds", [])),
        delta_values=list(payload.get("delta_values", [])),
        reuse_result_roots=[_resolve_repo_path(value) for value in payload.get("reuse_result_roots", [])],
    ).validate()

    existence_base = build_run_config(str(spec.existence_base_config), {})
    perturbation_base = build_run_config(str(spec.perturbation_base_config), {})
    for run_config in (existence_base, perturbation_base):
        if run_config.data.source != "synthetic":
            raise ValueError("base configs must use data.source='synthetic'")
        if run_config.data.mode != "fourier":
            raise ValueError("base configs must use data.mode='fourier'")
    if existence_base.test.method not in SUPPORTED_EXISTENCE_METHODS:
        raise ValueError(
            "existence_base_config must use an existence test.method in "
            f"{sorted(SUPPORTED_EXISTENCE_METHODS)}"
        )
    if perturbation_base.test.method != "perturbation_test":
        raise ValueError("perturbation_base_config must use test.method='perturbation_test'")
    return spec


def expand_fourier_kmax_study_conditions(
    spec: FourierKmaxExistencePerturbationStudySpec,
) -> list[FourierKmaxStudyCondition]:
    conditions: list[FourierKmaxStudyCondition] = []
    for k_max in spec.k_max_values:
        for seed in spec.seeds:
            for test_kind in ("existence", "perturbation"):
                conditions.append(
                    FourierKmaxStudyCondition(
                        test_kind=test_kind,
                        seed=int(seed),
                        k_min=int(spec.k_min),
                        k_max=int(k_max),
                        run_name=(
                            f"{spec.experiment_name}__test-{test_kind}__kmin-{int(spec.k_min):02d}"
                            f"__kmax-{int(k_max):03d}__seed-{int(seed):03d}"
                        ),
                    )
                )
    return conditions


def build_fourier_kmax_study_run_config(
    existence_base_run_config: RunConfig,
    perturbation_base_run_config: RunConfig,
    spec: FourierKmaxExistencePerturbationStudySpec,
    condition: FourierKmaxStudyCondition,
) -> RunConfig:
    base_run_config = existence_base_run_config if condition.test_kind == "existence" else perturbation_base_run_config
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
    if condition.test_kind == "perturbation":
        mapping["test"]["delta"] = [float(value) for value in spec.delta_values]
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def manifest_path_for_spec(spec: FourierKmaxExistencePerturbationStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: FourierKmaxExistencePerturbationStudySpec) -> Path:
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


def _extract_common_fourier_metadata(
    payload: Mapping[str, Any],
    *,
    expected_sigma: float,
    expected_dependent_xy: bool,
    expected_poly_degree: int,
) -> tuple[Optional[dict[str, object]], list[dict[str, object]]]:
    warnings: list[dict[str, object]] = []
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
    run_name = str(config.get("output", {}).get("run_name", ""))

    if sigma_value is None or k_min_value is None or k_max_value is None:
        warnings.append(
            {
                "warning_type": "missing_metadata",
                "result_json_path": "",
                "run_name": run_name,
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
                "result_json_path": "",
                "run_name": run_name,
                "message": "Skipping result because sigma/dependent_xy/poly_degree do not match the sweep base config",
            }
        )
        return None, warnings

    return (
        {
            "run_name": run_name,
            "k_min": int(k_min_value),
            "k_max": int(k_max_value),
            "sigma": sigma,
            "seed": int(seed_value) if seed_value is not None else -1,
            "dependent_xy": int(dependent_xy),
            "poly_degree": poly_degree,
            "metric": str(payload.get("metric")),
            "runtime_sec": float(payload.get("runtime_sec")),
        },
        warnings,
    )


def extract_existence_record(
    result_json_path: str | Path,
    *,
    expected_sigma: float,
    expected_dependent_xy: bool,
    expected_poly_degree: int,
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[dict[str, object]], list[dict[str, object]]]:
    path = Path(result_json_path).resolve()
    payload = load_result_payload(path)
    if payload.get("method_name") not in SUPPORTED_EXISTENCE_METHODS:
        return None, []

    common, warnings = _extract_common_fourier_metadata(
        payload,
        expected_sigma=expected_sigma,
        expected_dependent_xy=expected_dependent_xy,
        expected_poly_degree=expected_poly_degree,
    )
    for warning in warnings:
        warning["result_json_path"] = str(path)
    if common is None:
        return None, warnings

    record = {
        "result_json_path": str(path),
        "run_name": str(common["run_name"]),
        "method_name": str(payload.get("method_name")),
        "metric": str(common["metric"]),
        "k_min": int(common["k_min"]),
        "k_max": int(common["k_max"]),
        "sigma": float(common["sigma"]),
        "seed": int(common["seed"]),
        "dependent_xy": int(common["dependent_xy"]),
        "poly_degree": int(common["poly_degree"]),
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(common["runtime_sec"]),
        "truth_source": "manifest" if manifest_entry is not None else "metadata",
    }
    return record, warnings


def extract_perturbation_records(
    result_json_path: str | Path,
    *,
    expected_sigma: float,
    expected_dependent_xy: bool,
    expected_poly_degree: int,
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    path = Path(result_json_path).resolve()
    payload = load_result_payload(path)
    if payload.get("method_name") != "perturbation_test":
        return [], []

    common, warnings = _extract_common_fourier_metadata(
        payload,
        expected_sigma=expected_sigma,
        expected_dependent_xy=expected_dependent_xy,
        expected_poly_degree=expected_poly_degree,
    )
    for warning in warnings:
        warning["result_json_path"] = str(path)
    if common is None:
        return [], warnings

    artifacts = payload.get("artifacts", {})
    delta_summaries = artifacts.get("delta_summaries", {}) if isinstance(artifacts, Mapping) else {}
    if not isinstance(delta_summaries, Mapping):
        return [], warnings

    rows: list[dict[str, object]] = []
    for delta_key, summary in sorted(delta_summaries.items(), key=lambda item: float(item[0])):
        if not isinstance(summary, Mapping):
            continue
        rows.append(
            {
                "result_json_path": str(path),
                "run_name": str(common["run_name"]),
                "method_name": str(payload.get("method_name")),
                "metric": str(common["metric"]),
                "k_min": int(common["k_min"]),
                "k_max": int(common["k_max"]),
                "sigma": float(common["sigma"]),
                "seed": int(common["seed"]),
                "dependent_xy": int(common["dependent_xy"]),
                "poly_degree": int(common["poly_degree"]),
                "delta": float(summary.get("delta", float(delta_key))),
                "p_value": float(summary["p_value"]),
                "stat_true": float(summary.get("score_mean", payload.get("stat_true"))),
                "runtime_sec": float(common["runtime_sec"]),
                "truth_source": "manifest" if manifest_entry is not None else "metadata",
            }
        )
    return rows, warnings


def summarize_existence_rows(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k_max"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for k_max in sorted(grouped):
        group_rows = grouped[k_max]
        p_values = np.asarray([float(row["p_value"]) for row in group_rows], dtype=np.float64)
        stat_true = np.asarray([float(row["stat_true"]) for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "k_max": int(k_max),
                "n_runs": int(len(group_rows)),
                "p_value_mean": float(np.mean(p_values)),
                "p_value_std": float(np.std(p_values)),
                "p_value_min": float(np.min(p_values)),
                "p_value_max": float(np.max(p_values)),
                "stat_true_mean": float(np.mean(stat_true)),
            }
        )
    return summaries


def summarize_perturbation_rows(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["k_max"]), float(row["delta"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (k_max, delta), group_rows in sorted(grouped.items()):
        p_values = np.asarray([float(row["p_value"]) for row in group_rows], dtype=np.float64)
        stat_true = np.asarray([float(row["stat_true"]) for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "k_max": int(k_max),
                "delta": float(delta),
                "n_runs": int(len(group_rows)),
                "p_value_mean": float(np.mean(p_values)),
                "p_value_std": float(np.std(p_values)),
                "p_value_min": float(np.min(p_values)),
                "p_value_max": float(np.max(p_values)),
                "stat_true_mean": float(np.mean(stat_true)),
            }
        )
    return summaries


def build_boxplot_rows(
    existence_rows: list[Mapping[str, object]],
    perturbation_rows: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in existence_rows:
        rows.append(
            {
                "k_max": int(row["k_max"]),
                "test_label": "existence",
                "test_kind": "existence",
                "delta": "",
                "seed": int(row["seed"]),
                "p_value": float(row["p_value"]),
            }
        )
    for row in perturbation_rows:
        rows.append(
            {
                "k_max": int(row["k_max"]),
                "test_label": f"delta={float(row['delta']):.6g}",
                "test_kind": "perturbation",
                "delta": float(row["delta"]),
                "seed": int(row["seed"]),
                "p_value": float(row["p_value"]),
            }
        )
    return rows
