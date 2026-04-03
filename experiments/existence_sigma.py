from __future__ import annotations

import copy
import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from data.schemas import DatasetBundle, RunConfig, run_config_from_mapping
from experiments.configuration import build_run_config


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_NULL_FAMILIES = {"matched_shuffled"}
RUN_NAME_MODE_PATTERN = re.compile(r"(?:^|[_-])(radial|noise|fourier)(?:[_-]|$)")


@dataclass
class ExistenceSigmaStudySpec:
    experiment_name: str
    base_config: Path
    output_root: Path
    alpha: float
    sigma_values: list[float]
    seeds: list[int]
    include_radial: bool
    fourier_k_values: list[int]
    null_family: str
    reuse_result_roots: list[Path] = field(default_factory=list)

    def validate(self) -> "ExistenceSigmaStudySpec":
        if not self.experiment_name:
            raise ValueError("experiment_name is required")
        if not self.base_config.exists():
            raise ValueError(f"base_config does not exist: {self.base_config}")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")

        self.sigma_values = [float(value) for value in self.sigma_values]
        if not self.sigma_values:
            raise ValueError("sigma_values must contain at least one value")
        if any(value < 0.0 for value in self.sigma_values):
            raise ValueError("sigma_values entries must be >= 0")

        self.seeds = [int(value) for value in self.seeds]
        if not self.seeds:
            raise ValueError("seeds must contain at least one value")

        self.fourier_k_values = [int(value) for value in self.fourier_k_values]
        if any(value <= 0 for value in self.fourier_k_values):
            raise ValueError("fourier_k_values entries must be > 0")
        if not self.include_radial and not self.fourier_k_values:
            raise ValueError("At least one of include_radial or fourier_k_values must be enabled")

        if self.null_family not in SUPPORTED_NULL_FAMILIES:
            raise ValueError(
                f"Unsupported null_family '{self.null_family}'. Expected one of {sorted(SUPPORTED_NULL_FAMILIES)}"
            )

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.reuse_result_roots = [path.resolve() for path in self.reuse_result_roots]
        return self


@dataclass(frozen=True)
class SweepCondition:
    truth_label: str
    mode: str
    sigma: float
    seed: int
    k: Optional[int]
    null_family: Optional[str]
    run_name: str

    @property
    def family_label(self) -> str:
        if self.mode == "fourier":
            return f"fourier_k={int(self.k)}"
        return self.mode


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _format_number_for_name(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _build_run_name(spec: ExistenceSigmaStudySpec, condition: SweepCondition) -> str:
    parts = [
        spec.experiment_name,
        f"truth-{condition.truth_label}",
        f"mode-{condition.mode}",
        f"sigma-{_format_number_for_name(condition.sigma)}",
        f"seed-{int(condition.seed):03d}",
    ]
    if condition.k is not None:
        parts.append(f"k-{int(condition.k):02d}")
    if condition.null_family is not None:
        parts.append(f"null-{condition.null_family}")
    return "__".join(parts)


def load_existence_sigma_spec(path: str | Path) -> ExistenceSigmaStudySpec:
    spec_path = Path(path).resolve()
    with open(spec_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    spec = ExistenceSigmaStudySpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        alpha=float(payload.get("alpha", 0.05)),
        sigma_values=list(payload.get("sigma_values", [])),
        seeds=list(payload.get("seeds", [])),
        include_radial=bool(payload.get("include_radial", True)),
        fourier_k_values=list(payload.get("fourier_k_values", [])),
        null_family=str(payload.get("null_family", "matched_shuffled")),
        reuse_result_roots=[_resolve_repo_path(value) for value in payload.get("reuse_result_roots", [])],
    ).validate()

    base_run_config = build_run_config(str(spec.base_config), {})
    if base_run_config.data.source != "synthetic":
        raise ValueError("base_config must use data.source='synthetic'")
    if base_run_config.test.method != "parallel_permutation":
        raise ValueError("base_config must use test.method='parallel_permutation'")
    return spec


def expand_existence_sigma_conditions(spec: ExistenceSigmaStudySpec) -> list[SweepCondition]:
    conditions: list[SweepCondition] = []
    for sigma in spec.sigma_values:
        for seed in spec.seeds:
            if spec.include_radial:
                for truth_label in ("alternative", "null"):
                    null_family = spec.null_family if truth_label == "null" else None
                    condition = SweepCondition(
                        truth_label=truth_label,
                        mode="radial",
                        sigma=float(sigma),
                        seed=int(seed),
                        k=None,
                        null_family=null_family,
                        run_name="",
                    )
                    conditions.append(
                        SweepCondition(
                            **{
                                **asdict(condition),
                                "run_name": _build_run_name(spec, condition),
                            }
                        )
                    )

            for k in spec.fourier_k_values:
                for truth_label in ("alternative", "null"):
                    null_family = spec.null_family if truth_label == "null" else None
                    condition = SweepCondition(
                        truth_label=truth_label,
                        mode="fourier",
                        sigma=float(sigma),
                        seed=int(seed),
                        k=int(k),
                        null_family=null_family,
                        run_name="",
                    )
                    conditions.append(
                        SweepCondition(
                            **{
                                **asdict(condition),
                                "run_name": _build_run_name(spec, condition),
                            }
                        )
                    )
    return conditions


def build_condition_run_config(base_run_config: RunConfig, spec: ExistenceSigmaStudySpec, condition: SweepCondition) -> RunConfig:
    mapping = copy.deepcopy(base_run_config.to_dict())
    mapping.setdefault("data", {})
    mapping.setdefault("test", {})
    mapping.setdefault("output", {})

    mapping["data"]["source"] = "synthetic"
    mapping["data"]["mode"] = condition.mode
    mapping["data"]["sigma"] = float(condition.sigma)
    mapping["data"]["seed"] = int(condition.seed)
    mapping["data"].pop("k", None)
    if condition.k is not None:
        mapping["data"]["k_min"] = 1
        mapping["data"]["k_max"] = int(condition.k)
    else:
        mapping["data"].pop("k_min", None)
        mapping["data"].pop("k_max", None)
    mapping["test"]["method"] = "parallel_permutation"
    mapping["test"]["seed"] = int(condition.seed)
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def annotate_alternative_dataset(dataset: DatasetBundle, condition: SweepCondition) -> DatasetBundle:
    meta = dict(dataset.meta or {})
    meta["truth_label"] = "alternative"
    meta["generating_mode"] = condition.mode
    meta["null_family"] = None
    meta["shuffled_expression"] = False
    if condition.k is not None:
        meta["k"] = int(condition.k)
    return DatasetBundle(
        S=np.asarray(dataset.S, dtype=np.float32),
        A=np.asarray(dataset.A, dtype=np.float32),
        meta=meta,
    ).validate()


def matched_shuffle_dataset(dataset: DatasetBundle, *, seed: int, condition: SweepCondition) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dataset.n_cells)
    meta = dict(dataset.meta or {})
    meta["truth_label"] = "null"
    meta["generating_mode"] = condition.mode
    meta["null_family"] = condition.null_family
    meta["shuffled_expression"] = True
    meta["shuffle_seed"] = int(seed)
    if condition.k is not None:
        meta["k"] = int(condition.k)
    return DatasetBundle(
        S=np.asarray(dataset.S, dtype=np.float32),
        A=np.asarray(dataset.A[perm], dtype=np.float32),
        meta=meta,
    ).validate()


def prepare_dataset_for_condition(dataset: DatasetBundle, condition: SweepCondition) -> DatasetBundle:
    if condition.truth_label == "alternative":
        return annotate_alternative_dataset(dataset, condition)
    if condition.truth_label == "null" and condition.null_family == "matched_shuffled":
        return matched_shuffle_dataset(dataset, seed=condition.seed, condition=condition)
    raise ValueError(
        f"Unsupported condition combination truth_label={condition.truth_label!r}, null_family={condition.null_family!r}"
    )


def manifest_path_for_spec(spec: ExistenceSigmaStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def analysis_dir_for_spec(spec: ExistenceSigmaStudySpec) -> Path:
    return spec.output_root / "analysis"


def scan_result_json_paths(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path.resolve() for path in root_path.rglob("*_result.json"))


def load_manifest_entries(path: str | Path) -> dict[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries: dict[Path, dict[str, Any]] = {}
    for entry in payload.get("runs", []):
        result_path = Path(entry["result_json_path"]).resolve()
        entries[result_path] = dict(entry)
    return entries


def load_result_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_mode_from_payload(payload: Mapping[str, Any]) -> Optional[str]:
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        data = config.get("data", {})
        if isinstance(data, Mapping):
            mode = data.get("mode")
            if isinstance(mode, str) and mode:
                return mode
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        dataset_meta = artifacts.get("dataset_meta", {})
        if isinstance(dataset_meta, Mapping):
            mode = dataset_meta.get("mode")
            if isinstance(mode, str) and mode:
                return mode
    return None


def infer_sigma_from_payload(payload: Mapping[str, Any]) -> Optional[float]:
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        data = config.get("data", {})
        if isinstance(data, Mapping) and data.get("sigma") is not None:
            return float(data["sigma"])
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        dataset_meta = artifacts.get("dataset_meta", {})
        if isinstance(dataset_meta, Mapping) and dataset_meta.get("sigma") is not None:
            return float(dataset_meta["sigma"])
    return None


def infer_k_from_payload(payload: Mapping[str, Any]) -> Optional[int]:
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        data = config.get("data", {})
        if isinstance(data, Mapping):
            if data.get("k") is not None:
                return int(data["k"])
            if data.get("k_max") is not None:
                return int(data["k_max"])
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        dataset_meta = artifacts.get("dataset_meta", {})
        if isinstance(dataset_meta, Mapping):
            if dataset_meta.get("k") is not None:
                return int(dataset_meta["k"])
            if dataset_meta.get("k_max") is not None:
                return int(dataset_meta["k_max"])
    return None


def infer_seed_from_payload(payload: Mapping[str, Any]) -> Optional[int]:
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        data = config.get("data", {})
        if isinstance(data, Mapping) and data.get("seed") is not None:
            return int(data["seed"])
    return None


def infer_truth_label_from_payload(payload: Mapping[str, Any]) -> Optional[str]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        dataset_meta = artifacts.get("dataset_meta", {})
        if isinstance(dataset_meta, Mapping):
            truth_label = dataset_meta.get("truth_label")
            if isinstance(truth_label, str) and truth_label:
                return truth_label
    mode = infer_mode_from_payload(payload)
    if mode == "noise":
        return "null"
    if mode in {"radial", "fourier", "checkerboard"}:
        return "alternative"
    return None


def infer_null_family_from_payload(payload: Mapping[str, Any]) -> Optional[str]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        dataset_meta = artifacts.get("dataset_meta", {})
        if isinstance(dataset_meta, Mapping):
            null_family = dataset_meta.get("null_family")
            if isinstance(null_family, str) and null_family:
                return null_family
    mode = infer_mode_from_payload(payload)
    if mode == "noise":
        return "mode_noise"
    return None


def infer_family_label(mode: Optional[str], k: Optional[int]) -> Optional[str]:
    if mode is None:
        return None
    if mode == "fourier":
        return f"fourier_k={int(k)}" if k is not None else "fourier_k=unknown"
    return mode


def infer_run_name_mode(run_name: str) -> Optional[str]:
    match = RUN_NAME_MODE_PATTERN.search(run_name)
    if match is None:
        return None
    return str(match.group(1))


def build_warning_record(*, warning_type: str, result_json_path: Path, run_name: str, message: str) -> dict[str, Any]:
    return {
        "warning_type": warning_type,
        "result_json_path": str(result_json_path),
        "run_name": run_name,
        "message": message,
    }


def extract_result_record(
    result_json_path: str | Path,
    *,
    manifest_entry: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(result_json_path).resolve()
    payload = load_result_payload(path)
    warnings: list[dict[str, Any]] = []

    method_name = payload.get("method_name")
    if method_name != "parallel_permutation":
        return None, warnings

    config = payload.get("config", {})
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    if not isinstance(data_cfg, Mapping) or data_cfg.get("source") != "synthetic":
        return None, warnings

    run_name = str(payload.get("config", {}).get("output", {}).get("run_name", path.stem.replace("_result", "")))
    mode = infer_mode_from_payload(payload)
    sigma = infer_sigma_from_payload(payload)
    k = infer_k_from_payload(payload)
    seed = infer_seed_from_payload(payload)
    truth_label = infer_truth_label_from_payload(payload)
    null_family = infer_null_family_from_payload(payload)

    if manifest_entry is not None:
        truth_label = str(manifest_entry["truth_label"])
        null_family = manifest_entry.get("null_family")

    mode_from_name = infer_run_name_mode(run_name)
    if mode_from_name is not None and mode is not None and mode_from_name != mode:
        warnings.append(
            build_warning_record(
                warning_type="run_name_mode_mismatch",
                result_json_path=path,
                run_name=run_name,
                message=f"Run name suggests mode '{mode_from_name}' but metadata mode is '{mode}'",
            )
        )

    if truth_label is None or mode is None or sigma is None:
        warnings.append(
            build_warning_record(
                warning_type="missing_metadata",
                result_json_path=path,
                run_name=run_name,
                message="Skipping result because mode, sigma, or truth label could not be inferred",
            )
        )
        return None, warnings

    family_label = infer_family_label(mode, k)
    record = {
        "result_json_path": str(path),
        "run_name": run_name,
        "method_name": str(method_name),
        "metric": str(payload.get("metric")),
        "mode": str(mode),
        "k": "" if k is None else int(k),
        "sigma": float(sigma),
        "seed": "" if seed is None else int(seed),
        "truth_label": str(truth_label),
        "null_family": "" if null_family is None else str(null_family),
        "family_label": "" if family_label is None else str(family_label),
        "p_value": float(payload.get("p_value")),
        "reject": int(float(payload.get("p_value")) < 0.05),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
    }
    return record, warnings


def summarize_condition_rows(rows: Iterable[Mapping[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["truth_label"]),
            str(row["mode"]),
            str(row["family_label"]),
            float(row["sigma"]),
            "" if row["k"] == "" else str(row["k"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (truth_label, mode, family_label, sigma, k), group in sorted(grouped.items()):
        rejects = np.asarray([int(item["reject"]) for item in group], dtype=np.float64)
        n_runs = int(rejects.size)
        rate = float(rejects.mean()) if n_runs else math.nan
        standard_error = float(math.sqrt(rate * max(0.0, 1.0 - rate) / max(n_runs, 1))) if n_runs else math.nan
        ci_delta = 1.96 * standard_error if n_runs else math.nan
        summaries.append(
            {
                "summary_metric": "power" if truth_label == "alternative" else "null_rejection_rate",
                "truth_label": truth_label,
                "mode": mode,
                "family_label": family_label,
                "sigma": float(sigma),
                "k": k,
                "alpha": float(alpha),
                "n_runs": n_runs,
                "n_reject": int(rejects.sum()),
                "rate": rate,
                "standard_error": standard_error,
                "ci_lower": max(0.0, rate - ci_delta) if n_runs else math.nan,
                "ci_upper": min(1.0, rate + ci_delta) if n_runs else math.nan,
            }
        )
    return summaries


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], *, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
