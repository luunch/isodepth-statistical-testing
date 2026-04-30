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

from data.schemas import DatasetBundle, RunConfig, SUPPORTED_EXISTENCE_METHODS, run_config_from_mapping
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
    if base_run_config.test.method not in SUPPORTED_EXISTENCE_METHODS:
        raise ValueError(
            "base_config must use an existence test.method in "
            f"{sorted(SUPPORTED_EXISTENCE_METHODS)}"
        )
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
                            **{**asdict(condition), "run_name": _build_run_name(spec, condition)}
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
                            **{**asdict(condition), "run_name": _build_run_name(spec, condition)}
                        )
                    )
    return conditions


def build_condition_run_config(
    base_run_config: RunConfig, spec: ExistenceSigmaStudySpec, condition: SweepCondition
) -> RunConfig:
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


def load_result_payload(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def scan_result_json_paths(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path.resolve() for path in root_path.rglob("*_result.json") if path.is_file())


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


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]], *, fieldnames: list[str]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out_path


def _record_family_label(mode: str, k_value: object) -> str:
    if str(mode) == "fourier" and k_value not in {"", None}:
        return f"fourier_k={int(k_value)}"
    return str(mode)


def extract_result_record(
    result_json_path: str | Path,
    *,
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
        data_cfg = {}
    output_cfg = config.get("output", {}) if isinstance(config, Mapping) else {}
    if not isinstance(output_cfg, Mapping):
        output_cfg = {}
    artifacts = payload.get("artifacts", {})
    dataset_meta = artifacts.get("dataset_meta", {}) if isinstance(artifacts, Mapping) else {}
    if not isinstance(dataset_meta, Mapping):
        dataset_meta = {}

    mode = str(data_cfg.get("mode", dataset_meta.get("generating_mode", dataset_meta.get("mode", ""))))
    sigma = float(data_cfg.get("sigma", dataset_meta.get("sigma", 0.0)))
    seed = int(data_cfg.get("seed", dataset_meta.get("seed", -1)))
    k_value = data_cfg.get("k", dataset_meta.get("k"))
    if k_value in {None, ""}:
        k_value = data_cfg.get("k_max", dataset_meta.get("k_max", ""))
    if k_value in {None, ""}:
        k_value = ""
    else:
        k_value = int(k_value)

    run_name = str(output_cfg.get("run_name", path.stem.replace("_result", "")))
    truth_label = str(dataset_meta.get("truth_label", manifest_entry.get("truth_label") if manifest_entry else "null"))
    null_family = dataset_meta.get("null_family", manifest_entry.get("null_family") if manifest_entry else None)
    if null_family in {None, ""}:
        null_family = ""

    mode_match = RUN_NAME_MODE_PATTERN.search(run_name)
    if mode_match is not None and mode and mode_match.group(1) != mode:
        warnings.append(
            {
                "warning_type": "run_name_mode_mismatch",
                "result_json_path": str(path),
                "run_name": run_name,
                "message": f"Run name implies mode '{mode_match.group(1)}' but payload reports '{mode}'",
            }
        )

    record = {
        "result_json_path": str(path),
        "run_name": run_name,
        "method_name": str(payload.get("method_name")),
        "metric": str(payload.get("metric")),
        "mode": mode,
        "k": k_value,
        "sigma": sigma,
        "seed": seed,
        "truth_label": truth_label,
        "null_family": null_family,
        "family_label": _record_family_label(mode, k_value),
        "p_value": float(payload.get("p_value")),
        "stat_true": float(payload.get("stat_true")),
        "runtime_sec": float(payload.get("runtime_sec")),
    }
    return record, warnings


def summarize_condition_rows(rows: Iterable[Mapping[str, object]], *, alpha: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float, object], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row["truth_label"]),
            str(row["mode"]),
            str(row["family_label"]),
            float(row["sigma"]),
            row.get("k", ""),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (truth_label, mode, family_label, sigma, k_value), group_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][3], "" if item[0][4] == "" else int(item[0][4])),
    ):
        rejects = np.asarray([int(row.get("reject", 0)) for row in group_rows], dtype=np.float64)
        n_runs = int(rejects.size)
        rate = float(np.mean(rejects))
        standard_error = float(math.sqrt(max(rate * (1.0 - rate) / max(n_runs, 1), 0.0)))
        ci_half_width = 1.96 * standard_error
        summaries.append(
            {
                "summary_metric": "power" if truth_label == "alternative" else "null_rejection_rate",
                "truth_label": truth_label,
                "mode": mode,
                "family_label": family_label,
                "sigma": sigma,
                "k": k_value,
                "alpha": float(alpha),
                "n_runs": n_runs,
                "n_reject": int(np.sum(rejects)),
                "rate": rate,
                "standard_error": standard_error,
                "ci_lower": max(0.0, rate - ci_half_width),
                "ci_upper": min(1.0, rate + ci_half_width),
            }
        )
    return summaries
