"""Synthetic kernel-noise study: coordinate vs block permutation across kernel grid.

For each (kernel distance, delta, data seed):
  1. Generate and cache one synthetic dataset (``datasets/{key}.npz``).
  2. Run ``parallel_permutation`` on the cached dataset.
  3. Run ``block_permutation`` at each configured block radius on the same cache.

Resume: skips existing result JSONs and cached datasets.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from data import generate_synthetic_dataset
from data.schemas import DataConfig, DatasetBundle, RunConfig, run_config_from_mapping
from experiments.configuration import build_manifest_config_snapshot, save_standardized_outputs

REPO_ROOT = Path(__file__).resolve().parent.parent
TestMethod = Literal["parallel_permutation", "block_permutation"]


@dataclass
class KernelNoiseStudySpec:
    experiment_name: str
    existence_base_config: Path
    block_base_config: Path
    output_root: Path
    alpha: float
    kernel_distances_um: list[float]
    deltas: list[float]
    seeds: list[int]
    block_radii_um: list[float]
    coordinate_um_per_unit: float

    def validate(self) -> "KernelNoiseStudySpec":
        if not self.existence_base_config.exists():
            raise ValueError(f"existence_base_config missing: {self.existence_base_config}")
        if not self.block_base_config.exists():
            raise ValueError(f"block_base_config missing: {self.block_base_config}")
        if not self.kernel_distances_um:
            raise ValueError("kernel_distances_um must be non-empty")
        if not self.deltas:
            raise ValueError("deltas must be non-empty")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.block_radii_um:
            raise ValueError("block_radii_um must be non-empty")
        if float(self.coordinate_um_per_unit) <= 0:
            raise ValueError("coordinate_um_per_unit must be > 0")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.existence_base_config = self.existence_base_config.resolve()
        self.block_base_config = self.block_base_config.resolve()
        self.output_root = self.output_root.resolve()
        return self


@dataclass(frozen=True)
class DatasetKey:
    kernel_distance_um: float
    delta: float
    data_seed: int

    @property
    def slug(self) -> str:
        return (
            f"d{_format_number(self.kernel_distance_um)}"
            f"_delta{_format_number(self.delta)}"
            f"_seed{int(self.data_seed)}"
        )


@dataclass(frozen=True)
class KernelNoiseRunCondition:
    dataset_key: DatasetKey
    test_method: TestMethod
    block_radius_um: float | None
    run_name: str


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _format_number(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def load_kernel_noise_study_spec(path: str | Path) -> KernelNoiseStudySpec:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    spec = KernelNoiseStudySpec(
        experiment_name=str(payload["experiment_name"]),
        existence_base_config=_resolve_repo_path(payload["existence_base_config"]),
        block_base_config=_resolve_repo_path(payload["block_base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        alpha=float(payload.get("alpha", 0.05)),
        kernel_distances_um=[float(v) for v in payload["kernel_distances_um"]],
        deltas=[float(v) for v in payload["deltas"]],
        seeds=[int(v) for v in payload["seeds"]],
        block_radii_um=[float(v) for v in payload["block_radii_um"]],
        coordinate_um_per_unit=float(payload["coordinate_um_per_unit"]),
    )
    return spec.validate()


def analysis_dir_for_spec(spec: KernelNoiseStudySpec) -> Path:
    return spec.output_root / "analysis"


def datasets_dir_for_spec(spec: KernelNoiseStudySpec) -> Path:
    return spec.output_root / "datasets"


def runs_dir_for_spec(spec: KernelNoiseStudySpec) -> Path:
    return spec.output_root / "runs"


def manifest_path_for_spec(spec: KernelNoiseStudySpec) -> Path:
    return spec.output_root / "manifest.json"


def expand_kernel_noise_conditions(spec: KernelNoiseStudySpec) -> list[KernelNoiseRunCondition]:
    conditions: list[KernelNoiseRunCondition] = []
    for distance in spec.kernel_distances_um:
        for delta in spec.deltas:
            for seed in spec.seeds:
                key = DatasetKey(kernel_distance_um=float(distance), delta=float(delta), data_seed=int(seed))
                conditions.append(
                    KernelNoiseRunCondition(
                        dataset_key=key,
                        test_method="parallel_permutation",
                        block_radius_um=None,
                        run_name=f"{key.slug}_coord",
                    )
                )
                for radius in spec.block_radii_um:
                    r_slug = _format_number(float(radius))
                    conditions.append(
                        KernelNoiseRunCondition(
                            dataset_key=key,
                            test_method="block_permutation",
                            block_radius_um=float(radius),
                            run_name=f"{key.slug}_block_r{r_slug}",
                        )
                    )
    return conditions


def _jsonify_meta(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify_meta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_meta(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_dataset_cache(path: Path, dataset: DatasetBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(_jsonify_meta(dataset.meta))
    np.savez(path, S=dataset.S.astype(np.float32), A=dataset.A.astype(np.float32), meta_json=np.asarray([meta_json]))


def load_dataset_cache(path: Path) -> DatasetBundle:
    payload = np.load(path, allow_pickle=False)
    meta = json.loads(str(payload["meta_json"][0]))
    return DatasetBundle(S=np.asarray(payload["S"], dtype=np.float32), A=np.asarray(payload["A"], dtype=np.float32), meta=meta).validate()


def dataset_cache_path(spec: KernelNoiseStudySpec, key: DatasetKey) -> Path:
    return datasets_dir_for_spec(spec) / f"{key.slug}.npz"


def _load_base_mapping(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_data_config(
    base_mapping: dict[str, Any],
    *,
    key: DatasetKey,
) -> DataConfig:
    data_payload = copy.deepcopy(base_mapping["data"])
    data_payload["seed"] = int(key.data_seed)
    data_payload["delta"] = float(key.delta)
    kernel_payload = dict(data_payload.get("kernel") or {"type": "exp"})
    kernel_payload["distance"] = float(key.kernel_distance_um)
    data_payload["kernel"] = kernel_payload
    return DataConfig(**data_payload).validate()


def _build_run_config_for_condition(
    spec: KernelNoiseStudySpec,
    condition: KernelNoiseRunCondition,
    *,
    existence_mapping: dict[str, Any],
    block_mapping: dict[str, Any],
) -> RunConfig:
    base_mapping = existence_mapping if condition.test_method == "parallel_permutation" else block_mapping
    merged = copy.deepcopy(base_mapping)
    merged.setdefault("output", {})
    merged["output"]["out_dir"] = str(runs_dir_for_spec(spec))
    merged["output"]["run_name"] = condition.run_name
    merged["data"] = _build_data_config(base_mapping, key=condition.dataset_key).to_dict()
    merged["test"] = copy.deepcopy(base_mapping["test"])
    merged["test"]["method"] = condition.test_method
    merged["test"]["seed"] = int(condition.dataset_key.data_seed)
    if condition.test_method == "block_permutation":
        merged["test"]["block_radius"] = float(condition.block_radius_um)
        merged["test"]["coordinate_um_per_unit"] = float(spec.coordinate_um_per_unit)
        merged["test"]["block_jitter"] = bool(merged["test"].get("block_jitter", True))
    return run_config_from_mapping(merged).validate()


def _result_json_path(spec: KernelNoiseStudySpec, run_name: str) -> Path:
    return runs_dir_for_spec(spec) / run_name / f"{run_name}_result.json"


def ensure_dataset_cached(
    spec: KernelNoiseStudySpec,
    key: DatasetKey,
    *,
    existence_mapping: dict[str, Any],
) -> Path:
    cache_path = dataset_cache_path(spec, key)
    if cache_path.exists():
        return cache_path
    data_cfg = _build_data_config(existence_mapping, key=key)
    dataset = generate_synthetic_dataset(data_cfg)
    save_dataset_cache(cache_path, dataset)
    print(f"  cached dataset: {cache_path}", flush=True)
    return cache_path


def load_cached_dataset(cache_path: Path) -> DatasetBundle:
    return load_dataset_cache(cache_path)


def run_kernel_noise_study(
    spec: KernelNoiseStudySpec,
    *,
    spec_path: str | None = None,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, object]:
    existence_mapping = _load_base_mapping(spec.existence_base_config)
    block_mapping = _load_base_mapping(spec.block_base_config)
    conditions = expand_kernel_noise_conditions(spec)
    if max_runs is not None:
        conditions = conditions[:max_runs]

    unique_keys = sorted(
        {condition.dataset_key for condition in conditions},
        key=lambda key: (key.kernel_distance_um, key.delta, key.data_seed),
    )

    manifest_payload: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "existence_base_config": str(spec.existence_base_config),
        "block_base_config": str(spec.block_base_config),
        "output_root": str(spec.output_root),
        "analysis_dir": str(analysis_dir_for_spec(spec)),
        "datasets_dir": str(datasets_dir_for_spec(spec)),
        "alpha": float(spec.alpha),
        "kernel_distances_um": list(spec.kernel_distances_um),
        "deltas": list(spec.deltas),
        "seeds": list(spec.seeds),
        "block_radii_um": list(spec.block_radii_um),
        "coordinate_um_per_unit": float(spec.coordinate_um_per_unit),
        "unique_dataset_count": len(unique_keys),
        "planned_run_count": len(conditions),
        "runs": [],
    }
    if spec_path is not None:
        manifest_payload["config_snapshot"] = build_manifest_config_snapshot(
            spec_path,
            {
                "existence_base_config": spec.existence_base_config,
                "block_base_config": spec.block_base_config,
            },
        )

    if dry_run:
        manifest_payload["planned_datasets_preview"] = [key.slug for key in unique_keys[:10]]
        manifest_payload["planned_runs_preview"] = [
            {
                "run_name": condition.run_name,
                "dataset_key": condition.dataset_key.slug,
                "test_method": condition.test_method,
                "block_radius_um": condition.block_radius_um,
            }
            for condition in conditions[:20]
        ]
        return manifest_payload

    spec.output_root.mkdir(parents=True, exist_ok=True)
    datasets_dir_for_spec(spec).mkdir(parents=True, exist_ok=True)
    runs_dir_for_spec(spec).mkdir(parents=True, exist_ok=True)
    analysis_dir_for_spec(spec).mkdir(parents=True, exist_ok=True)

    from methods.permutation import run_permutation_method

    cached_datasets: dict[str, DatasetBundle] = {}
    for key in unique_keys:
        cache_path = ensure_dataset_cached(spec, key, existence_mapping=existence_mapping)
        cached_datasets[key.slug] = load_cached_dataset(cache_path)

    for index, condition in enumerate(conditions, start=1):
        result_path = _result_json_path(spec, condition.run_name)
        if result_path.exists():
            print(f"[{index}/{len(conditions)}] skip existing {condition.run_name}", flush=True)
            manifest_payload["runs"].append(
                {
                    "run_name": condition.run_name,
                    "dataset_key": condition.dataset_key.slug,
                    "dataset_cache_path": str(dataset_cache_path(spec, condition.dataset_key)),
                    "kernel_distance_um": float(condition.dataset_key.kernel_distance_um),
                    "delta": float(condition.dataset_key.delta),
                    "data_seed": int(condition.dataset_key.data_seed),
                    "test_method": condition.test_method,
                    "block_radius_um": condition.block_radius_um,
                    "result_json_path": str(result_path.resolve()),
                    "skipped": True,
                }
            )
            continue

        print(f"[{index}/{len(conditions)}] {condition.run_name}", flush=True)
        run_config = _build_run_config_for_condition(
            spec,
            condition,
            existence_mapping=existence_mapping,
            block_mapping=block_mapping,
        )
        dataset = cached_datasets[condition.dataset_key.slug]
        result = run_permutation_method(dataset, run_config.test)
        _, saved_path = save_standardized_outputs(dataset, result, run_config)
        manifest_payload["runs"].append(
            {
                "run_name": condition.run_name,
                "dataset_key": condition.dataset_key.slug,
                "dataset_cache_path": str(dataset_cache_path(spec, condition.dataset_key)),
                "kernel_distance_um": float(condition.dataset_key.kernel_distance_um),
                "delta": float(condition.dataset_key.delta),
                "data_seed": int(condition.dataset_key.data_seed),
                "test_method": condition.test_method,
                "block_radius_um": condition.block_radius_um,
                "p_value": float(result.p_value),
                "stat_true": float(result.stat_true),
                "result_json_path": str(saved_path.resolve()),
                "skipped": False,
            }
        )

    manifest_path = manifest_path_for_spec(spec)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    print(f"Saved manifest to: {manifest_path}", flush=True)
    return manifest_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic kernel-noise coordinate vs block study.")
    parser.add_argument("--spec", required=True, help="Path to configs/experiments/kernel_noise_study.json")
    parser.add_argument("--dry-run", action="store_true", help="Print planned datasets/runs without executing")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of test runs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    spec = load_kernel_noise_study_spec(args.spec)
    payload = run_kernel_noise_study(
        spec,
        spec_path=args.spec,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
