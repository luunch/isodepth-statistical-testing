from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from analysis.plots import save_combined_celltype_metric_distribution, save_metric_distribution_plot
from data.schemas import RunConfig, SUPPORTED_EXISTENCE_METHODS, TestResult, run_config_from_mapping
from data.transforms import zscore_covariate
from experiments.configuration import build_run_config
from experiments.core.paths import repo_root
from experiments.core.study_io import load_result_payload, scan_result_json_paths, write_csv
from experiments.studies.random_gene_panel_null.lib import _surviving_gene_count_from_result
from methods.metrics import canonicalize_metric_name, compute_metric, permutation_p_value
from methods.trainers.isodepth import (
    fit_closed_form_decoder,
    fit_poisson_glm_irls,
    train_fixed_covariate_model,
)

REPO_ROOT = repo_root(__file__)
HVG3000_COVARIATE_KEY = "hvg3000_isodepth"


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
    hvg3000_config: Optional[Path] = None
    hvg3000_result_json: Optional[Path] = None
    covariate_from_hvg3000: bool = False
    covariate_obs_key: str = HVG3000_COVARIATE_KEY

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
        self.covariate_from_hvg3000 = bool(self.covariate_from_hvg3000)
        self.covariate_obs_key = str(self.covariate_obs_key or HVG3000_COVARIATE_KEY).strip()
        if not self.covariate_obs_key:
            raise ValueError("covariate_obs_key must be a non-empty string")
        if self.hvg3000_config is not None and not self.hvg3000_config.exists():
            raise ValueError(f"hvg3000_config does not exist: {self.hvg3000_config}")
        if self.hvg3000_result_json is not None and not self.hvg3000_result_json.exists():
            raise ValueError(f"hvg3000_result_json does not exist: {self.hvg3000_result_json}")
        if self.covariate_from_hvg3000 and self.hvg3000_config is None and self.hvg3000_result_json is None:
            raise ValueError(
                "covariate_from_hvg3000 requires hvg3000_config or hvg3000_result_json"
            )

        self.base_config = self.base_config.resolve()
        self.output_root = self.output_root.resolve()
        self.gmt_path = self.gmt_path.resolve()
        if self.hvg3000_config is not None:
            self.hvg3000_config = self.hvg3000_config.resolve()
        if self.hvg3000_result_json is not None:
            self.hvg3000_result_json = self.hvg3000_result_json.resolve()
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

    hvg3000_config = payload.get("hvg3000_config")
    hvg3000_result_json = payload.get("hvg3000_result_json")
    spec = PathwayPanelSweepSpec(
        experiment_name=str(payload["experiment_name"]),
        base_config=_resolve_repo_path(payload["base_config"]),
        output_root=_resolve_repo_path(payload["output_root"]),
        gmt_path=_resolve_repo_path(payload["gmt_path"]),
        n_perms=int(payload.get("n_perms", 249)),
        n_reruns=int(payload.get("n_reruns", 30)),
        alpha=float(payload.get("alpha", 0.05)),
        min_requested_genes=int(payload.get("min_requested_genes", 15)),
        hvg3000_config=_resolve_repo_path(hvg3000_config) if hvg3000_config else None,
        hvg3000_result_json=_resolve_repo_path(hvg3000_result_json) if hvg3000_result_json else None,
        covariate_from_hvg3000=bool(payload.get("covariate_from_hvg3000", False)),
        covariate_obs_key=str(payload.get("covariate_obs_key", HVG3000_COVARIATE_KEY)),
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
    if spec.covariate_from_hvg3000:
        mapping["test"]["covariate"] = {"type": spec.covariate_obs_key}
    mapping["output"]["out_dir"] = str(spec.output_root / "runs")
    mapping["output"]["run_name"] = condition.run_name
    return run_config_from_mapping(mapping)


def result_json_path_for_run_config(run_config: RunConfig) -> Path:
    out_dir = Path(run_config.output.out_dir) / run_config.output.run_name
    return out_dir / f"{run_config.output.run_name}_result.json"


def hvg3000_result_json_for_spec(spec: PathwayPanelSweepSpec) -> Path:
    if spec.hvg3000_result_json is not None:
        return spec.hvg3000_result_json
    if spec.hvg3000_config is None:
        raise ValueError("hvg3000_config or hvg3000_result_json is required")
    return result_json_path_for_run_config(build_run_config(str(spec.hvg3000_config), {}))


def load_reference_isodepth(result_json_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``true_isodepth`` and per-type ``S`` from a saved existence-test run."""
    result_json_path = Path(result_json_path)
    npz_candidates = sorted(result_json_path.parent.rglob("*_isodepths.npz"))
    if not npz_candidates:
        raise FileNotFoundError(f"No *_isodepths.npz under {result_json_path.parent}")
    npz = np.load(npz_candidates[0], allow_pickle=False)
    if "true_isodepth" not in npz or "S" not in npz:
        raise KeyError(f"Missing true_isodepth/S in {npz_candidates[0]}")
    iso = np.asarray(npz["true_isodepth"], dtype=np.float32).reshape(-1)
    spatial = np.asarray(npz["S"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[0] != iso.shape[0]:
        raise ValueError(
            f"Bad S/isodepth shapes in {npz_candidates[0]}: S={spatial.shape}, iso={iso.shape}"
        )
    return iso, spatial


def _per_type_zscore_coordinates(S: np.ndarray) -> np.ndarray:
    coords = np.asarray(S, dtype=np.float64)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return np.asarray((coords - mean) / safe_std, dtype=np.float32)


def inject_reference_isodepth_covariate(
    dataset,
    reference_isodepth: np.ndarray,
    reference_S: np.ndarray,
    *,
    obs_key: str = HVG3000_COVARIATE_KEY,
) -> None:
    """Attach a fixed HVG3000 isodepth as ``dataset.meta['covariate_values']``.

    The reference NPZ stores the per-type re-standardized coordinates used in
    ``_process_single_celltype_separate``. These sweeps restrict to one clone, so
    the same transform of ``dataset.S`` must match ``reference_S`` cell-for-cell.
    """
    iso = np.asarray(reference_isodepth, dtype=np.float32).reshape(-1)
    ref_S = np.asarray(reference_S, dtype=np.float32)
    if iso.shape[0] != int(dataset.n_cells):
        raise ValueError(
            f"HVG3000 isodepth length {iso.shape[0]} != dataset n_cells {dataset.n_cells}"
        )
    names = list(dataset.meta.get("cell_type_names") or [])
    if names and len(names) != 1:
        raise ValueError(
            "HVG3000 isodepth covariate inject expects a single cell type; "
            f"got {names}"
        )
    aligned_S = _per_type_zscore_coordinates(dataset.S)
    if aligned_S.shape != ref_S.shape or not np.allclose(aligned_S, ref_S, atol=1e-4):
        raise ValueError(
            "HVG3000 reference S does not match the pathway-run cells after the "
            "same per-type coordinate z-score; refusing to inject a misaligned covariate."
        )
    dataset.meta["covariate_values"] = iso
    dataset.meta["covariate_obs_key"] = obs_key


def run_hvg3000_reference(spec: PathwayPanelSweepSpec) -> Path:
    """Train the 3000-HVG isodepth (true model + pipeline-minimum 1 perm) if missing."""
    if spec.hvg3000_config is None and spec.hvg3000_result_json is not None:
        return spec.hvg3000_result_json
    if spec.hvg3000_config is None:
        raise ValueError("hvg3000_config is required to train the HVG3000 reference")

    run_config = build_run_config(str(spec.hvg3000_config), {})
    run_config.test.n_reruns = int(spec.n_reruns)
    result_path = result_json_path_for_run_config(run_config)
    if result_path.exists():
        print(f"HVG3000 reference already exists: {result_path}", flush=True)
        return result_path

    from data import load_dataset
    from experiments.configuration import save_standardized_outputs
    from methods.permutation import run_permutation_method

    print(
        f"Training HVG3000 isodepth ({run_config.output.run_name}, "
        f"n_reruns={run_config.test.n_reruns}, n_perms={run_config.test.n_perms})",
        flush=True,
    )
    dataset = load_dataset(run_config.data, covariate=run_config.test.covariate)
    result = run_permutation_method(dataset, run_config.test)
    _, result_path = save_standardized_outputs(dataset, result, run_config)
    print(f"Saved HVG3000 reference to: {result_path}", flush=True)
    return Path(result_path)


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
        "stat_covariate": float("nan"),
        "p_value_covariate": float("nan"),
    }
    artifacts = payload.get("artifacts", {})
    summaries = artifacts.get("per_type_summaries", {}) if isinstance(artifacts, Mapping) else {}
    if isinstance(summaries, Mapping) and summaries:
        stat_covs = []
        p_covs = []
        for summary in summaries.values():
            if not isinstance(summary, Mapping):
                continue
            if summary.get("stat_covariate") is not None:
                stat_covs.append(float(summary["stat_covariate"]))
            if summary.get("p_value_covariate") is not None:
                p_covs.append(float(summary["p_value_covariate"]))
        if stat_covs:
            record["stat_covariate"] = float(np.mean(stat_covs))
        if p_covs:
            record["p_value_covariate"] = float(np.mean(p_covs))
    return record, warnings


def apply_all_genes_covariate_baseline(
    result_json_path: str | Path,
    reference_isodepth: np.ndarray,
    reference_S: np.ndarray,
) -> Optional[dict[str, Any]]:
    """Refit the fixed all-genes isodepth (treated as a covariate) to one pathway
    run's own gene panel, and add the resulting loss as a third line on that run's
    per-group and combined ``*_metric_distribution.png`` null-distribution plots.

    This mirrors the existing ``stat_covariate``/``p_value_covariate`` dual-line
    convention used for ``test.covariate`` runs
    (``analysis/plots.py::save_metric_distribution_plot`` /
    ``save_combined_celltype_metric_distribution``) — except the covariate latent here
    is the isodepth trained on ALL genes for the same cells (loaded from
    ``reference_full_result_json``), fit via closed-form OLS when the run's own
    decoder is ``linear``/``quadratic`` (gradient descent for ``nn``), evaluated
    against the run's own preprocessed expression matrix (post covariate-whitening,
    i.e. the exact array the isodepth model was trained on).

    Persists the covariate stat back into the run's own result JSON (so it survives
    across re-analysis) and rewrites its PNGs in place. Only supports
    ``cell_type_mode == "separate"`` results (what the pathway panel sweep always
    produces); returns ``None`` (no-op) otherwise, or if no group's saved
    ``*_isodepths.npz`` has matching coordinates to ``reference_S``.
    """
    result_json_path = Path(result_json_path)
    with open(result_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("method_name") not in SUPPORTED_EXISTENCE_METHODS:
        return None

    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict) or artifacts.get("cell_type_mode") != "separate":
        return None
    cell_type_names = list(artifacts.get("cell_type_names", []))
    per_type_summaries = dict(artifacts.get("per_type_summaries", {}))
    if not cell_type_names or not per_type_summaries:
        return None

    config = payload.get("config", {})
    test_cfg = config.get("test", {}) if isinstance(config, Mapping) else {}
    if not isinstance(test_cfg, Mapping):
        test_cfg = {}
    metric = str(payload.get("metric", test_cfg.get("metric", "nll_gaussian_mse")))
    decoder_type = str(test_cfg.get("decoder", "nn"))
    is_poisson = canonicalize_metric_name(metric) == "nll_poisson_mse"

    run_dir = result_json_path.parent
    run_name = run_dir.name

    ref_iso = np.asarray(reference_isodepth, dtype=np.float64).reshape(-1)
    ref_S = np.asarray(reference_S, dtype=np.float64)
    latent = zscore_covariate(ref_iso)

    per_type_results: dict[str, dict[str, Any]] = {}
    for type_name in cell_type_names:
        summary = per_type_summaries.get(type_name)
        if not isinstance(summary, dict):
            continue
        safe_name = type_name.replace(" ", "_").replace("/", "_")
        group_dir = run_dir / safe_name
        npz_candidates = sorted(group_dir.glob("*_isodepths.npz"))
        if not npz_candidates:
            continue
        npz = np.load(npz_candidates[0], allow_pickle=False)
        if "A" not in npz or "S" not in npz:
            continue
        S_pw = np.asarray(npz["S"], dtype=np.float64)
        A_pw = np.asarray(npz["A"], dtype=np.float64)
        if S_pw.shape[0] != ref_S.shape[0] or not np.allclose(S_pw, ref_S, atol=1e-5):
            continue

        stat_perm = np.asarray(summary["stat_perm"], dtype=np.float64)
        if decoder_type in ("linear", "quadratic"):
            if is_poisson:
                pred_cov = fit_poisson_glm_irls(latent, A_pw, decoder_type)
            else:
                pred_cov = fit_closed_form_decoder(latent, A_pw, decoder_type)
        else:
            run_cfg = run_config_from_mapping(config)
            _, pred_cov = train_fixed_covariate_model(
                ref_iso,
                A_pw.astype(np.float32),
                run_cfg.test,
                device=torch.device("cpu"),
                model_label=f"{run_name}/{type_name} all-genes covariate",
            )
        stat_cov = float(compute_metric(metric, A_pw, pred_cov))
        p_cov = float(permutation_p_value(metric, stat_cov, stat_perm))

        summary["stat_covariate"] = stat_cov
        summary["p_value_covariate"] = p_cov
        per_type_summaries[type_name] = summary

        per_type_results[type_name] = {
            "stat_perm": stat_perm,
            "stat_true": float(summary["stat_true"]),
            "p_value": float(summary["p_value"]),
            "n_cells": int(summary["n_cells"]),
            "stat_covariate": stat_cov,
            "p_value_covariate": p_cov,
        }

        subset_result = TestResult(
            method_name="parallel_permutation",
            metric=metric,
            p_value=float(summary["p_value"]),
            stat_true=float(summary["stat_true"]),
            stat_perm=stat_perm,
            runtime_sec=0.0,
            n_cells=int(summary["n_cells"]),
            n_genes=int(A_pw.shape[1]),
            config={},
            artifacts={
                "stat_covariate": stat_cov,
                "p_value_covariate": p_cov,
                "covariate_label": "All-genes isodepth (OLS)",
            },
        ).validate()
        plot_path = save_metric_distribution_plot(
            subset_result, group_dir / f"{safe_name}_metric_distribution.png"
        )
        apaths = dict(summary.get("artifact_paths", {}))
        apaths["metric_distribution_plot"] = str(plot_path)
        summary["artifact_paths"] = apaths

    if not per_type_results:
        return None

    combined_names = [name for name in cell_type_names if name in per_type_results]
    combined_path = save_combined_celltype_metric_distribution(
        per_type_results,
        combined_names,
        run_dir / f"{run_name}_combined_metric_distribution.png",
        metric=metric,
        covariate_label="All-genes isodepth (OLS)",
    )

    artifacts["per_type_summaries"] = per_type_summaries
    artifacts["combined_metric_distribution_plot"] = str(combined_path)
    artifacts["all_genes_covariate_applied"] = True
    payload["artifacts"] = artifacts

    with open(result_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "result_json_path": str(result_json_path),
        "n_groups_updated": len(per_type_results),
        "per_type": {
            name: {
                "stat_covariate": data["stat_covariate"],
                "p_value_covariate": data["p_value_covariate"],
            }
            for name, data in per_type_results.items()
        },
    }
