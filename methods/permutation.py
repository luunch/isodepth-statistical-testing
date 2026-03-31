from __future__ import annotations

import time

import numpy as np
import torch

from data.schemas import DatasetBundle, TestConfig, TestResult
from methods.metrics import (
    canonicalize_metric_name,
    compute_metric,
    compute_metric_batch,
    metric_prefers_lower,
    permutation_p_value,
)
from methods.perturbation import run_perturbation_robustness_method
from methods.trainers import (
    resolve_device,
    train_gaston_mix_model,
    train_frozen_decoder_model,
    train_isodepth_model,
    train_parallel_isodepth_model,
)


def _extract_isodepth_from_model(model, S: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        s_t = torch.tensor(S, dtype=torch.float32, device=device)
        d = model.encoder(s_t).detach().cpu().numpy().reshape(-1)
    return np.asarray(d, dtype=np.float32)


def _extract_gaston_mix_isodepth(gates: np.ndarray, isodepths: list[np.ndarray]) -> np.ndarray:
    combined = np.zeros(gates.shape[0], dtype=np.float32)
    for p in range(gates.shape[1]):
        combined += gates[:, p].astype(np.float32) * np.asarray(isodepths[p], dtype=np.float32).reshape(-1)
    return combined


def _select_extreme_index(metric: str, stat_perm: np.ndarray) -> int:
    if metric_prefers_lower(metric):
        return int(np.argmin(stat_perm))
    return int(np.argmax(stat_perm))


def _select_low_high_indices(stat_perm: np.ndarray) -> tuple[int, int]:
    return int(np.argmin(stat_perm)), int(np.argmax(stat_perm))


def run_parallel_permutation_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    model, predictions = train_parallel_isodepth_model(dataset.S, dataset.A, config, device=device)
    stats = compute_metric_batch(metric, dataset.A, predictions)
    stat_true = float(stats[0])
    stat_perm = np.asarray(stats[1:], dtype=np.float64)
    low_idx, high_idx = _select_low_high_indices(stat_perm)
    n_models = config.n_perms + 1
    s_t = torch.tensor(dataset.S, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    s_batched = torch.zeros((n_models, dataset.n_cells, 2), dtype=torch.float32, device=device)
    s_batched[0] = s_t
    perm_indices = []
    for m in range(1, n_models):
        perm = torch.randperm(dataset.n_cells, generator=generator)
        perm_indices.append(perm.cpu().numpy())
        s_batched[m] = s_t[perm.to(device=device)]
    with torch.no_grad():
        isodepth_batched = model.encoder(s_batched).detach().cpu().numpy().squeeze(-1)
    runtime_sec = time.time() - start

    return TestResult(
        method_name="parallel_permutation",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=stat_true,
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": model,
            "pred_true": np.asarray(predictions[0], dtype=np.float32),
            "true_isodepth": np.asarray(isodepth_batched[0], dtype=np.float32),
            "lowest_isodepth": np.asarray(isodepth_batched[low_idx + 1], dtype=np.float32),
            "lowest_S": np.asarray(s_batched[low_idx + 1].detach().cpu().numpy(), dtype=np.float32),
            "lowest_stat": float(stat_perm[low_idx]),
            "lowest_perm_index": low_idx,
            "highest_isodepth": np.asarray(isodepth_batched[high_idx + 1], dtype=np.float32),
            "highest_S": np.asarray(s_batched[high_idx + 1].detach().cpu().numpy(), dtype=np.float32),
            "highest_stat": float(stat_perm[high_idx]),
            "highest_perm_index": high_idx,
        },
    ).validate()


def run_gaston_mix_closed_form_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    model, _, gates, isodepths = train_gaston_mix_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        model_label="true model",
    )
    s_t = torch.tensor(dataset.S, dtype=torch.float32, device=device)
    a_t = torch.tensor(dataset.A, dtype=torch.float32, device=device)
    a_centered = a_t - a_t.mean(dim=0, keepdim=True)

    with torch.no_grad():
        X_cols = [torch.ones((s_t.shape[0], 1), device=device)]
        for p in range(config.n_experts):
            g_p = torch.tensor(gates[:, p : p + 1], dtype=torch.float32, device=device)
            d_p = torch.tensor(isodepths[p], dtype=torch.float32, device=device)
            X_cols.append(g_p * d_p)
            X_cols.append(g_p)
        X = torch.cat(X_cols, dim=1)
        Q, _ = torch.linalg.qr(X)
        proj_true = Q @ (Q.T @ a_centered)

    proj_true_np = proj_true.detach().cpu().numpy().astype(np.float32)
    a_centered_np = a_centered.detach().cpu().numpy().astype(np.float32)
    stat_true = compute_metric(metric, a_centered_np, proj_true_np)

    stat_perm = np.zeros(config.n_perms, dtype=np.float64)
    rng = np.random.default_rng(config.seed)
    for m in range(config.n_perms):
        perm_idx = torch.tensor(rng.permutation(dataset.n_cells), dtype=torch.long, device=device)
        a_perm = a_centered[perm_idx]
        proj_perm = Q @ (Q.T @ a_perm)
        stat_perm[m] = compute_metric(
            metric,
            a_perm.detach().cpu().numpy().astype(np.float32),
            proj_perm.detach().cpu().numpy().astype(np.float32),
        )

    runtime_sec = time.time() - start
    combined_isodepth = _extract_gaston_mix_isodepth(gates, isodepths)
    low_idx, high_idx = _select_low_high_indices(stat_perm)
    return TestResult(
        method_name="gaston_mix_closed_form",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=float(stat_true),
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": model,
            "pred_true": proj_true_np,
            "gates": gates,
            "isodepths": isodepths,
            "true_isodepth": combined_isodepth,
            "lowest_isodepth": combined_isodepth.copy(),
            "lowest_S": np.asarray(dataset.S, dtype=np.float32),
            "lowest_stat": float(stat_perm[low_idx]),
            "lowest_perm_index": low_idx,
            "highest_isodepth": combined_isodepth.copy(),
            "highest_S": np.asarray(dataset.S, dtype=np.float32),
            "highest_stat": float(stat_perm[high_idx]),
            "highest_perm_index": high_idx,
        },
    ).validate()


def run_full_retraining_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    true_model, pred_true = train_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_offset=0,
        model_label="true model",
    )
    stat_true = compute_metric(metric, dataset.A, pred_true)
    true_isodepth = _extract_isodepth_from_model(true_model, dataset.S, device)

    rng = np.random.default_rng(config.seed)
    stat_perm = np.zeros(config.n_perms, dtype=np.float64)
    lowest_stat = None
    lowest_isodepth = None
    lowest_S = None
    highest_stat = None
    highest_isodepth = None
    highest_S = None
    for i in range(config.n_perms):
        perm = rng.permutation(dataset.n_cells)
        s_perm = dataset.S[perm]
        model_perm, pred_perm = train_isodepth_model(
            s_perm,
            dataset.A,
            config,
            device=device,
            seed_offset=i + 1,
            model_label=f"permuted model {i + 1}/{config.n_perms}",
        )
        stat_perm[i] = compute_metric(metric, dataset.A, pred_perm)
        current_isodepth = _extract_isodepth_from_model(model_perm, s_perm, device)
        if lowest_stat is None or stat_perm[i] < lowest_stat:
            lowest_stat = float(stat_perm[i])
            lowest_isodepth = current_isodepth
            lowest_S = np.asarray(s_perm, dtype=np.float32)
        if highest_stat is None or stat_perm[i] > highest_stat:
            highest_stat = float(stat_perm[i])
            highest_isodepth = current_isodepth
            highest_S = np.asarray(s_perm, dtype=np.float32)

    runtime_sec = time.time() - start
    return TestResult(
        method_name="full_retraining",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=float(stat_true),
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": true_model,
            "pred_true": pred_true,
            "true_isodepth": true_isodepth,
            "lowest_isodepth": np.asarray(lowest_isodepth, dtype=np.float32),
            "lowest_S": np.asarray(lowest_S, dtype=np.float32),
            "lowest_stat": float(lowest_stat),
            "lowest_perm_index": int(np.argmin(stat_perm)),
            "highest_isodepth": np.asarray(highest_isodepth, dtype=np.float32),
            "highest_S": np.asarray(highest_S, dtype=np.float32),
            "highest_stat": float(highest_stat),
            "highest_perm_index": int(np.argmax(stat_perm)),
        },
    ).validate()


def run_frozen_encoder_method(
    dataset: DatasetBundle, config: TestConfig, device: torch.device | None = None
) -> TestResult:
    dataset.validate()
    config.validate()
    metric = canonicalize_metric_name(config.metric)
    device = device or resolve_device(config.device)

    start = time.time()
    true_model, pred_true = train_isodepth_model(
        dataset.S,
        dataset.A,
        config,
        device=device,
        seed_offset=0,
        model_label="true model",
    )
    stat_true = compute_metric(metric, dataset.A, pred_true)
    true_isodepth = _extract_isodepth_from_model(true_model, dataset.S, device)

    rng = np.random.default_rng(config.seed)
    stat_perm = np.zeros(config.n_perms, dtype=np.float64)
    lowest_stat = None
    lowest_isodepth = None
    lowest_S = None
    highest_stat = None
    highest_isodepth = None
    highest_S = None
    for i in range(config.n_perms):
        perm = rng.permutation(dataset.n_cells)
        s_perm = dataset.S[perm]
        model_perm, pred_perm = train_frozen_decoder_model(
            true_model,
            s_perm,
            dataset.A,
            config,
            device=device,
            seed_offset=i + 1,
            model_label=f"permuted decoder {i + 1}/{config.n_perms}",
        )
        stat_perm[i] = compute_metric(metric, dataset.A, pred_perm)
        current_isodepth = _extract_isodepth_from_model(model_perm, s_perm, device)
        if lowest_stat is None or stat_perm[i] < lowest_stat:
            lowest_stat = float(stat_perm[i])
            lowest_isodepth = current_isodepth
            lowest_S = np.asarray(s_perm, dtype=np.float32)
        if highest_stat is None or stat_perm[i] > highest_stat:
            highest_stat = float(stat_perm[i])
            highest_isodepth = current_isodepth
            highest_S = np.asarray(s_perm, dtype=np.float32)

    runtime_sec = time.time() - start
    return TestResult(
        method_name="frozen_encoder",
        metric=metric,
        p_value=permutation_p_value(metric, stat_true, stat_perm),
        stat_true=float(stat_true),
        stat_perm=stat_perm,
        runtime_sec=runtime_sec,
        n_cells=dataset.n_cells,
        n_genes=dataset.n_genes,
        config={"test": config.__dict__.copy()},
        artifacts={
            "model": true_model,
            "pred_true": pred_true,
            "true_isodepth": true_isodepth,
            "lowest_isodepth": np.asarray(lowest_isodepth, dtype=np.float32),
            "lowest_S": np.asarray(lowest_S, dtype=np.float32),
            "lowest_stat": float(lowest_stat),
            "lowest_perm_index": int(np.argmin(stat_perm)),
            "highest_isodepth": np.asarray(highest_isodepth, dtype=np.float32),
            "highest_S": np.asarray(highest_S, dtype=np.float32),
            "highest_stat": float(highest_stat),
            "highest_perm_index": int(np.argmax(stat_perm)),
        },
    ).validate()


def run_permutation_method(dataset: DatasetBundle, config: TestConfig) -> TestResult:
    device = resolve_device(config.device)
    print(f"device: {device}")

    if config.method == "perturbation_robustness":
        return run_perturbation_robustness_method(dataset, config, device=device)
    if config.method == "parallel_permutation":
        return run_parallel_permutation_method(dataset, config, device=device)
    if config.method == "full_retraining":
        return run_full_retraining_method(dataset, config, device=device)
    if config.method == "frozen_encoder":
        return run_frozen_encoder_method(dataset, config, device=device)
    if config.method == "gaston_mix_closed_form":
        return run_gaston_mix_closed_form_method(dataset, config, device=device)
    raise ValueError(f"Unsupported test.method '{config.method}'")
