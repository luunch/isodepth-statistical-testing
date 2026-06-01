"""Expected best-of-K rerun loss curve.

A ``parallel_permutation`` isodepth fit trains ``test.n_reruns`` independent random
initializations of the network and keeps the rerun with the lowest training
reconstruction loss.  This module quantifies how many reruns are actually worth
running: given the per-rerun training losses obtained by training ``R`` reruns on a
single (unpermuted) dataset, it estimates, for each ``k`` in ``1..R``, the expected
value of the best (minimum) loss when only ``k`` reruns are kept.

The estimate is a Monte-Carlo over random size-``k`` subsets of the ``R`` reruns
(``n_subsamples`` draws per ``k``, min loss per draw, averaged).  The resulting curve
flattens once extra reruns stop meaningfully improving the best loss -- the value of
``k`` at the elbow is the number of reruns that is actually necessary.

Pure ``numpy`` / ``matplotlib`` (no torch, no GPU).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def expected_min_loss_curve(
    rerun_losses: Sequence[float] | np.ndarray,
    *,
    ks: Optional[Sequence[int]] = None,
    n_subsamples: int = 100,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Monte-Carlo expected best-of-``k`` loss for every ``k`` in ``ks``.

    Parameters
    ----------
    rerun_losses:
        1-D array of length ``R`` with the final training loss of each rerun.
    ks:
        K values to evaluate.  Defaults to ``1, 2, ..., R``.
    n_subsamples:
        Number of random size-``k`` subsets drawn per ``k`` (the user-requested
        "select k samples ``n`` times each").  When ``k == R`` the subset is the
        full set and a single deterministic draw is used.
    seed:
        Seed for the subsampling RNG.

    Returns
    -------
    dict with keys ``ks``, ``mean`` (expected min loss), ``std``, ``lo``/``hi``
    (5th/95th percentiles of the per-draw min loss), all aligned with ``ks``.
    """
    losses = np.asarray(rerun_losses, dtype=np.float64).reshape(-1)
    n_reruns = losses.shape[0]
    if n_reruns == 0:
        raise ValueError("rerun_losses must be non-empty")
    if ks is None:
        ks = list(range(1, n_reruns + 1))
    else:
        ks = sorted({int(k) for k in ks if 1 <= int(k) <= n_reruns})
    if not ks:
        raise ValueError("No valid k values in 1..R")

    rng = np.random.default_rng(seed)
    mean = np.zeros(len(ks))
    std = np.zeros(len(ks))
    lo = np.zeros(len(ks))
    hi = np.zeros(len(ks))

    for j, k in enumerate(ks):
        if k >= n_reruns:
            mins = np.array([float(losses.min())])
        else:
            # Vectorized: one random ordering per draw, take the first k indices.
            order = np.argsort(rng.random((n_subsamples, n_reruns)), axis=1)[:, :k]
            mins = np.take(losses, order).min(axis=1)
        mean[j] = mins.mean()
        std[j] = mins.std()
        lo[j], hi[j] = np.percentile(mins, [5, 95])

    return {
        "ks": np.asarray(ks, dtype=np.int64),
        "mean": mean,
        "std": std,
        "lo": lo,
        "hi": hi,
    }


def render_expected_min_loss_figure(
    curve: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: Optional[str] = None,
    n_subsamples: int = 100,
    metric_label: str = "training reconstruction loss (MSE)",
) -> Path:
    """Render the single expected-best-of-K loss graph and save it to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = np.asarray(curve["ks"])
    mean = np.asarray(curve["mean"])
    lo = np.asarray(curve["lo"])
    hi = np.asarray(curve["hi"])
    n_reruns = int(ks.max()) if ks.size else 0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, mean, "-o", color="C0", ms=3, label="expected min loss")
    ax.fill_between(ks, lo, hi, color="C0", alpha=0.2, label="5-95% of draws")
    ax.set_xlabel("number of reruns k")
    ax.set_ylabel(f"expected best-of-k {metric_label}")
    ax.set_title(
        title
        or f"Expected best-of-k rerun loss (R={n_reruns}, {n_subsamples} subsamples per k)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
