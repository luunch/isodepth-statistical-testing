"""Repo-root resolution that survives nested package/script moves."""
from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Walk parents until ``run_permutation.py`` is found."""
    p = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "run_permutation.py").is_file():
            return parent
    raise RuntimeError(f"Could not locate repo root from {p}")
