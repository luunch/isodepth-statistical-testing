"""Shared synthetic dataset cache helpers for kernel-noise studies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.schemas import DatasetBundle


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
    return DatasetBundle(
        S=np.asarray(payload["S"], dtype=np.float32),
        A=np.asarray(payload["A"], dtype=np.float32),
        meta=meta,
    ).validate()
