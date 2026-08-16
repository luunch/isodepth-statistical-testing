"""Tiny JSON-spec loader shared by scripts/studies drivers."""
from __future__ import annotations

import json
from pathlib import Path


def load_spec(spec_path: str | Path) -> dict:
    with open(spec_path) as fh:
        return json.load(fh)
