"""Shared study-spec loading and the CLI flags every study accepts.

Study-specific fields stay in each study's own loader. The keys and flags
here are the ones repeated across those loaders: the JSON object itself,
``experiment_name`` / ``base_config`` / ``output_root``, repo-relative paths,
and ``--spec`` / ``--dry-run`` / ``--max-runs``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.core.paths import repo_root


def load_spec(spec_path: str | Path) -> dict[str, Any]:
    with open(spec_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Study spec must be a JSON object: {spec_path}")
    return payload


def resolve_repo_path(path_value: str | Path, *, start: str | Path | None = None) -> Path:
    """Resolve a spec path against the repo root. Absolute paths stay absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root(start) / path).resolve()


def resolve_repo_paths(
    values: list[str | Path] | tuple[str | Path, ...],
    *,
    start: str | Path | None = None,
) -> list[Path]:
    return [resolve_repo_path(value, start=start) for value in values]


def common_study_fields(payload: Mapping[str, Any]) -> tuple[str, Path, Path]:
    """The three fields present on essentially every study spec."""
    return (
        str(payload["experiment_name"]),
        resolve_repo_path(payload["base_config"]),
        resolve_repo_path(payload["output_root"]),
    )


def analysis_dir_for_output(output_root: str | Path) -> Path:
    return Path(output_root) / "analysis"


def runs_dir_for_output(output_root: str | Path) -> Path:
    return Path(output_root) / "runs"


def manifest_path_for_output(output_root: str | Path) -> Path:
    return Path(output_root) / "manifest.json"


def add_spec_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str = "Path to the experiment spec JSON",
) -> None:
    parser.add_argument("--spec", required=True, help=help)


def add_sweep_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing them",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of planned runs",
    )


def sweep_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_spec_argument(parser)
    add_sweep_run_arguments(parser)
    return parser


def analysis_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_spec_argument(parser)
    return parser
