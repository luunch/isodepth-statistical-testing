"""Shim — stable import/CLI path. See docs/SCRIPTS_EXPERIMENTS_LAYOUT.md."""
from __future__ import annotations

from importlib import import_module
import sys

_TARGET = "scripts.data_prep.segment_cosmx_celltype_regions"
_mod = import_module(_TARGET)

# When imported under the legacy name, alias the real module so
# ``from experiments.X import _private`` keeps working.
if __name__ != "__main__":
    sys.modules[__name__] = _mod
else:
    if hasattr(_mod, "main"):
        _mod.main()
    else:
        import runpy
        runpy.run_module(_TARGET, run_name="__main__")
