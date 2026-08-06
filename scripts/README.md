# Scripts layout

See [`docs/SCRIPTS_EXPERIMENTS_LAYOUT.md`](../docs/SCRIPTS_EXPERIMENTS_LAYOUT.md).

| Folder | Purpose |
|---|---|
| `data_prep/` | Build/annotate h5ads, CosMx segmentation, downloads |
| `regen/` | Regenerate plots from finished run artifacts |
| `posthoc/` | True-vs-null / GSEA diagnostics on finished runs |
| `studies/<topic>/` | Topic-scoped experiment drivers |

Top-level `scripts/*.py` files are **compatibility shims** for old import/CLI paths.
New code should live in the folders above, not at the top level.

Shared helpers belong in `experiments/core/` or `methods/`, not here.
