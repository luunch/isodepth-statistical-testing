# Scripts + Experiments Layout

Archive snapshot: `archive/pre-cleanup-20260806` (from `origin/main` before this reorg).

## Import stability

These public paths must keep working (shims re-export from new locations when needed):

- `from experiments.configuration import …`
- `from experiments.recursive_svg import …`
- `python -m experiments.<study>_sweep` / `_analysis` (flat module names)
- `from experiments.kernel_noise_study import load_dataset_cache` (re-exports `experiments.core.dataset_cache`)
- `from experiments.existence_sigma import load_result_payload, scan_result_json_paths, write_csv`

Prefer new imports:

- `from experiments.core.dataset_cache import load_dataset_cache`
- `from experiments.core.study_io import load_result_payload, scan_result_json_paths, write_csv`
- `from experiments.core.study_spec import load_spec`
- `from experiments.core.paths import repo_root`

## `experiments/` target

| Path | Role |
|---|---|
| `configuration.py`, `recursive_svg.py` | Framework (stay top-level) |
| `core/` | Shared study helpers (`study_io`, `dataset_cache`, `study_spec`, `paths`) |
| `studies/<name>/` | Active sweep families (existence_sigma, fourier_kmax, …) |
| `diagnostics/` | batchsize, bias detection, rerun_* |
| `figures/` | publication_figures, merfish viz |

Flat `experiments/<name>.py` files are **shims** that re-export from the nested modules so README/`python -m` entrypoints do not break.

## `scripts/` target

| Path | Role |
|---|---|
| `data_prep/` | h5ad builders, DeLeakage, CosMx segmentation, MOSTA download |
| `regen/` | Plot regeneration from finished runs |
| `posthoc/` | True-vs-null diagnostics on finished runs |
| `studies/<topic>/` | Topic-scoped drivers (cosmx, dlpfc_layer3, msr_diagnostics, …) |

Top-level `scripts/<name>.py` shims remain only where other code still does `from scripts.<name> import …`.

## Where new code goes

| Kind | Location |
|---|---|
| Config I/O / standardized outputs | `experiments/configuration.py` |
| Multi-condition sweep + analysis | `experiments/studies/<name>/` |
| One-off diagnostic on finished runs | `scripts/posthoc/` or `scripts/studies/<topic>/` |
| Dataset packaging | `scripts/data_prep/` |
| Helper used by many studies | `experiments/core/` or `methods/` — never leave in `scripts/` |

## Archived from mainline (recoverable on archive branch)

Kernel-noise null one-shots and ad-hoc diags removed in the cleanup commit:

- `_diag.py`, `_diag2.py`
- `kernel_noise_{four_null,block_parallel,cholesky_null,variogram_null,spgp_450ep,spgp_parallel,rank_msr,joint_trunc_msr,joint_trunc_aclength,joint_trunc_r30_delta05,multiscale_gene_cov_null}.py`
- `joint_trunc_{rank_msr_smoke,partial_rank_smoke,msr_positive_controls}.py`
