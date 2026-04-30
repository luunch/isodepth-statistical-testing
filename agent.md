# AGENT.md


## Environment Setup
Commands to activate testing enviroment:
```bash
conda activate isodepth_env
```
Currently running files will not work as this environment has no gpu setup.

## 1. Purpose
This folder is a research workflow to define a statistical test for determining when spatial transcriptomic genetic variation has arisen via random chance or due to some intrinsic quality of the data. We also aim to establish a biological gradient of this genetic variation. Core literature on this method of encoding an isodepth is GASTON and GASTON-mix. We replicate the architectures of these neural networks to define an isodepth on our data.

## 2. Allowed Actions
- Modify code in: ./analysis, ./data, ./experiments, ./methods, ./validation, ./tests
- Create new files in: ./analysis, ./data, ./experiments, ./methods, ./validation, ./tests, ./configs
- Run commands: NEED TO FURTHER SPECIFY
- Install dependencies: yes/no

## 3. Forbidden Actions
- Do not modify: ./data/h5ad/
- Do not access: ./data/h5ad/
- Do not use:

## 4. Development Workflow
1. Understand the task
2. Identify relevant files
3. Make minimal changes 
4. Run tests
5. Ensure formatting/linting passes

## 5. Coding Standards

## 6. Testing Requirements

## 7. Project Structure
Explain key directories and what lives where.

## 8. Common Tasks

## 9. Reusable Investigation Notes
- `configs/mouse_hippocampus_existence.json` currently omits `data.q`, while prior successful hippocampus existence and perturbation runs used `q: 10`, yielding a 20-dimensional Poisson low-rank latent feature space instead of full gene expression.
- For single-run Slurm wrappers around `run_permutation.py`, prefer CLI overrides instead of cloning configs when only `q`, `decoder`, or `run_name` change; invoke it as `python run_permutation.py`, not `python -m run_permutation.py`.
- Slurm wrappers that import `torch` need the conda-packaged CUDA libs prepended in `LD_LIBRARY_PATH`, with `nvjitlink` before `cusparse`; otherwise imports can fail with `libcusparse.so.12: undefined symbol __nvJitLinkComplete_12_4`.
- `methods/trainers/isodepth.py` trains all `n_perms + 1` permutation models in parallel for `parallel_permutation`, so GPU memory grows with both permutation count and feature dimension; raw-gene hippocampus runs are therefore much more memory-sensitive than the `q: 10` configs.
- When `n_reruns > 1`, `save_standardized_outputs` now emits a square-grid PNG of the true-data rerun isodepths using the trainer metadata key `true_rerun_isodepths`; it does not attempt to plot reruns for every permutation/null model.
- `cross_validation` reuses the same coordinate-permutation null as `parallel_permutation`, but trains every batched model on a shared train-mask and computes the observed/null statistic from the complementary held-out test mask.
- Existence sweep/analysis modules now treat `parallel_permutation` and `cross_validation` as interchangeable base/result methods; they should preserve the configured existence method instead of rewriting it during sweep expansion.
- The h5ad preprocessing pipeline now supports `data.log1p`; it applies `log(1 + x)` after gene filtering and before z-scoring, and it is intentionally rejected when `data.q` is set because the Poisson low-rank transform expects count-scale inputs.
- `experiments/parallelization_comparison.py` is the dedicated contamination-check harness for parallel vs sequential permutation training; it forces `n_reruns = 1`, requires `sgd_batch_size = 0`, reuses one shared permutation batch, aligns initial weights by slicing a reference parallel state, and accepts an explicit `epochs` override in the experiment spec.
- `export_gaston_true_isodepth.py` now aims to mirror `configs/mouse_hippocampus_existence.json` as closely as possible: it uses the CA1 h5ad defaults, separates `data_seed` and `train_seed` (defaulting to `0` and `42`), exports the `true_isodepth` from the `parallel_permutation` true-model slot rather than a standalone fit, and disables the older extra `q`-space/coordinate z-scoring unless `--rescale-q-inputs` is explicitly requested.
- In the current `/weka` mount, Python processes may not be able to create new files under the repo-local `results/` tree even though code edits succeed there; `export_gaston_true_isodepth.py` therefore falls back to `/tmp/isodepth_export/` when the requested default output location is read-only and explicit output paths were not supplied.
- **MERFISH visualization**: `experiments/merfish_hypothalamus_visualization.py` loads `configs/experiments/merfish_hypothalamus_visualization.json`, applies the same spatial coordinate null shuffles as `parallel_permutation` (torch `randperm` on CPU), and writes per-gene PNGs plus `resolved_genes.json`. Each run also saves **`spatial_principal_coordinates.png`** (PCA on cell positions only: PC1/PC2 gradient fill + contour lines; path recorded in JSON). Default output folders when `--out-dir` is omitted: JSON `visualization.out_dir` (typically `results/merfishhypothalamus_visualization/`); `results/merfish_hypothalamus_visualization_nocontour/` if `--no-contour`; `results/merfish_hypothalamus_visualization_nozero/` if `--hide-zero-expression` or `"hide_zero_expression": true` under `visualization`; `results/merfish_hypothalamus_visualization_nocontour_nozero/` if both. Hide-zero skips drawing near-zero cells (`|value|<=1e-15`); filenames gain a `_nozero` suffix when that mode is on. Use `--out-dir` to override. It does not call `run_permutation.py` or train models.
