# Agent notes — isodepth-statistical-testing

## GPU memory model (parallel_permutation)
- The batched trainer trains `M = (n_perms + 1) * n_reruns` independent models at once
  (e.g. 99 perms + 1 true, 30 reruns => 3000 parallel models).
- Dominant VRAM terms scale with `M * cells * genes`:
  - decoder output-layer weights `ParallelLinear(M, 20, G)` => `(M, G, 20)` (x4 with grad+Adam m/v),
  - per-minibatch activations `(M, sgd_batch_size, G)` plus an equal-size squared-error buffer,
  - finalize forward chunks 128 *models* but over ALL cells: `(128, N, G)`.
- Expression matrix `A` is NOT duplicated per model — `broadcast_a` keeps a single `(N, G)` tensor
  (see `train_batched_isodepth_model`). Only trainable weights + per-model outputs replicate.
- Reruns (`n_reruns`) are an optimization-robustness multiplier (pick best train loss), NOT
  statistically required — lowering it is the cheapest memory win. Permutations ARE required.
- On `torch.cuda.OutOfMemoryError`, `train_parallel_isodepth_model` auto-halves into chunks and
  prints `[OOM split] Trying N chunks ...`; run still succeeds but very slowly.

## Gene subsetting
- `data.top_var_genes` (DataConfig, int, default 0, h5ad-only) keeps the top-N scanpy highly
  variable genes via `sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=N)`, then
  subsets `adata[:, highly_variable]` right after read (so X/layers/var_names stay aligned).
  `0` = use all genes; `N >= n_genes` warns and keeps all. Requires `scanpy` (installed in
  `isodepth_env`). CLI: `--top-var-genes N`. Implemented in `data/h5ad_loader.py`.
- `flavor="seurat"` expects log-transformed input; for raw counts the statistically correct
  flavor is `seurat_v3` (not currently wired up).
- The old `data.highly_variable_only` boolean (read a precomputed `adata.var['highly_variable']`
  column) was removed in favor of `top_var_genes`.
- Only `min_cells_per_gene` and `top_var_genes` control gene subsetting.

## Env
- conda env: `isodepth_env` (`source ~/miniforge3/etc/profile.d/conda.sh && conda activate isodepth_env`).
- `torch` import in this env is slow (~seconds); a hang at import is usually just that, not a crash.
- No `pytest` in the env; run tests with `python -m unittest tests.test_<name>`.

## Removed: `exact_existence` method
- The iterative dimension-selection test method `exact_existence` (and its `test.max_spatial_dims`
  config field + `--max-spatial-dims` CLI flag) was fully removed from schemas, `methods/permutation.py`
  (`run_exact_existence_method` + `_summarize_exact_existence_*` + now-unused `_delta_p_value`/
  `_select_low_high_indices` helpers), `analysis/plots.py` (`_save_exact_existence_triptych` + the
  unused `_as_dimension_matrix`), `experiments/configuration.py`, `run_permutation.py`, README, and tests.
  `test.alpha` was kept (still a valid config field; used by the existence-sigma experiments).
- Pre-existing unrelated test failure: `tests/test_schemas.py::test_unknown_covariate_type_rejected`
  fails because of a separate working-tree `CovariateConfig` refactor (validate now only rejects empty
  strings, since any non-empty string is treated as a valid `obs` key). Not caused by the removal.

## Rerun-count experiment (`experiments/rerun_convergence.py`)
- Quantifies how many `n_reruns` an isodepth fit actually needs. Trains `R` reruns (default 100)
  concurrently on the SAME unpermuted dataset (`n_perms=0`, `patience=0`, `covariate=None`) via
  `train_parallel_isodepth_model`, then Monte-Carlo subsamples `k` of `R` reruns (`--n-subsamples`,
  default 100) for each `k=1..R`, taking the min training loss per draw and averaging => expected
  best-of-k loss curve. Outputs ONE graph to `results/<run_name>_rerun_convergence/`.
- Owns `n_reruns`/`n_perms`/`patience`; all other settings (epochs, lr, q, decoder, device, seed)
  come from the `--config`. CLI overrides: `--n-reruns --n-subsamples --epochs --max-cells --q
  --seed --device --out-dir --run-name`. Run as a script: `python experiments/rerun_convergence.py
  --config ...` (the file inserts REPO_ROOT into sys.path; `python -m experiments.rerun_convergence`
  also works only when cwd is the repo root since there is no `experiments/__init__.py`).
- Math justification: rerun selection uses the unmasked training MSE, and the existence statistic
  `nll_gaussian_mse` is a strictly monotonic transform of that same MSE, so best-of-k behaviour is
  fully determined by the per-rerun MSE vector (`train_loss_per_rerun`). Pure analysis/plot helpers
  live in `analysis/rerun_convergence.py` (`expected_min_loss_curve`, `render_expected_min_loss_figure`).
- Sandbox note: the repo `results/` tree is on the `/weka/scratch` mount, which the agent shell
  sandbox sees as root-owned/read-only; real runs as `ajain71` write there fine. To produce an
  agent-side demo figure, pass `--out-dir` under `/weka/home/ajain71` or run the shell with full
  permissions. Quick CPU smoke test: `--config configs/radial.json --device cpu --epochs 100`.
