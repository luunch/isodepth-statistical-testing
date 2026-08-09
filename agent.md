# Agent notes

## `*_loss_curve.png` plot: blue "True" line was NOT apples-to-apples with gray/p-value (FIXED, twice)

**Status: fixed** in `analysis/plots.py`, in two rounds:

**Round 1** — `_permutation_loss_curves_from_slots` was replaced with
`_true_and_perm_loss_curves_from_slots`, so the "True" curve (previously raw `loss_history` =
rerun 0 only) also selected the min-over-reruns loss like the gray/p-value curves.

**Round 2 (important correction)** — Round 1's "min over reruns" was computed **independently at
every epoch**, which does NOT match how the real pipeline selects a rerun
(`best_rerun_index_per_model = np.argmin(train_loss_per_rerun, axis=1)` in
`methods/trainers/isodepth.py`, computed **once** from the post-training loss). An epoch-wise
`np.nanmin` can silently switch which rerun is "winning" from epoch to epoch, stitching together
pieces of *different* reruns into an artificial lower envelope that no single actual training run
ever achieved — confirmed empirically: for this run's true model, the naive per-epoch argmin
switched between 3–4 different reruns over the course of training (see verification snippet
below). Fixed via new helper `_select_fixed_rerun_curves_from_slots`: for each model, pick the
rerun with the lowest loss **at the final recorded epoch only**, then use that *one* rerun's full
trajectory for every epoch. Both `_true_and_perm_loss_curves_from_slots` (blue/gray curves) and
`_pvalue_trajectory_from_slots` (red p-value) now use this same fixed-rerun selection, so all
three curves in the plot consistently describe the trajectories of the models that were actually
kept, rather than a hypothetical "best possible at each instant" envelope.

```python
# Check how many times the naive per-epoch argmin selection would have switched reruns
# (diagnostic used to confirm the round-1 bug before writing the round-2 fix):
import numpy as np
argmin_per_epoch = np.argmin(reshaped[:, model_idx, :], axis=1)  # reshaped: (n_epochs, n_models, n_reruns)
n_switches = np.sum(np.diff(argmin_per_epoch) != 0)
```

All `*_loss_curve.png` files across `results/calicost/**` that have a corresponding
`*_loss_history.npz` (16 clone dirs as of 2026-08-07, found via
`find results/calicost -name "*_loss_history.npz"`) were regenerated in place to reflect the fix
— see snippet below for how to regenerate from an npz + `TestResult` without rerunning the full
pipeline (can batch over `Path("results/calicost").rglob("*_loss_history.npz")`). Runs without a
`*_loss_history.npz` (older configs that didn't save per-slot rerun history) only ever plotted the
single `loss_history` curve with no permutation/p-value overlay, so they're unaffected by this fix
and weren't touched. All result.json files under `results/calicost` use
`metric: "nll_gaussian_mse"`, so that's a safe default when reconstructing a minimal `TestResult`
for regeneration.

```python
import numpy as np, sys
sys.path.insert(0, ".")
from data.schemas import TestResult
from analysis.plots import save_loss_curve_plot

d = np.load("<clone>_loss_history.npz", allow_pickle=True)
result = TestResult(
    method_name="parallel_permutation", metric="nll_gaussian_mse",
    p_value=0.0, stat_true=1.0, stat_perm=np.array([1.0, 1.0]),
    runtime_sec=0.0, n_cells=0, n_genes=0, config={},
    artifacts={
        "loss_history": d["loss_history"],
        "loss_history_per_slot": d["loss_history_per_slot"],
        "training_metadata": {"n_reruns": int(d["n_reruns"][0])},
    },
).validate()
save_loss_curve_plot(result, "<clone>_loss_curve.png", title="...")
```

Original bug description below, kept for context on why this mattered.

### Original issue

In `analysis/plots.py::save_loss_curve_plot` (and helpers `_permutation_loss_curves_from_slots`,
`_pvalue_trajectory_from_slots`):

- The blue **"True"** curve plotted is `loss_history`, which is only **rerun 0** of the
  true model's `n_reruns` (typically 10) random restarts — a single trajectory.
- The faint gray **permutation** curves and the red **p-value** trajectory both use the
  **min over all reruns** at each epoch for every model (true and each permutation), matching
  final model selection logic used for the actual significance test.

Because the gray/p-value curves get to pick their best-of-N at every epoch while blue shows only
one run, it's possible (and happened in practice, e.g.
`results/calicost/HT112C1_U2_loss_difference_gt0p7/loss_diff_gt0p7_cropx_tumor_logcounts/2.0/2.0_loss_curve.png`)
for the blue line to visually sit *above* all the gray permutation curves near the end of
training even though the p-value correctly collapses to the floor (true's best-of-N rerun is far
below every permutation's best-of-N). This is a plotting-artifact mismatch, not a bug in the
significance test itself.

Additionally, with `n_perms` in the hundreds, gray permutation line alpha is set very low
(`min(0.35, max(0.04, 4.0/n_perms))`, so ~0.04 at n=249) and the best-of-N band is often only
~0.01 loss units wide by the end of training — so the gray band can be effectively invisible,
swallowed by the blue line's stroke width, reinforcing the illusory "true above all nulls" look.

To verify/reproduce this kind of check, load `<clone>_loss_history.npz` (contains `n_reruns`,
`loss_history` (slot 0 only), `loss_history_per_slot` shape `(n_epochs, (1+n_perms)*n_reruns)`)
and manually reshape/min over reruns per model to compare true-vs-perm at any epoch.

Python env for this repo: `/weka/home/ajain71/miniforge3/envs/isodepth_env/bin/python`
(system `python3`/`python` do not have numpy etc.).

## `loss_diff_clone2_linear` run: stale/legacy config, must match cells by coordinates, not by reloading

`results/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2/loss_diff_clone2_linear`
(sample `HT306P1-S1H1Fc2U1Z1Bs1`, clone `2.0`, decoder `linear`, `n_cells=195`) whitens on
`calicost_tumor_proportion` (tumor clone proportion) **only** -- its
`loss_diff_clone2_linear_result.json` -> `config.data.covariate_whitening` has no total-counts
term, so total-counts/library-depth is an *unmodeled* technical covariate for this run's test.

This run's saved `config.data` predates `obs_numeric_filters`/`spatial_crop`/
`spatial_denoise_radius_um` being tracked in output serialization (those keys are entirely
absent from the dumped JSON, not just `null`), so naively calling `load_dataset` on the saved
config reloads 197 clone-2.0 cells (only the 2 NaN-`calicost_tumor_proportion` spots dropped)
instead of the 195 the model actually trained on -- **do not** assume reload order/count matches
`{type}_isodepths.npz` for this run. (The sibling config
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2.json` now produces a
*different*, newer run `loss_diff_clone2_linear_gt0p5_cropy` with an explicit
`obs_numeric_filters`/`spatial_crop`/`spatial_denoise_radius_um`, not this one -- it's not a
usable substitute config for reproducing `loss_diff_clone2_linear` exactly.)

To align any newly-computed per-spot quantity (e.g. QC covariates from raw h5ad `obs`) with the
saved `true_isodepth` (order-sensitive, length 195) for this run: un-standardize the saved,
z-scored `S` from `{type}_isodepths.npz` using `coord_mean`/`coord_std` from
`result.json["artifacts"]["dataset_meta"]`, then nearest-neighbor-match (`scipy.spatial.cKDTree`)
back to raw `adata.obsm["spatial"]` coordinates for clone-2.0 spots with non-NaN
`calicost_tumor_proportion` in the source h5ad. Matches are near-exact (max residual ~1e-3 in
raw pixel units, well below any plausible spot spacing) and unique across all 195 spots --
verified empirically. Source h5ad (`data/h5ad/calicost/HT306P1-S1H1Fc2U1Z1Bs1.h5ad`) already has
precomputed QC obs columns `total_counts`/`pct_mt`/`n_genes`/`log1p_total_counts` (added via
`scripts/data_prep/add_qc_obs_columns.py`, raw-counts-layer-based), so no need to recompute
these from raw counts once spots are matched.

Implemented in `scripts/posthoc/loss_diff_clone2_linear_qc_covariate_diagnostics.py` (run via
`python -m scripts.posthoc.loss_diff_clone2_linear_qc_covariate_diagnostics`), which produces
`.../loss_diff_clone2_linear/loss_diff_clone2_linear_pct_mt_total_counts_diagnostics.png`: a
2-column (pct_mt, total_counts) x 3-row (scatter-vs-isodepth-with-Spearman-rho/p, spatial
colored by isodepth, spatial colored by covariate) diagnostic grid, matching the style of the
earlier `isodepth vs pct_mt` reference plot but for this specific run/clone. Result: strong
residual confounding found for this run -- `rho(isodepth, total_counts) = 0.92` (p=2e-81),
`rho(isodepth, pct_mt) = -0.69` (p=2e-28); the spatial total_counts map visually looks almost
identical to the spatial isodepth map, consistent with total_counts (unlike tumor_proportion)
not being whitened out here.

### GSEA for `loss_diff_clone2_linear` — general `postprocess_gsea_isodepth.py` silently skips this run

`scripts/posthoc/postprocess_gsea_isodepth.py` (generic pre-ranked-GSEA-vs-isodepth tool, takes
`config_path result_json_path --gmt <gmt>`) can't be used directly on `loss_diff_clone2_linear`:
its `_extract_groups` "separate" cell-type branch reloads the dataset via the saved config (197
cells, per the mismatch above), so the shape check against the saved 195-cell NPZ fails and the
group is silently skipped (just a `[warn] skipping ...: cell mismatch` line, no error) --
producing zero output, not a crash.

Fixed via a dedicated script, `scripts/posthoc/gsea_loss_diff_clone2_linear.py` (run via
`python -m scripts.posthoc.gsea_loss_diff_clone2_linear --gmt data/gmt/h.all.v2024.1.Hs.symbols.gmt`),
which: (1) recovers the correct 195-cell subset via the same coordinate-matching technique as the
QC diagnostics script above, (2) re-runs `preprocess_celltype_subset` (from `data/h5ad_loader.py`)
on those correctly-matched raw counts using the run's saved `separate_preprocessing` params to
recover gene names 1:1-aligned with the saved NPZ `A` matrix's columns, (3) **verifies** the
reconstruction against the saved `A` (`max_abs_diff` printed; run confirmed this is exactly `0`,
i.e. cell matching + preprocessing are exactly reproducible) before trusting the recovered gene
names, then (4) reuses `_score_genes`/`_collapse_duplicate_genes`/`_gsea_preranked`/CSV+plot
writers imported directly from `postprocess_gsea_isodepth.py` (no duplicated GSEA math) to score
genes against the saved `A`/`true_isodepth` and run standard pre-ranked GSEA. Outputs written to
`.../loss_diff_clone2_linear/gsea_isodepth/2.0_{prerank_scores,gsea_results}.csv` +
`2.0_top_pathways.png`, matching the existing convention from sibling clone-2 GSEA runs (e.g.
`results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p5_cropy/gsea_isodepth/`).
Gene-set file used: `data/gmt/h.all.v2024.1.Hs.symbols.gmt` (MSigDB Hallmark, 50 sets; this is
the only `.gmt` file in the repo, under `data/gmt/`). Default GSEA params match sibling runs:
`min_size=15, max_size=500, n_permutations=250, weight=1.0, score_method=spearman, seed=0`.

Result (2026-08-07): 36/50 Hallmark pathways tested (14 excluded by min/max overlap size); top
hits by q-value are interferon-gamma response, EMT, myogenesis, complement (all NES < -1.5,
i.e. anti-correlated with isodepth) and DNA repair (NES ~+1.65, positively correlated) — all
q < 0.03. This is a distinct/independent gene ranking from the sibling `..._gt0p5_cropy` run
(different cell subset/crop), so don't assume the two GSEA result sets should match.

Follow-up finding: of `HALLMARK_HYPOXIA`'s 200 genes, all 200 are present in the full h5ad
transcriptome, but only 42 survive into the `top_var_genes=3000` HVG-filtered background
(2775 genes) that GSEA actually ranks against — i.e. the weak/non-significant `HALLMARK_HYPOXIA`
hit in this run's GSEA (`overlap_size=42`, NES~+0.50, q~0.99) reflects HVG filtering discarding
~80% of the pathway, not necessarily biological absence of a hypoxia signal.

### New `data.gene_list` preprocessing option — restrict analysis to a fixed gene panel instead of top-N HVG

Added (2026-08-07) as a general repo feature (not run-specific) to let any h5ad-backed config
restrict every preprocessing statistic (gene support via `min_cells_per_gene`,
`normalize_total` size factors, per-gene z-score mean/std, and any HVG dispersion calc if
`top_var_genes` were also nonzero) to a fixed, user-supplied gene panel (e.g. every gene in one
pathway/gene-set) instead of the usual data-driven top-N-HVG selection:

- `data.gene_list: Optional[list[str]]` field on `DataConfig` (`data/schemas.py`) — non-empty,
  no duplicates, h5ad-only, and **mutually exclusive with `data.top_var_genes > 0`** (raises in
  `validate()`; must set `top_var_genes: 0` to use the panel as-is — gene_list *replaces* HVG
  selection, it doesn't add to it).
- `_apply_gene_list_filter` in `data/h5ad_loader.py` subsets `adata` to
  `set(gene_list) & set(adata.var_names)` right after `_apply_gene_exclusions` (i.e. before the
  `top_var_genes` HVG branch and before the `cell_type="separate"` deferred-preprocessing split,
  so it applies uniformly to both direct and per-cell-type pipelines). Records
  `gene_list`/`gene_list_requested_count`/`gene_list_matched_count`/`gene_list_missing_genes` in
  `dataset.meta` (mirrors the `exclude_gene_patterns`/`excluded_gene_*` convention). Wired through
  `load_dataset_from_config` and added to the `_compact_run_config` h5ad key whitelist in
  `experiments/configuration.py` (next to `exclude_gene_patterns`) so it round-trips into the
  saved `result.json`'s `config.data` — unlike several older `DataConfig` fields (`top_var_genes`,
  `normalize_total`, `obs_numeric_filters`, `spatial_crop`, ... — see the whitelist for the full
  set), which are silently dropped from the dumped config because that whitelist predates them;
  this is a pre-existing, broader gap not fixed here.
- Tests: `tests/test_schemas.py` (`test_gene_list_*`, validation) and
  `tests/test_mosta_coordinate_scale.py`
  (`test_loader_restricts_to_fixed_gene_list_before_preprocessing`, loader-level filtering +
  meta bookkeeping, including a not-present gene being reported in `gene_list_missing_genes`
  rather than raising).

Applied it to a new run: `configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_hypoxia_genes.json`
(clone 2 of HT306P1, same tumor-proportion-only whitening / obs_numeric_filters / spatial_crop /
spatial_denoise_radius_um as `..._gt0p5_cropy`, but `top_var_genes: 0` +
`gene_list: <all 200 HALLMARK_HYPOXIA genes>`) → run name
`loss_diff_clone2_linear_gt0p5_cropy_hypoxia_genes`. All 200 requested genes matched the h5ad
(`gene_list_missing_genes: []`); after per-clone `min_cells_per_gene=3` filtering, 183 genes
entered training (vs. only 42 of 200 when going through the standard top-3000-HVG path — see the
GSEA follow-up finding above). Result: isodepth remains highly significant restricted to only
this pathway's genes, `p_value=0.004` (n=180 cells), i.e. the spatial isodepth signal in this
clone is not solely carried by non-hypoxia HVGs. Also reran with the tumor-proportion threshold
bumped from `gt:0.5` to `gt:0.7` (same config file edited in place, run name
`..._gt0p7_cropy_hypoxia_genes`, n=125 cells) — also `p_value=0.004`.

**IMPORTANT caveat on both hypoxia-gene runs (gt0p5 and gt0p7): the isodepth axis is almost
entirely a total-counts/library-size gradient, not necessarily a biological hypoxia signal.**
Both `..._gt0p5_cropy_hypoxia_genes` and `..._gt0p7_cropy_hypoxia_genes` inherit
`covariate_whitening: calicost_tumor_proportion` only (no total-counts whitening) from their
parent `..._gt0p5_cropy`-lineage config. Checked directly (reload via `load_dataset` with
`covariate_whitening.obs_key` temporarily extended to
`["calicost_tumor_proportion", "total_counts", "pct_mt"]` so the QC columns get carried through
the same crop/denoise/standardize pipeline as training, then verified cell-order identity against
the saved NPZ's `S` via nearest-neighbor match before trusting the alignment -- unlike the
original legacy `loss_diff_clone2_linear` run, this current config reloads cleanly with an
*exact* identity-order match, no coordinate-matching workaround needed):
- gt0.5 (n=180): `rho(isodepth, total_counts) = 0.95` (p=3e-90), `rho(isodepth, pct_mt) = -0.72`
  (p=6e-30), `rho(isodepth, tumor_proportion) = 0.47` (p=3e-11).
- gt0.7 (n=125): `rho(isodepth, total_counts) = 0.94` (p=4e-61), `rho(isodepth, pct_mt) = -0.62`
  (p=1e-14), `rho(isodepth, tumor_proportion) = 0.28` (p=0.0018).

i.e. total_counts correlates with the fitted isodepth far more strongly than tumor_proportion
does (the thing that's actually whitened out). This means the significant hypoxia-gene-panel
existence test in this clone is confounded and should **not** be read as evidence of a genuine
oxygen-gradient signal without first whitening total_counts (or repeating with a
`covariate_whitening.obs_key` that includes it) and checking whether significance survives.

**Follow-up (same day, later superseded — see below): originally tried whitening total_counts
jointly with tumor_proportion via config
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_hypoxia_genes_totalcounts_whitened.json`
using `covariate_whitening_obs_key: ["calicost_tumor_proportion", "total_counts"]` (raw counts) —
this config and its `.../loss_diff_clone2_linear_gt0p7_cropy_hypoxia_genes_totalcounts_whitened/`
result dir were later found to be mis-specified (linear whitening term on a right-skewed raw
covariate under-corrects a log-linear depth effect — see the dedicated section below) and were
deleted (2026-08-07). Do not recreate this exact config; use `log1p_total_counts` instead (the
sibling `..._tumor_logcounts` run/config, also below).**

**Earlier same-day attempt at whitening total-counts used `log1p_total_counts` (not raw
`total_counts`) and came out NON-significant — this was superseded by the raw-`total_counts`
version above, but the two runs/results should not be confused.** Run dir:
`results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p7_cropy_hypoxia_genes_tumor_logcounts/`
(created 18:39, ~27 min *before* the `..._totalcounts_whitened` run at 19:06 described above — the
name is a legacy naming convention shared with `loss_diff_clone2_linear_gt0p5_cropy_tumor_logcounts`,
not literally "tumor log-counts"). Identical `data` config to `..._totalcounts_whitened` (same
clone 2.0, `gt:0.7` tumor-proportion filter, `spatial_crop.y.lt=1.9`, same 200-gene HALLMARK_HYPOXIA
`gene_list`, `n_cells=125`) **except**:
- `covariate_whitening_obs_key: ["calicost_tumor_proportion", "log1p_total_counts"]` (log1p of
  total counts) instead of `["calicost_tumor_proportion", "total_counts"]` (raw).
- `test.n_perms: 249` instead of `1000` (this run predates the perm-count bump).

Result: `p_value=0.056` (`stat_true=30536.08`, several of the 249 permutation nulls fall *below*
true, e.g. 30490.8, 30504.7, 30512.2, 30518.4, 30519.3, 30526.4) — i.e. **not significant at
α=0.05** when total-counts is whitened in log1p space, in contrast to `p=0.000999` when whitened
in raw space. This raw-vs-log1p sensitivity for the *same* covariate is itself worth noting if
revisiting this analysis — the hypoxia-panel signal's significance here is not robust to that
choice, unlike the tumor-proportion-only vs. tumor-proportion+total-counts comparison (which was
robust).

**Root cause of the raw-vs-log1p gap, and which one to trust (2026-08-07 follow-up):
`log1p_total_counts` is the correct choice, and the `p=0.000999` raw-whitened result should be
treated as likely still confounded, not as evidence the hypoxia signal "survives" whitening.**
`covariate_loss_difference.py`'s `h(d, n)` uses a **linear** decoder (`nn.Linear`) on
`[isodepth_latent, n]` fit against already log/CPM-normalized, per-gene-z-scored expression — i.e.
the natural depth→expression relationship in this space is *log-linear* (that's the point of
CPM/log1p normalization), so a linear term should be fit on **log** depth, not raw depth. Checked
the actual per-cell covariate values saved in each run's
`artifacts.dataset_meta.covariate_whitening_values` (same 125 cells, same order):
- raw `total_counts`: range 2114–95899 (45×), skew=0.83 (right-skewed), CV=0.78.
- `log1p_total_counts`: range 7.66–11.47, skew=-0.27 (~symmetric), CV=0.09.
- Pearson(raw, log1p) = **0.927** (vs. Spearman = 1.0, since log1p is monotonic) — the gap
  (1 − 0.927² ≈ 14% of variance) is exactly the nonlinear part of the depth signal a straight line
  in *raw* counts cannot capture but a straight line in *log* counts can.

Mechanism: total_counts is strongly spatially smooth here (`rho(isodepth, total_counts)=0.94`,
noted above). A linear-in-raw-counts whitening term only partially removes that smooth depth
signal; the flexible spatial encoder (`d`) can then absorb the ~14% leftover as if it were genuine
spatial signal (permuted nulls can't, since their coordinates are shuffled) — artificially widening
the true-vs-null gap and inflating significance to `p=0.000999`. Whitening with `log1p_total_counts`
is well-specified for the actual log-linear depth effect and removes much more of that smooth
confound. **Conclusion: prefer `log1p_total_counts` (or any log/CPM-consistent transform) over raw
`total_counts` whenever adding a covariate to `loss-difference` whitening on log-normalized
expression — using the raw scale under-whitens the confound and can manufacture spurious
significance.** The raw-`total_counts` config/results were deleted for this reason (see above);
**only the `log1p_total_counts` version below should be used going forward.**

**IMPORTANT further finding (2026-08-07, later same day): for this `log1p_total_counts` run,
`stat_true` (the trained true model's loss) is NOT invariant to `test.n_perms`, even with a fixed
`test.seed=42` — and the drift is large enough to flip the significance conclusion.** Re-ran the
identical config
(`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_hypoxia_genes_tumor_logcounts.json`,
recreated from the result.json's saved config since the original config file had been edited
in-place into the now-deleted totalcounts_whitened version — same 125 cells, gt0.7, spatial_crop,
200-gene panel, `covariate_whitening_obs_key: ["calicost_tumor_proportion", "log1p_total_counts"]`)
at three different `n_perms`, holding everything else fixed:

| `n_perms` | `stat_true` | `p_value` |
|---|---|---|
| 249 (original) | 30536.08 | 0.056 |
| 500 (diagnostic, deleted after check) | 30530.91 | 0.018 |
| 999 (current, kept — see run dir below) | 30495.26 | **0.001** |

`stat_true` is **exactly bit-identical on repeat runs at a fixed `n_perms`** (verified: reran
`n_perms=999` twice, got `30495.255626719514` both times) — so this isn't GPU/CUDA
nondeterminism in the usual sense. But it drifts **monotonically downward as `n_perms` increases**,
and the drift (~41 loss units between 249 and 999 perms) is comparable in size to the entire width
of the permutation-null distribution itself (nulls span roughly 30490–30650, std≈32) — i.e. changing
`n_perms` alone, with no other change, is enough to flip this borderline test from "not significant"
to "highly significant." Root cause (from reading `methods/trainers/isodepth.py`
`train_parallel_isodepth_model`): the true model (slot 0) and all `n_perms` permuted-coordinate
models are trained **jointly in one batched tensor** of size `(n_perms+1)*n_reruns` parallel models;
per-model gradients are scale-corrected by an `active_count` divisor
(`methods/trainers/isodepth.py` ~line 1991/2020) intended to make the per-model effective step size
batch-size-invariant, but empirically this doesn't fully cancel — total batch size still measurably
changes the true model's 1000-epoch SGD/Adam trajectory (almost certainly via floating-point
non-associativity in the batched loss reduction, e.g. GPU batched-matmul/reduction kernels choosing
different accumulation order for different batch shapes; a tiny per-step perturbation, then
chaotically amplified over 1000 epochs and by the "best-of-10-reruns" `argmin` selection, which can
flip which rerun's basin of attraction wins).

**Practical implication: `n_perms` in this codebase's `parallel_permutation`/`loss-difference`
pipeline is not a pure "more precision, same test" knob — it can silently perturb the true
statistic too, for borderline cases.** Do not treat a single run's p-value at one `n_perms` as
final for a result this close to the null distribution's edge without checking sensitivity to
`n_perms` (or ideally to `test.seed`) the way this note did. The `p=0.001` result at `n_perms=999`
(kept, at `results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p7_cropy_hypoxia_genes_tumor_logcounts/`)
should be reported alongside this caveat, not as a clean, final answer — this specific
clone/covariate/gene-panel test is numerically borderline and its exact significance level is not
robust to incidental training-batch-size effects. If this matters for a real conclusion, the
next step should be characterizing the spread across several `test.seed` values at fixed
`n_perms`, or (better, a real code fix) decoupling the true model's training from the permutation
batch so `stat_true` is computed identically regardless of `n_perms`.

**Follow-up (2026-08-08): re-ran the same config at `n_perms=999` with `test.n_reruns` bumped
10→30 (per explicit request, for a more robust best-of-N loss estimate per slot), overwriting the
run dir in place (same `run_name`, `loss_diff_clone2_linear_gt0p7_cropy_hypoxia_genes_tumor_logcounts`).
Result: `p_value=0.001` again (true loss below all 999 nulls), but `stat_true` moved again —
**30590.53** at 30 reruns vs. **30495.26** at 10 reruns (both at the same `n_perms=999`,
`test.seed=42`). This is a second, independent confirmation of the batch-size-sensitivity finding
above: `stat_true` is not just sensitive to `n_perms` but also to `n_reruns` (both change the
total batched-model tensor size `(n_perms+1)*n_reruns`), and the direction isn't the "obvious"
one either — more reruns (more best-of-N candidates for the true model) made `stat_true` *worse*
(higher loss), not better, which only makes sense if different `n_reruns` values put the true
model through a measurably different SGD trajectory rather than just adding more independent
draws to the same argmin. The significance conclusion (p≈0.001, highly significant) has now held
at both 10 and 30 reruns, which is reassuring, but the underlying instability documented above is
still unresolved — do not read `stat_true`'s exact value as meaningful across different
`n_perms`/`n_reruns` settings, only the qualitative significant/not-significant call, and even
that should be treated cautiously given the `n_perms=249` (`p=0.056`) vs. `n_perms=999`
(`p=0.001`) flip found earlier.

Also deleted (per request) the two older, lower-rerun/lower-perm sibling result dirs from the
gt0.5/gt0.7 tumor-proportion-only (no total-counts whitening) hypoxia sweep —
`loss_diff_clone2_linear_gt0p5_cropy_hypoxia_genes` and `loss_diff_clone2_linear_gt0p7_cropy_hypoxia_genes`
(both 249 perms / 10 reruns, `p_value=0.004`) — since they're superseded by the higher-perm/rerun,
total-counts-whitened `..._tumor_logcounts` run above. Their configs
(`HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_hypoxia_genes.json`, still present) remain
reproducible if needed, but the result dirs themselves are gone.

### Why `loss-difference` covariate whitening needs the covariate on a compatible (log/CPM) scale, and a structural limitation it has even then

`methods/covariate_loss_difference.py` reveals the mechanism precisely: `n` (the covariate,
z-scored) is concatenated as an **extra input feature to the decoder** alongside the fitted
spatial latent `d`, i.e. it trains `h(d, n)`; it is *not* a residualization/orthogonalization step
that removes `n`'s contribution from `d` or from the data before fitting `d`. The permutation test
asks "does the *true* per-cell position `d(x,y)` predict expression better than `n` alone, vs.
permuted positions (with each cell's own `n` kept fixed)?" A significant p-value means position
carries information beyond each cell's own scalar covariate value — it does **not** by itself mean
`d` ends up uncorrelated with `n`, and it does **not** distinguish "that extra positional info is
genuinely about the biology of interest" from "that extra positional info is just a spatially
*smoothed/denoised* version of the same confound `n` measures noisily per cell" (nearby cells
share similar library size/tissue quality for reasons unrelated to any single cell's individual
`n` value, so position can recover a cleaner version of the `n` gradient than `n` itself supplies).

Checked this directly for the `..._tumor_logcounts` run (n=125, 200-gene hypoxia panel, 999
perms/30 reruns, whitening `["calicost_tumor_proportion","log1p_total_counts"]`) by reconstructing
the exact 125 training cells (coordinate-match `S`→raw h5ad via `coord_mean`/`coord_std`, same
technique as the GSEA recovery script, verified `max_abs_diff=0.0` against saved NPZ `A`) and
correlating the raw obs columns against the fitted `true_isodepth`:
- `total_counts` / `log1p_total_counts`: **rho=+0.863, p=3e-38** — barely reduced from the
  *unwhitened* gt0.7 value of 0.94 noted earlier in this file.
- `pct_mt` (never whitened): rho=-0.682, p=2e-18 — essentially unchanged from the unwhitened
  -0.62/-0.72 values noted earlier.
- `calicost_tumor_proportion` (whitened): rho=+0.294, p=9e-4 — this one *did* drop substantially
  (was 0.28-0.47 pre-whitening at similar magnitude, so less informative a check, but at least
  didn't get *stronger*).

So whitening `log1p_total_counts` did **not** decouple the fitted isodepth from total counts in
any strong sense — it's still one of the best single predictors of the latent field. Per the
mechanism above, this is not necessarily a bug, but it means **the existence-test significance
found so far cannot yet be attributed to hypoxia biology specifically vs. "any spatially-smooth,
total-counts/tissue-quality-correlated axis that a coherent ~180-200-gene panel would pick up."**

**Per-gene correlation sanity check (also done 2026-08-08, cheap — no retraining, just
`scipy.stats.spearmanr` of each of the 183 surviving panel genes' preprocessed expression against
`true_isodepth`):** mixed evidence for genuine hypoxia specificity. Canonical HIF-1 targets
`P4HA1` (rho=+0.37), `P4HA2` (+0.38), `PFKFB3` (+0.34), `PDK1` (+0.33), `SLC2A1`/GLUT1 (+0.28) are
individually significant and correctly signed (up with "deeper" isodepth), which is reassuring.
But the single *strongest* correlated genes in the panel (`IRS2` +0.49, `AMPD3` +0.47, `CASP6`
+0.45, `TES` +0.45, `CCNG2` +0.44) are not the most iconic hypoxia markers, and several textbook
HIF targets are weak/non-significant here (`VEGFA` +0.14 n.s., `LDHA` +0.03 n.s., `ANGPTL4` +0.02
n.s., `NDRG1` +0.09 n.s., `BNIP3L` +0.11 n.s.; `CA9` didn't survive per-clone `min_cells_per_gene`
filtering). So: some real hypoxia signal is present, but it's not an unambiguous, clean hypoxia
transcriptional program driving the result — consistent with the total-counts-leakage concern
above.

**Recommended next step (not yet run): random-gene-panel specificity control.** Run the identical
pipeline (same clone, same `n_cells`, same covariate whitening, same `n_perms`/`n_reruns`) on many
(e.g. 20-50) random draws of ~180-200 genes from the same transcriptome, and compare the
hypoxia panel's test statistic/p-value against that empirical null. If most random panels also
come out significant at similar p-values, the test is not hypoxia-specific — it's likely just
detecting the generic total-counts/tissue-architecture smooth gradient that a random panel of this
size would also partially track. If the hypoxia panel is a clear outlier relative to the random-
panel distribution, that's real evidence for hypoxia specificity. This is the most decisive
still-missing control and should be prioritized over further tweaking of the whitening covariate
scale.

### Random-gene-panel specificity control — RUN (2026-08-08): hypoxia is NOT specific, and the test is badly miscalibrated on this clone

Implemented as a general reusable experiment type, `experiments/studies/random_gene_panel_null/`
(`lib.py` + `sweep.py`, run via
`python -m experiments.studies.random_gene_panel_null.sweep --spec <spec.json>`), driven by
`configs/experiments/hypoxia_panel_specificity_study.json` (base config =
`..._hypoxia_genes_tumor_logcounts.json`, 30 random 200-gene panels seeded 0-29 sampled from the
same clone-2/gt0.7/cropy eligible gene universe, same `n_perms=249`/`n_reruns=30` for every panel
including the target, so target-vs-random comparisons are apples-to-apples). Outputs at
`results/experiments/hypoxia_panel_specificity/` (`manifest.json`, `runs/*/`,
`analysis/{analysis_summary.json,per_run_results.csv,analysis_warnings.csv,
target_vs_random_pvalue_histogram.png,target_vs_random_stat_true_histogram.png}`).

**Result: hypoxia panel is unremarkable relative to random panels.** Target `p_value=0.012`
(different from the `p=0.001`/`p=0.056` values quoted above because those used `n_perms=999`; this
study fixes `n_perms=249` for a fair comparison to the randoms) ranks only **10th out of 31**
(23rd percentile) — 9 of the 30 random panels have p-values as small or smaller
(`random_p_value_mean=0.055`, range 0.004-0.284). Not an outlier by any reasonable reading.

**Bigger finding: the test is severely miscalibrated on this clone/whitening setup, independent of
hypoxia.** Counting `analysis/per_run_results.csv` directly: **20 of 30 (67%) random 200-gene
panels are "significant" at α=0.05**, vs. the ~5% expected under a properly calibrated null. I.e.
almost *any* random ~200-gene panel comes out "significant" on this clone under
`covariate_whitening=[calicost_tumor_proportion, log1p_total_counts]` — strong direct confirmation
that the earlier-documented residual total-counts/tissue-quality confound (rho with fitted
isodepth still ~0.86-0.94 post-whitening) is being picked up by the spatial encoder for
essentially any spatially-coherent gene panel, not something specific to hypoxia genes.
**Conclusion: the earlier hypoxia-panel significance (`p=0.001`-`0.056` depending on `n_perms`)
should now be read as very likely a byproduct of this uncontrolled confound, not evidence of a
hypoxia-specific spatial signal** — and, more importantly, this miscalibration means *no*
gene-panel-restricted existence-test conclusion on this clone/whitening config should be trusted
without a similar random-panel control, until the whitening is improved (e.g. a nonlinear/spline
term on `log1p_total_counts`, or adding `pct_mt`) or the test's significance threshold is
recalibrated against an empirical random-panel null instead of a nominal α=0.05.

**Literature context (2026-08-08, web research, no new runs): the total-counts/hypoxia confound
here is a recognized, actively-researched open problem in spatial transcriptomics, not a flaw
specific to this pipeline — and the field's answer requires data this cohort doesn't have.**
Key findings, for whoever revisits this:
- [Bhuva et al., *Genome Biology* 2024, "Library size confounds biology in spatial
  transcriptomics data"](https://doi.org/10.1186/s13059-024-03241-7): across 25 datasets/4
  technologies, total-count differences genuinely reflect tissue architecture/biology, not just
  technical noise; standard (scRNA-seq-style) library-size normalization measurably destroys real
  spatial-domain signal. I.e. whitening `log1p_total_counts` here is a mainstream but
  known-to-be-blunt approach that risks removing real biology along with the confound — directly
  relevant to whether the significance drop under whitening (see `p=0.056` `log1p` vs `p=0.000999`
  raw section above) means "confound removed" or "biology removed."
- [SpaNorm, *Genome Biology* 2025](https://doi.org/10.1186/s13059-025-03565-y): the field's proposed
  fix is a spatial GLM (thin-plate splines) that decomposes each gene's spatially-smooth variation
  into a library-size-associated part and a library-size-*independent* part, discarding only the
  former — a strictly more principled version of this project's single linear whitening covariate.
  Worth considering if this line of work continues.
- Standard SVG tools ([SpatialDE](https://pmc.ncbi.nlm.nih.gov/articles/PMC6350895/),
  [SPARK-X](https://doi.org/10.1186/s13059-021-02404-0)) do also regress out log total
  counts/covariates before testing, so this project's general approach is mainstream, just cruder.
- For hypoxia scoring specifically: naive bulk-style scoring (ssGSEA/GSVA — summed/averaged raw
  expression, which is effectively what feeding raw per-gene z-scored expression into the decoder
  loss amounts to) is known to be highly sensitive to per-cell/per-spot detection-rate/dropout
  differences ([eLife 2022](https://elifesciences.org/articles/71994)) — the same failure mode as
  the total-counts confound found here. Fix used in that literature: dropout-robust, rank-based
  module scores (AUCell, JASMINE, SCSE) instead of raw expression level. A pan-cancer benchmark of
  70 hypoxia signatures × 14 scoring methods ([Cell Genomics
  2025](https://doi.org/10.1016/j.xgen.2025.100764)) found signature/score choice massively changes
  conclusions; naive mean/ssGSEA scoring performs near chance in places, robust central-tendency
  scores (interquartile mean, trimean) with Buffa/Ragnum signatures are current best practice.
  **Concrete suggested upgrade, not yet implemented: rerun the existence test using a dropout-robust
  hypoxia module score (e.g. AUCell) as the per-spot target instead of raw per-gene expression,**
  since that's specifically designed to resist the exact confound documented above.
- Orthogonal validation: across the literature on how hypoxia gradients are actually established
  (not just hypothesized) — e.g. pimonidazole IHC (pharmacological gold standard, requires
  pre-mortem injection, cannot be added retroactively), CA9/GLUT1/HIF-1α IHC, vessel-distance
  analysis against CD31/CD34-stained vasculature — none treat a transcriptomic signal alone as
  sufficient; all require an independent, non-transcriptomic marker. **This dataset has none of
  these.** Confirmed via `adata.uns['source']`/`adata.uns['spatial'][...]['metadata']`: this h5ad
  is processed **HTAN WUSTL 10X Visium** expression data (`source: "HTAN WUSTL Level 3 10X
  Visium"`) with CalicoST CNV calls layered on (`calicost: "Zenodo 10.5281/zenodo.14175627"`);
  `adata.uns['spatial']['HT306P1-S1H1Fc2U1Z1Bs1']` only has `scalefactors`/`metadata`, **no
  embedded tissue image** (checked directly, `'images' not in` that dict). The original HTAN WUSTL
  Visium submission likely has a paired H&E image in the HTAN data portal (external to this
  repo/h5ad) that could serve as a (weaker, but literature-standard) orthogonal sanity check —
  e.g. does the isodepth/total-counts gradient spatially coincide with necrotic-looking tissue
  morphology — but this hasn't been pulled or checked. **Bottom-line takeaway for any future
  write-up: no purely computational analysis on this cohort (however well confound-controlled) can
  fully confirm "real hypoxia" without either that external H&E check or accepting the field-wide
  caveat that transcriptomic hypoxia calls without an orthogonal marker are hypotheses, not
  confirmed findings.**

**Pitfall found in the study's own `stat_true`-based comparison — do not reuse as-is.**
`lib.py::compute_target_rank` was also applied to raw `stat_true` (giving
`target_stat_true_percentile_among_random=0.0`, i.e. "looks like a huge outlier") but **this
comparison is invalid and should be ignored**: `stat_true` is summed NLL over each panel's own
(different) genes — target has 183 surviving genes with likely lower intrinsic
variance/predictability than a typical random 200-gene draw, so its absolute loss scale isn't
comparable to other panels' at all (target's `stat_true≈30539` vs. random panels tightly clustered
`33188-33649` — a much bigger gap than the ~460-unit spread among randoms, which is about gene
identity/count, not spatial-signal strength). The only valid cross-panel-comparable quantity is
each panel's own within-panel p-value (computed against its own gene-matched permutation null,
i.e. gene identity held fixed) — that's the one showing no specificity. If reusing this study
pattern elsewhere, drop or fix the `stat_true` rank/percentile metric, or at minimum flag it as
not meaningful when panels don't share identical genes.

**Random-gene-panel specificity control (RUN 2026-08-08):** Implemented as a reusable study under
`experiments/studies/random_gene_panel_null/` (spec + sweep + analysis), driven by
`configs/experiments/hypoxia_panel_specificity_study.json`. Ran 31 conditions (1 target HALLMARK
hypoxia panel + 30 random 200-gene panels) with identical clone-2/gt0.7/tumor+logcounts-whitened
settings at **`n_perms=249`, `n_reruns=30`**, drawing random genes from the 14,606-gene universe
passing `min_cells_per_gene>=3` within the same obs subset (pre-crop). Outputs under
`results/experiments/hypoxia_panel_specificity/` (`manifest.json`, `runs/*`, `analysis/*`).

Headline comparison (`analysis/analysis_summary.json`):
| metric | hypoxia target | random panels (n=30) |
|---|---|---|
| `p_value` | **0.012** | mean 0.055, range [0.004, 0.284] |
| `stat_true` | **30539.1** | mean 33478.3, range [33187.6, 33649.0] |
| `n_genes_surviving` | 183 | mean ~199.9 (mostly 200) |

- **`stat_true` specificity: strong.** Hypoxia panel is rank **1/30** (lowest/best loss by a wide
  margin — ~2600 loss units below the best random panel, vs random-panel std ≈101). This is the
  clearest evidence so far that the hypoxia gene set captures spatial structure better than a
  typical 200-gene draw, even though it ends up with fewer surviving genes (183 vs ~200).
- **`p_value` specificity: weak / inconclusive.** Hypoxia `p=0.012` is only rank **10/30** among
  random panels (23rd percentile) — **9 random panels were equally or more significant** (several
  at `p=0.004`). So at this panel size, generic random gene sets often also yield significant
  existence tests; the hypoxia panel is not a clear p-value outlier.
- **Interpretation:** The hypoxia panel appears genuinely better at *fitting* a spatial axis
  (`stat_true` outlier), but the permutation-test *significance* is not unique to hypoxia — many
  random 200-gene panels also reject the null at comparable or stronger p-values. This supports
  treating the overall existence signal partly as a generic "coherent gene panel + spatial
  smoothness" phenomenon, while still leaving room for hypoxia-specific biology in the much-stronger
  absolute fit. The surviving-gene-count mismatch (183 vs 200) is a minor confound for direct
  loss/p-value comparison but if anything makes the hypoxia `stat_true` advantage *more*
  impressive (fewer genes, much lower loss).
- **Re-run command:** `python -m experiments.studies.random_gene_panel_null.sweep --spec
  configs/experiments/hypoxia_panel_specificity_study.json` then `python -m
  experiments.studies.random_gene_panel_null.analysis --spec
  configs/experiments/hypoxia_panel_specificity_study.json`.

**Hallmark pathway sweep (RUN 2026-08-08):** Implemented as
`experiments/studies/pathway_panel_sweep/` (spec + sweep + analysis), driven by
`configs/experiments/hallmark_pathway_sweep_clone2_study.json`. Ran all **50 MSigDB Hallmark**
pathways as `data.gene_list` panels with the same clone-2/gt0.7/tumor+logcounts-whitened settings
at **`n_perms=249`, `n_reruns=30`**. Outputs under
`results/experiments/hallmark_pathway_sweep_clone2/` (`manifest.json`, `runs/*`, `analysis/*`).

Headline (`analysis/analysis_summary.json`):
- **35/50 pathways significant** at α=0.05 (70%) — existence signal is common across Hallmark panels.
- **HALLMARK_HYPOXIA:** p=0.012 (significant), stat_true=30539, 183/200 genes surviving.
  Rank **29/50 by p-value** and **39/50 by stat_true** (lower is better) — not a standout among
  Hallmark pathways on either metric.
- **15 non-significant pathways:** APICAL_SURFACE, HEDGEHOG_SIGNALING, MYC_TARGETS_V2,
  NOTCH_SIGNALING, WNT_BETA_CATENIN_SIGNALING, CHOLESTEROL_HOMEOSTASIS, PANCREAS_BETA_CELLS,
  SPERMATOGENESIS, IL6_JAK_STAT3_SIGNALING, TGF_BETA_SIGNALING, UV_RESPONSE_UP,
  BILE_ACID_METABOLISM, APOPTOSIS, PI3K_AKT_MTOR_SIGNALING, KRAS_SIGNALING_UP.
- Many significant pathways share the floor p=0.004 (249 perms) — tie at minimum achievable p.
- **Re-run:** `python -m experiments.studies.pathway_panel_sweep.sweep --spec
  configs/experiments/hallmark_pathway_sweep_clone2_study.json` then `python -m
  experiments.studies.pathway_panel_sweep.analysis --spec
  configs/experiments/hallmark_pathway_sweep_clone2_study.json`.

