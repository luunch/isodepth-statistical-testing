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

Python env for this repo: `/weka/home/jhu/ajain71/miniforge3/envs/isodepth_env/bin/python`
(system `python3`/`python` do not have numpy etc.). **Note (2026-08-23):** this path has moved /
is unstable across sessions — it was previously `/weka/home/ajain71/...` (missing `jhu/`, now
404s, same stale-path pattern as the manifest bug below) and mid-session on 2026-08-23 a
*different* previously-working env at
`/weka/scratch/jhu/uchitra1/users/ajain71/miniforge3/envs/isodepth_env` (used earlier in this same
session, confirmed to have `sklearn` installed) disappeared entirely (`ls` on its parent dir no
longer showed a `miniforge3` subdirectory) while `/weka/home/jhu/ajain71/miniforge3/envs/isodepth_env`
kept working throughout — always run `mamba env list` first to get the current live path(s) rather
than trusting any hardcoded path (including this one) if a `mamba run -p <path>` invocation fails
with "prefix does not exist".

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
`python -m scripts.posthoc.gsea_loss_diff_clone2_linear --gmt data/gmt/h.all.v2026.1.Hs.symbols.gmt`),
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
Gene-set file used (updated 2026-08-10): `data/gmt/h.all.v2026.1.Hs.symbols.gmt` (MSigDB
Hallmark **2026.1.Hs**, Jan 2026, 50 sets). Prior CalicoST GSEA used
`data/gmt/h.all.v2024.1.Hs.symbols.gmt` (still kept under `data/gmt/` for reference); all three
CalicoST `gsea_isodepth/` outputs were regenerated with 2026.1, with pre-update CSVs archived
as `*_v2024.1.csv` alongside. Hallmark 2026.1 vs 2024.1 differs by **only 4 HGNC symbol
renames** (same 50 pathway names): `SLC22A18→SLC67A1` (bile-acid + UV-response-DN),
`PRPF4B→PRP4K` (G2M), `CENPJ→CPAP` (mitotic spindle). Default GSEA params match sibling runs:
`min_size=15, max_size=500, n_permutations=250, weight=1.0, score_method=spearman, seed=0`.

**Takeaway from the 2026.1 rerun:** no meaningful biological change. The only true gene-set
membership effect in any CalicoST GSEA was in `loss_diff_clone2_linear_gt0p5_cropy`, where
`SLC22A18` was in the HVG background so bile-acid / UV-response-DN each lost 1 overlapping gene;
both stayed non-significant. The other two runs (`no_mtribo_stress`, legacy
`loss_diff_clone2_linear`) had zero overlap change for the remapped symbols (renamed genes not
in those HVG lists), so GSEA was bit-identical under a controlled same-prerank GMT swap.
Apparent NES/rank churn vs the archived Aug-6 CSVs is mostly shared-RNG null-permutation
sensitivity in `_gsea_preranked` (one RNG stream across pathways; changing an earlier pathway's
overlap `k` shifts later NES/p), not biology — enrichment scores (ES) for unchanged pathways
are identical. Biological headline unchanged: interferon / EMT / myogenesis / complement
anti-correlated with isodepth; DNA repair positively correlated.

Result (2026-08-07, confirmed unchanged after 2026.1 GMT swap 2026-08-10): 36/50 Hallmark
pathways tested (14 excluded by min/max overlap size); top hits by q-value are
interferon-gamma response, EMT, myogenesis, complement (all NES < -1.5, i.e. anti-correlated
with isodepth) and DNA repair (NES ~+1.65, positively correlated) — all q < 0.03. This is a
distinct/independent gene ranking from the sibling `..._gt0p5_cropy` run (different cell
subset/crop), so don't assume the two GSEA result sets should match.

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

**Literature grounding for the overall method (2026-08-08, web research, no code changes): confirms
the base model, finds a close competitor, and frames the core structural critique.** Base model is
[GASTON, Chitra/Raphael group, *Nature Methods*
2025](https://doi.org/10.1038/s41592-024-02503-3) ("isodepth" = GASTON's term for the learned 1-D
topographic coordinate); GASTON's own paper validates biologically, not via a calibrated
permutation existence test — this repo's `parallel_permutation`/`covariate_loss_difference`/
recursive-peeling framework is filling a real, currently-unaddressed gap around it. Closest direct
competitor found: [LSGI, *Genome Biology*
2025](https://link.springer.com/article/10.1186/s13059-025-03716-1) — spatial gradient detection
via NMF loadings regressed on coordinates with an empirically-tuned R² threshold (cheap, but a
heuristic, not a calibrated p-value). Non-spatial analogue: pseudotime trajectory-existence tests
([EMST-permutation](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04875-9),
[tree-dimension test with closed-form log-normal
null](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009829)) — these
deliberately use graph statistics with either permutation or closed-form nulls specifically to
avoid retraining an expensive model per null draw, unlike this repo's per-permutation-per-rerun
neural-net retraining design. **Framing for future work**: this repo's flexibility (joint nonlinear
multi-gene modeling) is a real advantage over SVG tools (SpatialDE/SPARK-X, per-gene only) and LSGI
(linear/NMF only), but its per-permutation neural-net-retraining design is exactly why it's exposed
to the calibration/reproducibility problems documented above (67% random-panel false-positive rate;
`stat_true` drifting with `n_perms`/`n_reruns`) that closed-form/asymptotic-null methods don't have
— matching those methods' calibration guarantees by brute force (more perms/reruns) is expensive
and, per the batch-size finding, may not even converge to a stable value. Prioritized fix list if
this is revisited: (1) simple — make an empirical calibration check (random-panel or permuted-null
false-positive rate) a mandatory, automatic companion to every reported p-value, and drop/guard the
invalid cross-panel `stat_true` comparison in `compute_target_rank`; (2) moderate but well-scoped —
decouple true-model training from permutation-batch size in `methods/trainers/isodepth.py` (root
cause already diagnosed above); (3) hard, open research problem, not a patch — redesign covariate
whitening as an explicit residualization/orthogonalization (e.g. SpaNorm-style spatial-GLM
decomposition of the confound before testing) instead of decoder-concatenation, since the latter
structurally cannot distinguish genuine extra positional signal from a spatially-smoothed copy of
the whitened confound (mechanism already documented above, independently confirmed as a known
open problem in [Bhuva et al. 2024](https://doi.org/10.1186/s13059-024-03241-7) /
[SpaNorm 2025](https://doi.org/10.1186/s13059-025-03565-y)).

**Candidate positive/negative-control benchmark datasets for validating this method against known
ground truth (2026-08-08, web research, no code changes yet — not integrated).** Since neither
HT306P1 nor the broader project has an independent (histology/IHC) ground truth for any tested
pathway gradient, identified external public datasets with histology-validated positive controls
(and matched same-cohort negative controls) that could be used to sanity-check the existence-test
pipeline before trusting it on data with unknown ground truth:
- **Hypoxia/necrosis in GBM** (best tumor-pathology positive control): pathologists have long
  annotated "pseudopalisading cells around necrosis" (PAN) as a classic hypoxic niche.
  [Ravi et al. 2022 GBM spatial multi-omics (Dryad, 10x Visium)](https://datadryad.org/dataset/doi:10.5061/dryad.h70rxwdmj)
  has PAN/microvascular-proliferation pathologist annotations cross-validated against Ivy GAP LCM
  hypoxia signatures. [Greenwald et al., *Cell* 2024](https://doi.org/10.1016/j.cell.2024.03.029)
  builds a 5-layer spatial model on top of Ravi's 13 Visium samples matched to histopathology
  ([Zenodo data](https://zenodo.org/records/12624108)). A [2026 *Nat Commun* GBM
  atlas](https://link.springer.com/article/10.1038/s41467-026-69716-2) explicitly includes a
  **normal-brain sample as a built-in negative control** alongside PAN-annotated tumor
  samples — same paper/platform, ideal for a matched true-positive/true-negative pair.
- **Liver zonation** (best non-tumor, most rigorously validated oxygen-gradient positive control):
  [Halpern et al., *Nature* 2017](https://www.nature.com/articles/nature21065) — periportal
  (oxygenated) vs. pericentral (hypoxic) zonation validated via smFISH (orthogonal to
  transcriptomics) AND cross-checked against genes independently known to shift under
  *experimentally induced* chronic hypoxia (a causal anchor, stronger than spatial correlation
  alone). Not native Visium `(x,y)` coordinates (scRNA-seq + reconstructed lobule position), would
  need adaptation to this repo's `DatasetBundle` schema.
- **EMT/ECM gradient at tumor invasive front** (reproducible across cancer types, IHC/RNAscope
  cross-validated in independent cohorts): [follicular thyroid carcinoma, 2024](https://doi.org/10.1007/s12022-024-09798-0)
  (POSTN/DPYSL3 IHC-validated in a held-out cohort), [HNSCC tumor budding
  signature, 2026](https://doi.org/10.1186/s13073-026-01612-2) (28-gene signature, AUC=0.97 in
  SRT, validated against bulk TCGA-HNSC), [CRC invasive front, 2025](https://doi.org/10.1038/s42003-025-08799-x).
- **Practical next step, not yet done**: check whether the broader HTAN WUSTL cohort that
  `HT306P1`/`HT268B1` belong to includes any adjacent-normal tissue sections (same or sibling
  patients) — lowest-friction option since it needs no new external data-source integration, just
  pulling another sample already in the same collection, to get a same-cohort negative control.
- **Suggested validation plan**: run the existence test + random-gene-panel specificity control
  (pattern already built in `experiments/studies/random_gene_panel_null/`) on GBM PAN-annotated
  spots for `HALLMARK_HYPOXIA` (expect significant + specific) vs. normal-cortex spots from the
  same paper (expect null), and similarly tumor-core vs. invasive-margin spots for an EMT gene set.
  Passing both true-positive and true-negative cases on external, histology-anchored data would be
  much stronger validation than anything achievable on `HT306P1` clone 2 alone, where the ground
  truth is fundamentally unknown (see hypoxia/total-counts confound sections above).

**Detailed, pre-registered validation protocol for hypoxia existence-testing (designed 2026-08-08,
not yet executed).** Full plan, in case this is picked up later — see chat transcript for full
rationale, condensed here for reuse:
- **Data**: 2-4 GBM Visium sections from
  [Ravi et al. 2022 (Dryad)](https://datadryad.org/dataset/doi:10.5061/dryad.h70rxwdmj) or the
  [2026 *Nat Commun* GBM atlas](https://link.springer.com/article/10.1038/s41467-026-69716-2),
  ≥2 different patients with PAN (pseudopalisading-cells-around-necrosis) annotations, plus the
  atlas's normal-brain section. Needs a new `scripts/data_prep/` loader (mirroring
  `add_qc_obs_columns.py`) to harmonize into this repo's `DatasetBundle` schema with QC obs columns
  (`total_counts`/`pct_mt`/`log1p_total_counts`) and the pathologist region label preserved.
- **Three spot groups per section**: (1) PAN spots = positive; (2) leading-edge/cellular-tumor
  spots from the *same section* = matched within-tumor negative; (3) normal-brain section =
  tissue-level negative. Gene panel: reuse existing `HALLMARK_HYPOXIA` 200-gene `gene_list`.
- **Core runs**: `parallel_permutation` existence test per group per patient, at one **fixed**
  `n_perms`/`n_reruns` across every run (target AND randoms) to avoid the documented
  `stat_true`-drift confound; each under both no-whitening and `log1p_total_counts`-whitened
  conditions. Plus the existing `experiments/studies/random_gene_panel_null/` specificity sweep
  (≥30 random 200-gene panels) per group/condition/patient.
- **Pre-registered pass/fail criteria (fix before running)**: PAN should be significant AND an
  outlier vs. its own random-panel null, replicated across ≥2 patients; both negative groups should
  be non-significant or non-outlier, AND their random-panel false-positive rate should be near
  nominal (~5-10%) — if negative-control FPR is also inflated (like HT306P1's 67%), that shows the
  miscalibration is a property of the method, not of HT306P1 specifically.
- **Decisive experiment (do this early)**: check whether PAN's positive-control significance
  *survives* `log1p_total_counts` whitening. If yes → real evidence the method can detect true
  hypoxia net of a total-counts confound (raises confidence in HT306P1's own post-whitening
  result). If no, even in histology-confirmed-hypoxic tissue → evidence that this whitening
  approach is too aggressive for genuine hypoxia signals *in general*, arguing for the
  SpaNorm/mediator-aware redesign discussed earlier as a blanket fix, not a per-dataset judgment.
- **Spatial concordance**: correlate fitted isodepth against the histological PAN boundary directly
  (e.g. AUC/Spearman ρ vs. distance-to-PAN), not just the p-value.
- **Real-data bake-off**: run per-gene SPARK-X/SpatialDE + GSEA (reusing
  `postprocess_gsea_isodepth.py`) and an LSGI-style NMF+regression+R² proxy on the same 3 groups;
  compare correctness and wall-clock cost against this method.
- **Synthetic complement**: use existing `experiments/existence_sigma_sweep` synthetic
  infrastructure to simulate a panel where genes are individually weak but jointly coherent, plus
  an injected confound at controlled strength, swept across effect size/confound
  strength/sample-size (including small `n~125-195` to probe the earlier small-sample concern);
  compare calibration/power of this method vs. per-gene+GSEA vs. LSGI-proxy with *exact* known
  ground truth (the only step that can cleanly attribute any power/calibration gap to the method
  itself rather than real-data ground-truth uncertainty).
- **Practical**: prototype on 1 patient/1 whitening condition first to estimate wall-clock cost
  before committing to the full matrix (single runs took ~28s at `n_perms=249`/`n_reruns=30` in the
  earlier hypoxia-panel-specificity study; this plan's `n_perms=999` runs will be substantially
  slower, and the full matrix is 3 groups × 2 whitening conditions × ≥2 patients × ~31 runs each).

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

**Hallmark pathway sweep — HT268B1 clone 1 comparison (RUN 2026-08-08):** Same 50-pathway sweep on
**HT268B1 clone 1** (426 cells, gt0.7, tumor+logcounts whitening, no spatial crop) — chosen because
the whole-transcriptome existence test on this section was borderline (**p=0.052** with 3000 HVG).
Spec: `configs/experiments/hallmark_pathway_sweep_ht268b1_clone1_study.json`; outputs under
`results/experiments/hallmark_pathway_sweep_ht268b1_clone1/`.

Comparison vs HT306P1 clone 2 (125 cells, y-crop):
| | clone 2 (HT306P1) | clone 1 (HT268B1) |
|---|---|---|
| Significant pathways | 35/50 (70%) | 28/50 (56%) |
| HALLMARK_HYPOXIA p | 0.012 | 0.004 |
| Significant in both | — | 22 pathways |
| Non-significant in both | — | 9 (WNT, NOTCH, HEDGEHOG, APICAL_SURFACE, TGFβ, etc.) |
| Sig only clone 2 | — | 13 (glycolysis, angiogenesis, G2M, adipogenesis, …) |
| Sig only HT268B1 | — | 6 (KRAS_UP, PI3K/AKT/mTOR, IL6/JAK/STAT3, apoptosis, …) |

Interpretation: HT268B1 is somewhat less pathway-positive overall, but still majority significant;
hypoxia is *more* significant there. The **9 pathways non-significant in both** clones are the
most credible null set; **22 significant in both** are the most reproducible tumor-gradient programs.

## Hallmark pathway sweep clone2 analysis refresh (2026-08-10)

Matched full 3000-HVG reference for the clone-2 pathway sweep (same cells as all
pathway runs: clone 2.0, gt0.7 tumor prop, y-crop 1.9, denoise 300um, whitening
`[calicost_tumor_proportion, log1p_total_counts]`, n=125, S bit-identical):
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_gt0p7_cropy_tumor_logcounts_3000hvg.json`
→ `results/calicost/.../loss_diff_clone2_linear_gt0p7_cropy_tumor_logcounts_3000hvg/`
(used `n_perms=99`, `n_reruns=10` because `(249+1)*30` OOM-chunk merge crashed when
`n_cells==chunk_slot_count` wrongly sliced `covariate_values`; merge now skips that
buffer in `methods/trainers/isodepth.py`). GSEA on this run under its `gsea_isodepth/`.

Pathway-sweep analysis updates (`experiments/studies/pathway_panel_sweep/analysis.py`):
- plot `stat_true / n_genes_surviving` (no hypoxia line / no "lower is better")
- plot BH q-values; `significant` now means `q < alpha` (34/50)
- isodepth Spearman matrix + spatial grid vs FULL_3000HVG (pathways sign-oriented)
Outputs under `results/experiments/hallmark_pathway_sweep_clone2/analysis/`.

Expression-power follow-up (2026-08-11): existence p tracks pathway size/counts more than
detection or reference-isodepth alignment. Strongest Spearman with p: mean pathway raw
counts/cell ρ≈−0.56, n_genes_surviving ρ≈−0.50. Scatter saved as
`analysis/pathway_expression_power_scatter.png` (also regenerated by
`pathway_expression_power.py` / `python -m experiments.studies.pathway_panel_sweep.expression_power`).

## Hallmark pathway sweep HT268B1 clone1 analysis refresh (2026-08-11)

Same analysis refresh as clone2. Spec now points
`reference_full_result_json` at the existing matched full-HVG run
`results/calicost/HT268B1-Th1K3Fc2U1Z1Bs1/loss_diff_clone1_gt0p7_linear_tumor_logcounts/`
(n=426, A has 2760 genes post-HVG/`min_cells`, S bit-identical to all pathway runs).
Re-ran `python -m experiments.studies.pathway_panel_sweep.analysis --spec
configs/experiments/hallmark_pathway_sweep_ht268b1_clone1_study.json`. Removed old
`*_with_hypoxia_marked.png` plots. Under BH q<0.05: **22/50** significant (was 28/50
on raw p); hypoxia still significant (p=0.004). New outputs match clone2:
`hallmark_qvalue_distribution.png`, `hallmark_stat_true_per_gene_distribution.png`,
`isodepth_spearman_matrix.{csv,png}`, `pathway_isodepths_grid.png`.

**Isodepth Spearman matrix = significant pathways only (2026-08-11):**
`pathway_panel_sweep/analysis.py` now writes `isodepth_spearman_matrix.{csv,png}` with
`FULL_3000HVG` + pathways where `q < alpha` only (non-significant dropped from the matrix;
`spearman_vs_full` and the full `pathway_isodepths_grid` still use all pathways). Regenerated
for both `hallmark_pathway_sweep_ht268b1_clone1` (23×23 = 1+22) and
`hallmark_pathway_sweep_clone2` (35×35 = 1+34).

**All-genes reference line, attempt 1 (2026-08-23, WRONG interpretation, reverted):** Initially
added a vertical line to the two *sweep-level summary* histograms
(`hallmark_qvalue_distribution.png` / `hallmark_stat_true_per_gene_distribution.png`) marking the
all-genes run's own (independently-trained) p-value/stat_true_per_gene. **This was not what was
asked** — the request is to treat the all-genes isodepth as a fixed covariate, OLS-refit it to
each pathway's own genes, and plot that refit loss on each *per-run*
`{run_name}_combined_metric_distribution.png` null-distribution plot (analogous to the existing
`stat_covariate`/`p_value_covariate` dual-line pattern in
`analysis/plots.py::save_metric_distribution_plot` /
`save_combined_celltype_metric_distribution`, normally used for `test.covariate` configs) — not a
comparison of independently-fit stats across the pathway family. Reverted
(`git checkout -- experiments/studies/pathway_panel_sweep/analysis.py`); see the correct
implementation below.

**All-genes-isodepth-as-covariate line on each pathway's own `*_metric_distribution.png` /
`{run_name}_combined_metric_distribution.png` (2026-08-23, correct implementation):** New
`experiments/studies/pathway_panel_sweep/lib.py::apply_all_genes_covariate_baseline(result_json_path,
reference_isodepth, reference_S)`: for each pathway run, z-scores the fixed all-genes isodepth
(loaded once from `reference_full_result_json`, e.g. `FULL_3000HVG`/whole-transcriptome run) and
refits it as a 1-D covariate against that run's *own* saved post-preprocessing expression matrix
(`A` from its `*_isodepths.npz`, same array the isodepth model was trained on) — closed-form OLS
via `methods/trainers/isodepth.py::fit_closed_form_decoder` when `test.decoder` is
`linear`/`quadratic` (`fit_poisson_glm_irls` if `test.metric` is Poisson), else GD via
`train_fixed_covariate_model` for `decoder="nn"`. The resulting loss (`compute_metric` on the
refit predictions) and its permutation p-value (against the run's own `stat_perm`, already saved)
are added as a third "All-genes isodepth (OLS)" line/legend entry alongside the existing
"Isodepth"/observed line, via the pre-existing `stat_covariate`/`p_value_covariate` dual-line
branch in `analysis/plots.py::save_metric_distribution_plot` /
`save_combined_celltype_metric_distribution` (both now accept a `covariate_label=` override,
plumbed through `TestResult.artifacts["covariate_label"]` for the single-plot function — default
stays `"Covariate (midline) decoder"` for the pre-existing midline-covariate regen script that
doesn't pass a label). Persists `stat_covariate`/`p_value_covariate` into each
`artifacts.per_type_summaries[type]` in the run's own result JSON (survives re-analysis) and
rewrites its PNGs in place. Requires `cell_type_mode == "separate"` (what these sweeps always
produce) and matching `S` coordinates between the reference run's npz and the pathway's own npz
(same skip-with-warning pattern as the existing full-vs-pathway Spearman comparison); wired into
`pathway_panel_sweep/analysis.py`'s existing per-row full-isodepth-comparison loop (right after the
coordinate-match check, so `full_iso`/`full_S` are reused) — no new I/O beyond what that loop
already does. New `per_pathway_results.csv` columns `stat_covariate_all_genes` /
`p_value_covariate_all_genes` (mean across cell-type groups; both sweeps only ever have 1 group
per pathway run in practice). Ran for both sweeps end-to-end: **every single pathway in both
sweeps** has `p_value_covariate_all_genes == 1.0` (all 50/50 in each) — i.e. the whole-transcriptome
isodepth axis, even after an optimal OLS refit onto any individual 200-gene panel, never beats
that panel's own null distribution; each pathway's freely-trained isodepth (its own `p_value`) is
a categorically different/better fit than forcing the global axis onto it. Mean
`stat_covariate_all_genes` vs mean `stat_true`: HT268B1 clone1 82205 vs 78794; clone2 (HT306P1)
21619 vs 21251 — covariate consistently worse, as expected.

**Follow-up investigation: why does the all-genes covariate underperform even for
high-correlation pathways? (2026-08-23):** User pushback — many clone2 pathways have very high
`spearman_vs_full` (mean |rho|=0.84, median 0.91, 26/50 > 0.9; **clone1 pathways are only
weakly correlated**: mean |rho|=0.23, median 0.21, max 0.53, none > 0.7 — so the "isodepths match
almost completely" recollection only applies to clone2/HT306P1, not clone1/HT268B1). Investigated
whether `apply_all_genes_covariate_baseline`'s OLS refit has a bug, given even clone2's `hypoxia`
pathway (rho=0.84) still gets `p_value_covariate=1.0` (stat_covariate 31048 vs stat_true 30539,
vs null range ~30520-30670). Verified via two independent fresh re-trainings (direct
`train_parallel_isodepth_model` call, and the exact production `_process_single_celltype_separate`
path, both on `hallmark_pathway_sweep_ht268b1_clone1__apical_junction`'s saved `S`/`A`) that
`fit_closed_form_decoder` on a model's own freshly-extracted isodepth **always** achieves loss <=
that same model's actual (GD-trained) loss for the same z — i.e. **the OLS-refit methodology is
mathematically sound and self-consistent**, confirming no bug in the new covariate code. (Aside:
found that the *original* `apical_junction` run's saved `true_isodepth` does NOT reproduce its own
saved `stat_true` when OLS-refit — OLS gives 107355 vs saved stat_true=104242, which is
impossible if `true_isodepth` were really the exact z that produced that run's `pred_true`; both
fresh re-trainings above landed at 106610–107284, much closer to the OLS-refit figure than to
104242. Root cause not identified (ruled out: rerun/slot index-mismatch in
`_finalize_batched_parallel_training`'s `np.repeat`-based expansion, which is self-consistent;
ruled out: hidden IRLS/closed-form decoder substitution for Gaussian metric, which only applies to
Poisson per `_slot_metric_and_prediction`) — most likely explanation is plain CUDA
non-determinism across training runs (same `seed=42` does not give bit-identical results on GPU)
combined with a **highly multi-modal loss landscape** across the `n_reruns=30` random encoder
inits, i.e. 104242 may just be an unusually lucky one of 30 draws for that specific run. Not
fixed/flagged for follow-up if exact reproducibility ever matters.)

The real answer to "why does even 0.84-correlated covariate underperform the null": (1) Spearman
measures rank/monotonic agreement, not the linear fit a `decoder=linear` OLS refit needs — at
rho=0.84 there is substantial nonlinear/residual disagreement between the two isodepths, and
`nll_gaussian_mse` here is extremely tight (null spread is ~70 units on a ~30600 baseline, <0.25%
relative), so even small deviations from the *ideal* per-gene affine fit blow the loss past the
entire null band. (2) More fundamentally, it's a degrees-of-freedom mismatch, not a correlation
one: the "Isodepth" line (and every null-permutation draw) comes from **freely re-optimizing** a
2-layer-MLP encoder (2→20→20→1, i.e. effectively cell-level flexibility) jointly with the linear
decoder for that specific 200-gene panel; the "covariate" line freezes the latent to the
already-fixed full-transcriptome axis and only fits 2 parameters/gene (slope+intercept) via OLS.
Even a *perfectly rank-correlated but not affinely-identical* latent will lose to a
freshly-fit encoder+decoder, and critically the **null itself gets that same fresh-refit advantage
per permutation** (each null draw retrains its own encoder on shuffled coordinates) — so the
covariate line is being compared against a moving target that always gets to cheat via per-cell
flexibility, not just against the true pathway signal. This is an intrinsic property of comparing
a fixed 1-parameter covariate against a family of freely-refit existence tests, not a flaw in the
new covariate code.

**UNRESOLVED: cached hallmark-sweep artifacts do not reproduce under fresh retraining with
identical config/seed — casts doubt on the "covariate always loses" conclusion above
(2026-08-23):** For `hallmark_pathway_sweep_ht268b1_clone1__apical_junction`, retraining from
scratch via the exact production path (`_process_single_celltype_separate`, same `test` config
incl. `seed=42`, `n_reruns=30`, `epochs=1000`, `n_perms=249`, on the identical saved `S`/`A`)
gives `stat_true≈106610` and null range `[107202, 107546]` — ~2–3% *worse* (higher loss) than the
**cached** result.json's `stat_true=104242` / null `[104279, 104376]`. Also found the cached
run's saved `pred_true` is empirically **rank-3** (SVD of centered `pred_true` has singular values
`[5687, 412, 301, ~1e-13, ...]`, i.e. exactly 3 nonzero components) — mathematically impossible
for `decoder="linear"` (`nn.Linear(1, G)` on a 1-D latent can only produce rank ≤2 output: one
term from the scalar latent, one from the per-gene bias). Ruled out as explanations: (a) rerun/slot
index-mismatch in `_finalize_batched_parallel_training`'s `np.repeat`-based (n_models×n_reruns)
expansion — verified self-consistent via direct testing; (b) hidden closed-form/IRLS decoder
substitution — `_slot_metric_and_prediction` only does this for `nll_poisson_mse`, not
`nll_gaussian_mse`; (c) mislabeled `decoder="quadratic"` — fitting `[z, z², 1]` on the saved
`true_isodepth` still doesn't reproduce `pred_true` (residual as large as the raw fit). Root cause
still unknown — likely either the sweep was run under an earlier version of
`methods/trainers/isodepth.py` / `methods/permutation.py` (git history shows the training code has
changed across several commits; no timestamp in the result JSON to confirm), or there is
significant CUDA non-determinism + a highly multi-modal `n_reruns=30` optimization landscape (two
*independent* fresh retrains of this same pathway in this session — one quick, one full — differed
from each other by ~0.6%, and both differed from the cached value by ~2–3%, i.e. cached looks like
an outlier/lucky draw relative to current-code retrains, not just ordinary rerun noise).
**Practical impact:** the all-genes-covariate-vs-null comparison documented above
(`p_value_covariate_all_genes == 1.0` for 50/50 pathways in both sweeps) is computed against these
cached `stat_true`/`stat_perm` values without re-verifying them, and for this one pathway a fresh
null would have put the covariate stat (107387) *inside* the null range (~p=0.5) rather than
p=1.0. **Do not treat the "covariate never beats the null" conclusion as reliable until the
underlying sweep is re-run from scratch on current code and spot-checked for
reproducibility/self-consistency** (e.g. verify `fit_closed_form_decoder(saved true_isodepth, A) <=
saved stat_true`, which should always hold and did not for this pathway).

**Pre-existing bug found while re-running the above (NOT fixed, flagged for follow-up):**
`per_pathway_results.csv` / `significant_pathway_names` / spearman-matrix & grid labels are all
blank (`pathway_name=""`, `pathway_index=-1`) for **every** row in a fresh re-run of both hallmark
sweeps from this session's shell. Root cause: `manifest.json` (`load_manifest_entries` in
`experiments/studies/pathway_panel_sweep/lib.py`, shared pattern in `fourier_kmax*`,
`existence_sigma`, `existence_consistency`, `random_gene_panel_null`) stores each run's
`result_json_path` as an **absolute path baked in at sweep time**
(`/weka/scratch/uchitra1/users/ajain71/...`, missing a `jhu/` segment), which no longer exists as
a real path — only `/weka/scratch/jhu/uchitra1/users/ajain71/...` does now (this session's
`/home/jhu/ajain71/scratch_uchitra1` symlink resolves to the latter). Manifest keys and
freshly-scanned-file keys never match, so `manifest_entry` is `None` for every row and
`entry.get("pathway_name", "")` falls back to `""`. The two headline histogram plots are
unaffected (they read `p_value`/`stat_true` directly from each pathway's own result JSON, not
from the manifest), but the CSV/matrix/grid pathway labels are currently broken for any fresh
re-run. Not fixed here (out of scope, and the same brittle-absolute-path-in-manifest pattern
recurs in ~6 other study modules) — a real fix should join scanned result files to manifest rows
by `run_name` (derivable from the file path itself) instead of by exact absolute path string.

**Compared the actual pipeline against the user's intended 3-step GSEA-covariate design
(2026-08-23) — found the covariate was never wired through the real pipeline at all, plus a
genuine `test.n_perms`/`n_reruns` mismatch between the reference run and the pathway sweeps.**
User's intended design: (1) run the all-genes/3K-HVG test once, no permutations needed, just save
the isodepth; (2) same settings, gene-set-restricted subsets, with step-1's isodepth fed into the
model **as a covariate**; (3) permutation-test the subsets with parameters identical to step 1
except the gene set. Checked against the codebase:
- The codebase **already has** exactly this "isodepth as a fixed model covariate" mechanism —
  `TestConfig.covariate` (`data/schemas.py` `CovariateConfig`, types `midline` / `total_counts` /
  any `obs_key`) → `methods/permutation.py::_train_covariate_artifacts`. For `type=<obs_key>` (the
  relevant case here — a per-cell array read from `adata.obs[obs_key]`), it does *exactly* what my
  post-hoc `apply_all_genes_covariate_baseline` (`experiments/studies/pathway_panel_sweep/lib.py`)
  does: `zscore_covariate` → `fit_closed_form_decoder` (closed-form OLS for `linear`/`quadratic`
  decoders) → `compute_metric` → `permutation_p_value(stat_covariate, stat_perm)`. So the *math* my
  post-hoc script does is a faithful reproduction of the built-in mechanism.
- **But neither the reference (step-1) run nor any pathway-sweep run ever actually sets
  `test.covariate`** — checked the saved `config.test.covariate` in both hallmark sweeps'
  `reference_full_result_json` and in every pathway base config
  (`configs/calicost/*loss_difference*.json`, `configs/experiments/hallmark_pathway_sweep_*.json`):
  all are `null`. The only "covariate" concept actually active anywhere in these configs is a
  **completely different, pre-existing mechanism** — `data.covariate_whitening` (method
  `"loss-difference"`, whitening out `calicost_tumor_proportion` + `log1p_total_counts` from
  expression *before* fitting, for QC-confound removal) — do not confuse the two; they share the
  word "covariate" but are unrelated (one whitens inputs at the data layer, the other fixes the
  model's latent at the test layer).
  So the "all-genes isodepth as a `test.covariate`" design was never implemented in the actual
  training pipeline for this sweep — my post-hoc script is the *only* place it exists, and it runs
  external to any real training run, reusing whatever `stat_perm` happens to be cached in each
  pathway's already-saved result JSON (see the reproducibility finding above — that cached
  `stat_perm` does not reliably reproduce under a fresh retrain with identical settings).
- **Confirmed settings mismatch between step 1 and steps 2/3** (violates "parameters EXACT same as
  step 1 except gene set"): reference runs use `n_perms=99, n_reruns=10`
  (`loss_diff_clone1_gt0p7_linear_tumor_logcounts_result.json`,
  `loss_diff_clone2_linear_gt0p7_cropy_tumor_logcounts_3000hvg_result.json`), while both
  `hallmark_pathway_sweep_*_study.json` study configs override every pathway run to
  `n_perms=249, n_reruns=30`. `decoder="linear"` and `metric="nll_gaussian_mse"` do match. More
  reruns-per-permutation for the pathway's own free encoder+decoder null means that null is more
  thoroughly optimized (lower/tighter) than the reference's, which — independent of the covariate
  question — makes the pathway's own bar higher to beat than step 1's was.
  (`obs_numeric_filters`/`spatial_denoise_radius_um`/`normalize_total`/`spatial_crop` could not be
  verified from the reference's saved `config.data` — it's missing those keys entirely despite the
  run *name* implying `gt0.7` tumor filter + crop, consistent with the already-documented
  "saved config predates newer serialized fields" issue seen elsewhere for these older calicost
  runs, e.g. `scripts/posthoc/gsea_loss_diff_clone2_linear.py`'s docstring — so this one is
  unverified rather than confirmed-different.)
- **Net conclusion for the user:** the "covariate never wins" result documented above should not
  be trusted as a real biological finding. Two independent, compounding issues: (a) the covariate
  was only ever tested post-hoc against a *stale, possibly-unrepresentative* cached null instead of
  through the real `test.covariate` pipeline with a null computed fresh in the same run, and (b)
  even setting that aside, the pathway sweeps use 2.5x the permutations and 3x the reruns-per-model
  of the reference run, biasing the comparison further against a fixed covariate. **To actually
  answer whether the all-genes isodepth is a significant covariate for these gene subsets**, the
  isodepth would need to be written into `adata.obs` (or otherwise threaded into
  `dataset.meta["covariate_values"]`) and each pathway run re-executed with `test.covariate={"type":
  <obs_key>}` end-to-end through `_process_single_celltype_separate`/`_train_covariate_artifacts`,
  with `n_perms`/`n_reruns` matched to the reference — not recomputed post-hoc against old cached
  artifacts.



**Data source found and downloaded.** The Ravi et al. 2022 Dryad GBM dataset
(`https://datadryad.org/dataset/doi:10.5061/dryad.h70rxwdmj`) is unusable via script: its files API
requires a bearer token for individual files, and the bulk-download URL pattern
(`datadryad.org/downloads/file_stream/<id>`) is blocked by an AWS WAF (403) even with a plausible
`curl` request — no obvious workaround found, would need a real browser session/manual download.
**Used the companion [Greenwald et al. 2024 *Cell* Zenodo record](https://zenodo.org/records/12624108)
instead** (`doi:10.5281/zenodo.12624108`), which has no such protection — plain `curl` on
`https://zenodo.org/api/records/12624108/files/<name>/content` works. Per-sample Visium tar.gz files
are small (~15-50MB each, standard Space Ranger output: `barcodes.tsv.gz`, `features.tsv.gz`,
`matrix.mtx.gz`, `filtered_feature_bc_matrix.h5`, `tissue_positions_list.csv`,
`scalefactors_json.json`, `tissue_lowres_image.png`, `detected_tissue_image.jpg`, `metrics.csv`) —
no need to pull the whole multi-GB archive. File→sample mapping is in
`visium_dataset_description.csv` (also fetchable from the same API pattern).

**Picked patient ZH1007 as the first pair — same patient, two regions, ideal matched positive vs.
negative control:**
- `GBM_ZH1007nec.tar.gz` → **necrotic region** (positive control; expect true hypoxia signal —
  necrotic cores are the textbook hypoxic niche).
- `GBM_ZH1007inf.tar.gz` → **infiltrative region** (relative negative control — non-necrotic tumor
  margin, same patient, same batch/processing).

Loaded both into scanpy (`sc.read_10x_h5` + manual join of `tissue_positions_list.csv` for
`obsm['spatial']`, since these directories don't follow scanpy's `read_visium` expected layout —
had to build the AnnData by hand). **Gotcha**: this environment's numba needs a writable cache dir
when running from outside the repo (`NUMBA_CACHE_DIR=<writable dir>`, e.g. `/tmp/...`) — without it,
`scanpy.pp.normalize_total`/`score_genes` crash with a numba caching `RuntimeError` (setting
`NUMBA_DISABLE_JIT=1` instead "fixes" the crash but introduces a *real* bug, an `UnboundLocalError`
in scanpy's own `_normalize_csr` non-jitted fallback path — don't use that workaround, use the cache
dir fix).

Canonical h5ad files (raw counts, `obsm['spatial']` in pixel coords, `obs` QC columns
`total_counts`/`log1p_total_counts`/`n_genes`/`region_label`/`patient`/`slice_id`, matching this
repo's h5ad conventions) saved to
`data/h5ad/external_controls/{ZH1007-nec,ZH1007-inf}.h5ad` (955 and 1436 in-tissue spots
respectively, 33538 genes, not yet HVG-filtered/gene-panel-restricted — that happens at run-config
time like every other dataset in this repo). Not yet wired into a `configs/experiments/` run — that
is the natural next step (Phase 1: run the same `parallel_permutation`/hypoxia-gene-panel/
random-panel-specificity pipeline already built for HT306P1, on these two new sections).

**Visualization + first quantitative check** (`sc.tl.score_genes` with the same 200-gene
`HALLMARK_HYPOXIA` panel used elsewhere in this repo, scanpy's default control-gene-binned scoring,
not yet the more dropout-robust AUCell approach recommended earlier — that upgrade is still
pending): figure at
`results/experiments/hypoxia_gbm_positive_negative_control/figs/zh1007_nec_vs_inf_overview.png`
(H&E + spatial log-total-counts + spatial hypoxia-score + total-counts-vs-hypoxia-score scatter,
per sample).

- **Sanity check passed**: NEC's mean hypoxia module score (0.196) is significantly higher than
  INF's (0.123) (Mann-Whitney p≈6e-140, though with n=955/1436 spots this p-value is not itself
  meaningful evidence of effect size, just directional). This is a first-order confirmation the
  necrotic/infiltrative region labels track real hypoxia biology, as expected from the literature,
  before spending compute on the full existence-test pipeline.
- **The exact same total-counts confound documented for HT306P1 clone 2 reproduces here, in an
  independent public dataset.** NEC has *higher* median total_counts (15706) than INF (6062) —
  counter to a naive "necrotic tissue = less RNA" prior, likely reflecting different sequencing
  depth per sample library rather than per-spot biology (Number of Reads differs too: 272M vs
  215M) — so **cross-sample (NEC vs INF) total_counts comparisons are confounded by
  sample-level sequencing depth, not just spot-level biology; any downstream test should model
  each section independently (as this repo's per-clone/per-section pipeline already does) rather
  than pooling across sections naively.** Within each section, `corr(log1p(total_counts),
  hypoxia_score)` is positive and substantial even after CPM+log1p normalization: **+0.30 in NEC,
  +0.63 in INF**. I.e. even in this brand-new, independent dataset, spots with more sequencing depth
  systematically score higher on the hypoxia module — the same qualitative confound found in
  HT306P1, not an HT306P1/CalicoST-pipeline-specific artifact. This strengthens the case (from the
  literature review above) that this is a general spatial-transcriptomics phenomenon this project's
  test needs to handle robustly, not a one-off data-quality issue.

**Next steps (not yet done):** (1) build a run config pointing at these two new h5ads (same
`HALLMARK_HYPOXIA` gene_list, `log1p_total_counts` whitening, matching `n_perms`/`n_reruns` to the
HT306P1 hypoxia-panel-specificity study for apples-to-apples comparison); (2) run the existence test
+ random-gene-panel specificity sweep on both NEC (expect significant + specific) and INF (expect
weaker/less specific); (3) if time allows, pull 1-2 more patients (`ZH916`, `ZH881`, `ZH1019` all
have matched inf/T1 or inf/bulk pairs per `visium_dataset_description.csv`) to check replication
before trusting a single-patient result, per the pre-registered protocol above.

## Normal-brain negative control for hypoxia existence test (2026-08-09)

**Greenwald Zenodo (`doi:10.5281/zenodo.12624108`) has no normal-brain Visium sections** — only GBM
and IDH-mut tumor samples per `visium_dataset_description.csv`. The normal cortex references in
Greenwald *Cell* 2024 (UKF256_C, UKF265_C) come from Ravi et al. 2022 on Dryad
(`doi:10.5061/dryad.h70rxwdmj`), which remains script-inaccessible (AWS WAF on file downloads; API
404). The companion Zenodo `18380571` `Visium_GBM_samples.tar.gz` has Ravi *tumor* samples only
(GBM01, GBM15–26), not UKF256/265 normal cortex.

**Substitute negative control (ABORTED 2026-08-09):** Initial attempt used `Br6522_ant` from
spatialDLPFC — **not valid** for this question because it is not patient-matched to the GBM
sections (unlike ZH1007 nec/inf). Sweep killed after user review. The correct null requires
Ravi UKF256_C / UKF265_C normal cortex (Dryad, manual download) or within-section Struct-Norm
spots from the same Greenwald GBM sections.

**ZH1007 baseline (for comparison):** NEC hypoxia p=0.004 (stat≈245k), INF p=0.004 (stat≈370k) —
both significant. Random-panel specificity analysis (2026-08-09): **all 15 random 200-gene panels
also p=0.004** on both NEC and INF; hypoxia ranks **16/16** (lowest) by p-value and **1/16**
(lowest stat_true) — zero panel specificity, confirming permutation-test miscalibration on external
GBM data too.

**Gotcha (2026-08-09): don't manually background long GPU sweep jobs with `nohup ... &` in this
environment — it breaks CUDA device visibility.** Launching the nec/inf specificity sweeps via
`nohup python -m ... &` inside a single shell command failed instantly with `ValueError: Requested
CUDA device but CUDA is not available` (root cause: `cudaGetDeviceCount()` error 304, likely a
device-node/session detachment issue specific to manual job-control backgrounding in this sandboxed
shell). The identical command, issued as its own foreground shell call and left to auto-background
after the tool's own timeout (no `&`, no `nohup`), worked immediately on the same machine seconds
later. **Always launch long GPU training jobs via the harness's native auto-backgrounding; never via
manual `&`/`nohup` job control here.**

**User-raised and confirmed pitfall: raw `stat_true` is not a valid cross-panel comparison metric,
because gene-panel coherence (not spatial signal) drives most of the gap.** Hypoxia's `stat_true` sits
far below every random panel's (NEC: 245139 vs 266931±323; INF: 370139 vs 401130±395 — roughly an
8x-the-random-spread gap in both). This looks like strong specificity but isn't: checked mean absolute
pairwise gene-gene Pearson correlation (on the same standardized expression matrix used for training)
and found the **187-gene hypoxia panel is ~50-60% more internally correlated (0.058) than three
checked random 200-gene panels (0.037-0.039)** — i.e. hypoxia is a genuinely more coherent
(co-regulated) gene set, which makes it easier for *any* flexible shared 1-D-latent decoder to fit
well, whether that latent is driven by true spatial position or by permuted/arbitrary coordinates.
This directly explains the `stat_true` gap without requiring any real spatial specificity. **The only
valid apples-to-apples statistic is each panel's own p-value** (numerator/denominator share identical
genes) — and on that metric, per the table above, there is zero difference between hypoxia and random
panels (both saturate the `n_perms=249` floor). Do not use `stat_true` magnitude, even informally, as
suggestive evidence of anything spatial when comparing panels with different gene sets.

**Per-gene SVG diagnostic follow-up (NEC only, 2026-08-09): 175/187 hypoxia-panel genes individually
"significant" (BH q<0.05, `compute_isodepth_sig_genes` F-test) — investigated why, since this
initially looked implausibly high.** Recovered the trained model's actual per-gene fit (`pred_true`,
saved in `result.json["artifacts"]` even without `save_preds=true`) and the matching preprocessed
expression matrix (reloaded deterministically via `load_dataset` on the exact same config) to compute
real per-gene effect sizes rather than trusting the p-value count alone:
- **Mean per-gene R²=0.088, median=0.073, max=0.40** — i.e. a "significant" gene typically has the
  fitted isodepth axis explaining under 10% of its variance. With `n=955` spots, even small, consistent
  linear associations reach extreme statistical significance (98/175 genes hit the exact float64
  underflow floor `1.11e-16` for the F-test p-value) — this is a large-N-makes-small-effects-detectable
  artifact, not evidence of strong individual-gene signal. BH correction across only 187 tests doesn't
  meaningfully counteract this because most raw p-values are already many orders of magnitude below α.
- **Checked whether this NEC axis is the by-now-familiar total-counts confound — it is not**:
  `spearman(true_isodepth, total_counts) = -0.02` (p=0.54, n.s.). Unlike HT306P1 clone 2 (where
  post-whitening isodepth still tracked total_counts at rho≈0.86-0.94), whatever smooth axis GASTON
  found in ZH1007-nec is genuinely decoupled from library depth. The generic "most genes have some
  association" phenomenon here is being driven by a *different*, uncharacterized smooth spatial axis,
  not simply re-litigating the total-counts issue.
- **Directional coherence check (partial reassurance, but not decisive given the specificity result
  above):** across the full panel, rho signs split almost 50/50 (97 positive / 90 negative vs.
  fitted isodepth) — but the canonical HIF-1 target genes are *unanimous*: VEGFA ρ=-0.66, NDRG1
  ρ=-0.66, SLC2A1/GLUT1 ρ=-0.46, HK2 ρ=-0.42, LDHA ρ=-0.39, BNIP3L ρ=-0.23, P4HA1/2 ρ=-0.20/-0.13 — all
  correctly and consistently signed, more coordinated than the panel-wide split. This is a mild
  biological-coherence signal, but **cannot by itself be read as evidence of hypoxia specificity**
  given that the p-value-based random-panel check above already shows equal "significance" for random
  panels — the missing control (not yet run) is checking whether random panels' own top-|rho| genes
  show similarly internal coherence/consistent direction, which would settle whether this directional
  pattern is itself generic or hypoxia-specific.

**Revised bottom line for the whole external-validation effort**: the ZH1007 nec/inf pair does **not**
provide supporting evidence that the existence test is hypoxia-specific. If anything, it's fresh,
independent (different cohort, different pipeline, no CalicoST/CNV step at all) confirmation that the
test's calibration problem at this gene-panel-size/whitening/spot-count regime is general, not a
CalicoST-pipeline or HT306P1-specific artifact. Trying more positive/negative-control pairs (e.g. the
still-pending true normal-brain negative control from Ravi et al., manual Dryad download required) is
unlikely to resolve this on its own — the underlying fix needs to be at the method level (mandatory
random-panel calibration check per report, decoupling true/permutation training batch-size effects, or
an explicit SpaNorm-style residualization instead of decoder-concatenation whitening), all already
outlined in the HT306P1 section above.

## HT112C1 U1/U2 `pct_mt`/`total_counts` vs isodepth diagnostics (2026-08-09)

Same layout as `scripts/posthoc/loss_diff_clone2_linear_qc_covariate_diagnostics.py`, for the
tumor-proportion-only whitening cropx runs:

- U1: `results/calicost/HT112C1_U1_fig4_loss_diff_tumor_prop_gt0p7/loss_diff_gt0p7_cropx` — reload with
  `obs_numeric_filters gt0.7` + `spatial_crop x>-3` (no denoise). Current config JSON also lists
  denoise/crop for the sibling `..._cropx_3000hvg` run; do not trust `config.data` alone.
- U2: `results/calicost/HT112C1_U2_loss_difference_gt0p7/loss_diff_gt0p7_cropx` — despite the name,
  meta has **no** spatial_crop; matching reload needs `spatial_denoise_radius_um=300` (drops 20/564).

Per-clone NPZ `S` is **re-standardized within clone** (mean0/std1), not the global z-scored coords
in `dataset.S`. Script: `python -m scripts.posthoc.ht112_u1_u2_qc_covariate_diagnostics` →
`{clone}/{clone}_pct_mt_total_counts_diagnostics.png`.

Unlike HT306P1 clone 2 (rho(isodepth, total_counts)=0.92), HT112 U1 clones show little/no
total_counts confounding (clone 2.0: rho=0.09, n.s.); U2 clone 1.0 still shows moderate
total_counts association (rho=0.48).

## Xia2019 MERFISH U2OS linear-decoder existence test + Hallmark GSEA (2026-08-10)

Run: `results/jfan_merfish/260810_jfan_merfish_linear_decoder/` (`configs/jfanmerfish/jfan_raw_pooled_coord_perm.json`,
Xia et al. 2019 U2OS MERFISH, n=1368 cells, G=10050 genes after preprocess, linear decoder,
`n_perms=99`, `n_reruns=30`).

**Existence test:** p=0.01 (observed NLL below *all* 99 perms; gap ≈11.6σ of the null). Visually the
true/lowest-null/highest-null isodepth maps look similarly unstructured — significance is a
global NLL separation, not an obvious spatial gradient.

**Per-gene signal is weak / empty under the pipeline's SVG call:**
`*_isodepth_sig_genes.csv` is header-only (F-test of linear-decoder fit + BH q<0.05 → 0 genes).
Spearman |ρ| vs isodepth tops out ~0.22 (only 3 genes >0.2; 552 >0.1). Large N still makes many
tiny correlations nominally significant in a raw Spearman sense (~1372 genes BH q<0.05 on the
GSEA prerank scores), which is why pre-ranked GSEA can light up even when the F-test SVG list is
empty.

**GSEA / SVG (current, after OLS decoder refit 2026-08-11):** default prerank is
decoder-based: `score = slope(pred, isodepth) * max(Pearson(obs, pred), 0)`.
Legacy obs-correlation methods remain via `--score-method spearman|pearson`.

Jointly trained GD `pred_true` was far from the MSE optimum for the frozen latent
(slopes ~3% of OLS scale, ~55% sign agreement; median Pearson(obs, pred)≈0.007), which
is why the original F-test SVG list was empty. Canonical outputs now come from
`python -m scripts.posthoc.refit_linear_decoder_analyses configs/jfanmerfish/jfan_raw_pooled_coord_perm.json <result.json> --gmt data/gmt/h.all.v2026.1.Hs.symbols.gmt`
(`fit_closed_form_decoder` → `*_pred_true_ols_refit.npz` + regenerated plot/SVG/GSEA).
MSE mean-only=1.000 / gd=0.999907 / **ols=0.996665**; sig genes **0 → 2348** (q<0.05);
decoder-score GSEA still **0 pathways at q<0.05** (best: E2F_TARGETS NES=-1.296, p=0.012,
q=0.299; TNFA_SIGNALING_VIA_NFKB p=0.004, q=0.199). Intermediate `*_gd_decoder.*` /
`*_spearman_obs.*` / `*_decoder_raw.*` archives were deleted.

Root-cause mechanism, independently confirmed via the loss history (`*_loss_history.npz`):
the last 20/1000 epochs vary by only ~1e-5, i.e. training had fully **plateaued**, so the
GD/OLS gap is not "needs more epochs" but a converged-but-suboptimal joint fixed point.
Joint Adam training makes the decoder chase a constantly-shifting encoder latent (never the
truly frozen final one), and `true_isodepth` itself has tiny variance (std≈0.016 for this
run), which makes the loss curvature in the decoder-weight direction (∝ Var(z)) very flat —
so per-gene decoder weights are easily dominated by minibatch (`sgd_batch_size=128`) gradient
noise relative to the also-tiny informative signal. Net effect: the GD decoder ends up ~55%
sign-agreement with OLS (barely better than a coin flip) and ~3% of OLS's slope scale — a
noisy/shrunk estimator, not a real biological "the decoder can't fit these genes" signal.
`postprocess_gsea_isodepth.py` now exposes this generally as `--score-method decoder
--decoder-refit {closed-form,none}` (default `closed-form`, applies whenever
`test.decoder in {linear, quadratic}`; `none` reproduces the raw noisy `pred_true` behavior),
reusing the same `fit_closed_form_decoder` call as `refit_linear_decoder_analyses.py` above —
confirmed numerically identical output (same NES/p/q to the decimal) on this run.

**Training-time OLS (2026-09-16):** The post-hoc refit did not fix encoder training. Hippocampus thin CA2 (`mouse_hippocampus_coordinate_separate_gaussian_linear_p499_thin`, 414 cells) is the exhibit: GD train MSE plateaus at 1.0 (intercept-only), isodepth collapses to ~0.26–0.34, SVG list empty, yet the OLS-refit test is p=0.002 because `stat_true=0.99691` vs perm mean ~0.9974 (null std ~1e-5). The training p-value trajectory on that loss curve goes *up* toward 0.5 as both true and null hit the mean predictor. Fix: for `nll_gaussian_mse` + linear/quadratic (no concatenated covariate), each training forward solves OLS on `z.detach()` and applies those weights to `z`, so the encoder sees an exact linear head. Rerun selection uses the same OLS loss. `decoder="nn"` and loss-difference `h(d,n)` stay on joint Adam. Training loss uses `closed_form_gaussian_mse_torch` (``XᵀY`` against shared ``(N, G)`` targets) so it never materializes ``(M, N, G)`` predictions; that extra tensor was OOMing CA1 into 2 chunks and ~2.5s/epoch.

Math note: for a 1-D **linear** decoder, closed-form-refit decoder score is an exact monotonic
transform of raw `Pearson(obs, isodepth)` (verified rank-corr=1.0) — i.e. once GD noise is
removed, "decoder-based" ranking *is* direct-correlation ranking. It is not identical to
`--score-method spearman` (rank-corr≈0.877 vs Pearson here), and in this weak/borderline run
that Pearson-vs-Spearman gap is enough to flip FDR significance (Spearman's original 8
q<0.05 Hallmark hits, e.g. E2F NES=-1.94/q=0.033, drop to q≥0.199 under the
decoder-consistent/Pearson view) — read the Spearman-based hit list as a rank-metric-sensitivity
artifact in this weak-signal regime, not a robust finding.

**Expression variogram true vs coordinate-permutation null (2026-08-11):**
`python -m scripts.posthoc.plot_expression_variogram_true_vs_perm configs/jfanmerfish/jfan_raw_pooled_coord_perm.json <result.json>`
→ `{stem}_expression_variogram_true_vs_perm.{png,csv}` in the run dir. Gene-pooled empirical
semivariogram γ(d)=1−c(d) on z-scored A, all unordered cell pairs, shared distance bins; null uses
the same `_build_permuted_coordinate_batch` shuffles as the existence test (`test.seed`). For this
U2OS MERFISH run: true has a clear short-range excess covariance (peak c≈0.023 / Δγ≈−0.023 at
~140 µm) that decays to the flat perm null by ~600 µm; 28/40 bins fall outside the perm ±2 SD
band. So there *is* real (if weak, gene-pooled) spatial structure vs label-shuffle — enough for the
NLL test to reject — but the absolute effect is small and short-range, consistent with local
technical/packing structure rather than a tissue-scale expression gradient.

## Gene expression vs isodepth + spatial companion: rank by |OLS slope| among BH-significant genes (2026-09-04)

The binned `*_gene_expression_vs_isodepth.png` (1D) and companion
`*_svg_spatial_expression.png` (2D spatial scatter) show the **same top 5
genes**, ranked by **|OLS slope|** of observed expression on the plotted
coordinate (`A_g ≈ a + b · coord`).  When a parametric decoder is used
(`decoder_df` set, linear/quadratic F-test), panels are restricted to
**BH-significant genes (q < 0.05)** and then sorted by |OLS slope|; each panel
annotates signed `b` plus F-test p/q.  If no gene passes BH, both PNGs are
placeholder figures rather than non-significant genes.  Pass
`significant_only=False` to restore the old all-gene |OLS slope| ranking.
NN decoders (no F-test) still rank all genes by |OLS slope|.

The F-test CSV (`*_isodepth_sig_genes.csv`) is still written for parametric
decoders.  Comparison (4-row covariate) plots are unchanged: they still rank
by |OLS slope| without a significance filter.

Previous (2026-08-27) ranking used |OLS slope| over **all** genes and did not
require significance; F-test hits stayed in the CSV only.  CalicoST
`existence_permutation` gene-vs-isodepth graphs were regenerated after this
change.

## Inclusion-survey selected DBSCAN clusters (2026-08-24, corrected)

The first pass wrongly used whole clones from `survey_inclusion_clones.png`.
Those run dirs, clone-level configs/indices, `selected_panels/`, and
`inclusion_gt0p7_qc60-*.log` were deleted. The intended units are panels
1,2,3,4,5,6,7,11,12,18,21,24 of
`survey_inclusion_dbscan_clusters.png` (5-col, left-to-right then
top-to-bottom). Panel 20 (HT270P1 clone 1 cluster 0) was swapped for
panel 21 (HT288P1 clone 1 cluster 0, n=53). Artifacts:
`results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/selected_clusters/`
and `configs/calicost/inclusion_gt0p7_qc60/`. Spots are pinned with
`obs_indices` to the exact DBSCAN cluster (SpotSweeper pass +
`tumor_prop > 0.7` + cluster membership). Same test settings as before
(3000 HVG, linear decoder, 249 perms, 30 reruns, 1000 epochs). Submit:
`sbatch run_inclusion_gt0p7_qc60.sh`. Re-run only panel 21 with
`sbatch --array=10 run_inclusion_gt0p7_qc60.sh`.

**HVG without scanpy:** `_select_hvg_mask` falls back to numpy/pandas
Seurat-style HVG (`_seurat_hvg_mask_from_counts`) when scanpy is missing
(`isodepth_env` currently has no scanpy).

## Gaussian-blurred visium vs full pseudobulk (2026-08-24)

Spatial autocorrelation is **Gaussian blur of white noise**, not Cholesky
`exp`+`δ`.  Each cell's noise is a Gaussian-weighted average of nearby cells:
`Z'_i ∝ Σ_j Z_j exp(-‖x_i-x_j‖² / (2σ²))` with `kernel.type='smooth'`.
The sweep axis is **`gaussian_blur_sigma_um`: 15, 30, 55, 100 µm** (four
panels, not one blur length).  Neighbor cutoff is `4σ`.

Two aggregation methods, same blur / seeds / 100 µm hex pitch:

| Arm | Spec | Method | Geometry |
|---|---|---|---|
| Visium-style | `configs/experiments/kernel_blurred_visium_pseudobulk_sweep.json` | `visium_pseudobulk` | 100 µm pitch, 55 µm circles, gap cells dropped |
| Full pseudobulk | `configs/experiments/kernel_blurred_pseudobulk_sweep.json` | `binning` | 100 µm hex tiles, every cell assigned |

`bw15` / `bw30` / `bw55` / `bw100` in run names are **Gaussian blur σ in µm**, not
the bin size.  Both arms use the same synthetic data recipe (4096 cells, same
seeds); they look different in `_dataset.png` / `_isodepth.png` because those
figures are drawn **after** aggregation (Visium drops gap cells; hex binning
keeps every cell as a 100 µm tile).  Cell-level pre-aggregation figures are
`{run}_original_dataset.png` (expression + blur field) plus `{run}_overlay.png`
(capture geometry on the original cells).  Backfill existing runs with
`python scripts/posthoc/backfill_pre_aggregation_plots.py --spec <sweep.json>`.

`--spec` is required.  Results:
`results/experiments/kernel_blurred_visium_pseudobulk_sweep` and
`results/experiments/kernel_blurred_pseudobulk_sweep`.

```
python -m experiments.kernel_blurred_sweep --spec configs/experiments/kernel_blurred_visium_pseudobulk_sweep.json
python -m experiments.kernel_blurred_sweep --spec configs/experiments/kernel_blurred_pseudobulk_sweep.json
python -m experiments.kernel_blurred_sweep_analysis --spec configs/experiments/kernel_blurred_visium_pseudobulk_sweep.json
python -m experiments.kernel_blurred_sweep_analysis --spec configs/experiments/kernel_blurred_pseudobulk_sweep.json
```

## Inclusion panel02 non-significant: jagged decoder is covariate, not a failed isodepth fit (2026-08-25)

`panel02_HT112C1_U1_ST_Bn1_clone1_cluster1` (n=67, p=0.184). The black line on
`*_gene_expression_vs_isodepth.png` looks jagged even though `decoder="linear"`
and the plot sorts by isodepth. That is expected under loss-difference: the
decoder is `h(d, n) = Linear([isodepth, zscored tumor_prop])`, so `pred_true` is
exactly affine in `(iso, tumor_prop)` and **not** a function of isodepth alone.
For this run, median `R²(pred ~ tumor_prop)=0.97` vs `R²(pred ~ iso)=0.03`; the
top-5 genes by reconstruction loss (LGALS1 etc.) have `R²(pred ~ tumor)≈1` and
only `R²(A ~ iso)` of 0.006–0.11. Plotting that joint prediction vs isodepth
traces tumor-proportion jitter.

The existence test is whether spatial `d` beats coordinate-shuffled slots that
still see `n`. True NLL is only 0.90σ below the perm mean (45/249 nulls ≤ true).
Sibling cluster 0 on the same clone (`panel01`, n=239) has the same jagged
decoder (tumor still dominates `pred_true`) but z=19.9 / p=0.004 — small extra
spatial signal is detectable at larger n, not here. Do not rank genes by joint
`pred_true` MSE if the question is “which genes follow isodepth.” (As of
2026-08-27 the 1D/2D gene plots rank by |OLS slope| of expression on isodepth,
which avoids this trap.)

## Xia2019 MERFISH Visium-style pseudobulk (2026-08-25)

Config: `configs/jfanmerfish/jfan_raw_pooled_visium_p99.json`. Same Xia et al. 2019 U2OS
h5ad and training recipe as `configs/jfanmerfish/jfan_raw_pooled_coord_perm.json`, but
`test.method=visium_pseudobulk` (100 µm pitch / 55 µm circles). Native
`obsm['spatial']` is already microns (`obs` has `x_microns`/`y_microns`, bbox
~3.2 mm), so `coordinate_um_per_unit=1.0`. Cultured monolayer is sparse (median
NN ~45 µm), so a 55 µm Visium spot typically captures 1 cell (max 3); dry-run
gave 347 occupied pseudospots from 1368 cells (~71% dropped in inter-spot gaps).
`run_permutation.py` loads raw counts and applies CPM/log1p/z-score after
aggregation. Overlay is written automatically as `{run}_overlay.png`.
499-perm sibling: `configs/jfanmerfish/jfan_raw_pooled_visium_p499.json`
(`n_perms=499`, `epochs=2000`, run `260825_jfan_merfish_visium_pseudobulk_p499`).
Generic sbatch wrapper: `sbatch run_config.sh configs/jfanmerfish/jfan_raw_pooled_visium_p499.json`
(`CONFIG` is required; same run_name as the existing result folder, so a
resubmit overwrites `results/jfan_merfish/260825_jfan_merfish_visium_pseudobulk_p499/`).

**End-of-training OLS decoder (2026-08-26):** Gaussian `linear`/`quadratic` slots now
get `fit_closed_form_decoder` after the best rerun is chosen, for **every** slot
(true + permutations), matching Poisson IRLS. Joint Adam is unchanged during
training; only the finalized `pred_true` / `stat_true` / `stat_perm` use OLS.
Loss-difference `h(d, n)` models keep the GD head (concatenated covariate). This
Visium run's GD gene-expression / metric plots were archived as `*_gd_decoder.*`
before rerunning `configs/jfanmerfish/jfan_raw_pooled_visium_p99.json`. The GD decoder
here had ~4% of the OLS slope (TNS3 Pearson −0.24 vs plotted \(R^2=0.005\)); OLS
makes those black lines the actual least-squares fit. Encoder isodepth is the
same object (training loop unchanged).

## `p (covariate)` in hallmark pathway sweep + HVG3000 covariate is miscalibrated to 1.0 across the board (2026-08-26)

In `results/experiments/hallmark_pathway_sweep_clone2_hvg3000_covariate/`, checked all
50 pathway×celltype `*_metric_distribution.png` teal ("HVG3000 isodepth") lines: **0/50**
fall inside the null histogram range, **50/50** have `p_value_covariate == 1.0`. This is
not (necessarily) a biological result — it's an apples-to-oranges comparison baked into
the pipeline when `test.covariate.type` (e.g. `hvg3000_isodepth`) is combined with
`data.covariate_whitening: "loss-difference"` (both true here, per
`configs/experiments/hallmark_pathway_sweep_clone2_hvg3000_covariate_study.json` →
`configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_*_tumor_logcounts*.json`,
whitening on `[calicost_tumor_proportion, log1p_total_counts]`):

- `stat_true` (red line) and **every** `stat_perm` null value come from the joint
  `h(d, n)` model: `methods/architectures.py::_concat_covariate` concatenates the fixed
  `n = [tumor_proportion, log1p_total_counts]` onto the encoder's spatial latent before
  the decoder, for both the true layout and all coordinate-permuted slots (see also the
  `panel02 ... jagged decoder` note above). So the whole null histogram already "sees"
  tumor proportion + log-counts.
- `stat_covariate` (teal line, the fixed external HVG3000 isodepth) is fit by
  `methods/permutation.py::_train_covariate_artifacts` → `_covariate_type_values` branch,
  which does `fit_closed_form_decoder(zscore_covariate(hvg3000_isodepth), A, decoder_type)`
  — **`n` is never concatenated in on this path.** The covariate baseline is missing two
  of the most predictive variables in the dataset that both the null and the true stat get
  for free, which alone is enough to push it above every permutation regardless of whether
  the HVG3000 spatial axis is actually informative for the pathway's genes.

Net: don't trust `p (covariate)` from this pipeline as a fair "does the global isodepth
axis explain this pathway as well as its own fitted isodepth" test when
`covariate_whitening=loss-difference` is also active — it's structurally biased toward
`p=1`. A fix would need `_train_covariate_artifacts`'s values branch to also concatenate
`dataset.meta["covariate_whitening_values"]` (the same `n` used for `stat_true`/`stat_perm`)
onto the fixed latent before fitting/evaluating the covariate decoder.

**2026-08-26: dropped `log1p_total_counts` from this experiment's whitening, temporarily.**
`covariate_whitening_obs_key` was `["calicost_tumor_proportion", "log1p_total_counts"]` in
both `configs/calicost/HT306P1_S1H1Fc2U1Z1Bs1_loss_difference_clone2_hypoxia_genes_tumor_logcounts.json`
(base_config) and `..._gt0p7_cropy_tumor_logcounts_3000hvg_r30.json` (hvg3000_config) —
`log1p_total_counts` was added because total_counts/library size turned out to be strongly
spatially confounded with the fitted isodepth in this clone (rho≈0.86–0.95, see the
"structural limitation" section above), so it was whitened out to try to isolate spatial
signal beyond sequencing-depth/tissue-quality gradients. Per user request, reverted both
configs to single-covariate `covariate_whitening_obs_key: "calicost_tumor_proportion"` (also
dropped `log1p_total_counts` from `obs_drop_na`) to test this in isolation. Renamed
`output.run_name` in both (`..._tumor_logcounts` → `..._tumor_prop_only` /
`..._tumor_prop_only_3000hvg_r30`) so old two-covariate results aren't confused with new
ones — harmless since the pathway sweep overrides `run_name` per-pathway anyway
(`experiments/studies/pathway_panel_sweep/lib.py::build_pathway_run_config`). Note the
`log1p_total_counts` asymmetry documented just above (missing from the covariate-baseline
path) is independent of this change and still applies even with only `tumor_proportion`
whitened. Have NOT yet re-run the HVG3000 reference or the pathway sweep with these edited
configs — configs only, no new results yet as of this note.

**2026-08-26: same `log1p_total_counts` drop for HT268B1 clone 1, and deleted the old
two-covariate sweep outputs.** Same change in
`configs/calicost/HT268B1_slice1_U1_loss_difference_clone1_hypoxia_genes_tumor_logcounts.json`
and `..._gt0p7_tumor_logcounts_3000hvg.json` (`covariate_whitening_obs_key` →
`"calicost_tumor_proportion"`; `run_name` `..._tumor_logcounts` → `..._tumor_prop_only`).
Deleted the incorrect `loss-difference` + HVG3000-covariate outputs so they cannot be reused:

- `results/experiments/hallmark_pathway_sweep_clone2_hvg3000_covariate/`
- `results/experiments/hallmark_pathway_sweep_ht268b1_clone1_hvg3000_covariate/`
- HVG3000 refs: `results/calicost/HT306P1_S1H1Fc2U1Z1Bs1/loss_diff_clone2_linear_gt0p7_cropy_tumor_logcounts_3000hvg_r30/`
  and `results/calicost/HT268B1-Th1K3Fc2U1Z1Bs1/loss_diff_clone1_gt0p7_linear_tumor_logcounts_3000hvg/`

Resubmit both from the repo root:

```
sbatch --export=ALL,SPEC=configs/experiments/hallmark_pathway_sweep_clone2_hvg3000_covariate_study.json \
  run_pathway_panel_sweep.sh
sbatch --export=ALL,SPEC=configs/experiments/hallmark_pathway_sweep_ht268b1_clone1_hvg3000_covariate_study.json \
  run_pathway_panel_sweep.sh
```

## Inclusion existence sweep results (2026-08-26)

12/12 selected DBSCAN clusters finished under
`results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/existence/`
(249 perms; min p = 0.004). **11/12 reject at α=0.05; 9/12 sit at the p floor.**
Only non-significant run: panel 02 HT112C1 U1 clone 1 cluster 1 (n=67, p=0.184,
z=0.90). Weak rejects: panel 06 (p=0.032, n=69) and panel 07 (p=0.012, n=139).
Every non-HT112C1 cluster hits p=0.004, including the smallest (panel 21, n=53).

## Dropping `log1p_total_counts` from whitening makes both hallmark sweeps 50/50 significant on one shared library-size axis (2026-08-26)

Both `hallmark_pathway_sweep_{ht268b1_clone1,clone2}_hvg3000_covariate` were re-run after
reverting `covariate_whitening_obs_key` to `"calicost_tumor_proportion"` only (see the
config-change note above). The result is **not** a mild shift — it is the depth confound
returning and saturating the test:

| | clone1 (HT268B1, n=426) | clone2 (HT306P1, n=125) |
|---|---|---|
| significant (BH q<0.05), 2-covariate → 1-covariate | 22/50 → **50/50** | 34/50 → **50/50** |
| p-values now | all 50 at floor 0.004 | all 50 at floor 0.004 |
| z = (null_mean − stat_true)/null_sd | 9.6 – 19.6 | 5.0 – 7.1 |
| `stat_true` < `null_min` | 50/50 | 50/50 |
| pairwise off-diagonal \|rho\| in `isodepth_spearman_matrix.csv` | mean 0.84, min 0.70 (was mean 0.23, max 0.53) | mean 0.94, min 0.82 (was mean 0.84) |
| \|rho(pathway isodepth, `log1p_total_counts`)\| | mean 0.63, range 0.56–0.67 | mean 0.94, range 0.85–0.96 |
| \|rho(pathway isodepth, `calicost_tumor_proportion`)\| | mean 0.18 (still whitened) | mean 0.18 |

The matched HVG3000 reference axis itself moved the same way for clone1:
`rho(iso, log1p_total_counts)` = **0.61** with tumor-prop-only whitening
(`loss_diff_clone1_gt0p7_linear_tumor_prop_only_3000hvg`) vs **0.17** with the old
`[tumor_proportion, log1p_total_counts]` whitening (`loss_diff_clone1_gt0p7_linear_tumor_logcounts`,
still on disk — useful as the direct before/after comparator since the sweep outputs were deleted).
Clone2 reference: −0.955.

Reproduce the correlation check by un-standardizing `S` from `{run}/1.0/1.0_isodepths.npz` with
`coord_mean`/`coord_std` and `cKDTree`-matching into the source h5ad's `obs`
(max residual ~2.6e-2 raw px, unambiguous) — same technique as the clone-2 QC section above.

Interpretation: with library size unmodeled, every 200-gene panel recovers essentially the *same*
spatial axis, and that axis is the sequencing-depth/tissue-quality gradient. This is the expected
outcome given the already-documented random-gene-panel control (20/30 random 200-gene panels were
"significant" on clone2 *even with* `log1p_total_counts` whitened) — removing the covariate simply
pushes that miscalibration to 100%. **The one-covariate sweeps should be read as a positive control
for the confound, not as biology; do not report 50/50 as a pathway result.**

Two secondary consequences of the same change:
- `p_value_covariate_all_genes` flipped from **1.0 in 50/50** (both sweeps, documented above) to
  0.004 in **40/50** (clone1) and **49/50** (clone2). Partly real (one shared axis now explains
  every panel), partly because the `_train_covariate_artifacts` asymmetry — `n` never concatenated
  on the covariate path — is now a much smaller handicap: the omitted `n` is just tumor_proportion
  (\|rho\|~0.18) instead of tumor_proportion + log-counts.
- The new HVG3000 references were submitted with **`test.n_perms: 1`**, so their own p-values
  (clone1 reports `p=0.5`) are meaningless — they exist only to supply the reference isodepth for
  the `hvg3000_isodepth` covariate. Don't compare that 0.5 against the old
  `tumor_logcounts` reference's p=0.052 (249 perms).

## Inclusion existence: standard permutation sibling of loss-difference (2026-08-27)

Same 12 DBSCAN clusters as `configs/calicost/inclusion_gt0p7_qc60/` (same
`obs_indices` `.npy` files), but **no** `data.covariate_whitening`. That is the
standard coordinate-permutation existence test (`h(d)` only) vs the original
`covariate_whitening: "loss-difference"` runs that concatenate tumor proportion
into `h(d, n)`. Everything else is copied (3000 HVG, linear decoder, 249 perms,
30 reruns, 1000 epochs). Configs:
`configs/calicost/inclusion_gt0p7_qc60_permutation/`. Output:
`results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/existence_permutation/`
(sibling of `existence/`, so the loss-difference results are not overwritten).
Submit: `sbatch run_inclusion_gt0p7_qc60_permutation.sh`.

**Blurred-permutation effect-size overlays (2026-09-14, removed from these plots 2026-10-02):** 10/20 µm × 20 had been drawn on all 12 `existence_permutation` null histograms (50 µm dropped after the first pass — Visium NN ≈ 85 µm, so 50 µm dominated the axis). Those lines are off the per-clone and combined metric plots again, and `blurred_permutations` is gone from `configs/calicost/inclusion_gt0p7_qc60_permutation/` and from `run_inclusion_gt0p7_qc60_permutation.sh`. The `*_blurred_perm_losses.json` sidecars and `*_metric_distribution_pre_blur_overlay.png` (old NLL axis) are still on disk. Cohort figure from that pass: `results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/existence_permutation_blur_effect_size.png`.

**What those permutation isodepths actually track (2026-08-27):** all 12 finished
and reject the coordinate-shuffle null (11/12 at p=0.004; panel 06 p=0.044;
panel 07 p=0.012; panel 21 p=0.008). That spatial structure is real versus
label-shuffle, but it is mostly **not** a tissue-depth/hypoxia axis. With no
covariate whitening, the 1-D encoder recovers the dominant spatially smooth
HVG module, which on Visium is usually %MT, library size, or ribo:

- Panel 12 (HT265 clone 2, n=566): ρ(iso, %MT)=+0.77, top 5 all `MT-*`, 9/20
  MT, gene-axis alignment with %MT = 0.92.
- Panel 01 (HT112 U1 clone 1): top 5 all `MT-ND*`/`MT-CYB`; ρ(iso, %MT)=−0.48.
- Panel 24 (HT306 clone 2 — same clone as the earlier library-size confound):
  ρ(iso, log1p total counts)=**−0.91**, R²=0.83; MT-ND4/MT-CYB still in top 5.
- Panel 11: ρ(iso, log counts)=−0.68, top genes RPS18/RPS20/GAS5.
- Panel 05: top genes ACTB/ACTG1/KRT18, but gene-axis alignment with %MT = −0.77
  and R²(%MT+depth)=0.48 — QC axis even when MT is not the steepest single gene.
- Panel 18: ρ(iso, log counts)=+0.56; top gene RPL22 plus ISGs.

|OLS slope| ranking on z-scored A is the same as |Pearson r|, so a latent that
is collinear with %MT will put the ~5–13 HVG mitochondrial genes at the top of
the gene plots. That is expected, not independent biology.

Cleaner exceptions: panel 04 (MUC2/AGR2/PHGR1 goblet/secretory; ρ vs %MT and
vs depth both ~0; R²=0.01) and, more weakly, panels 06/07 (epithelial/Paneth/Ig;
p=0.044/0.012). Numbers:
`results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/existence_permutation/_isodepth_qc_diagnosis.json`.

Grouped |Spearman ρ| bar chart of those same three covariates vs permutation
isodepth (one trio per panel, y in [0, 1]):
`results/calicost/cohort_survey/gt1clone/inclusion_gt0p7_qc60/existence_permutation/isodepth_covariate_spearman_abs.png`
(regenerate with `python scripts/posthoc/plot_inclusion_permutation_isodepth_covariate_rho.py`).
|ρ| vs total counts equals |ρ| vs log1p total counts (Spearman is rank-based).

The three inclusion metric contact sheets (tumor_prop / pct_mt / total_counts)
show only the 12 existence-test clusters, cropped to each cluster on a square
axis (same viridis 0–1 colorbar as isodepth). No whole-tissue background.

## CosMx Visium-pseudospot overlays (2026-08-27)

**Do not overlay on the full CosMx section or a padded bbox of mixed cell
types.** Restrict to one previously annotated cell-type region via its
`obs_indices` `.npy`, plot those cells first, then run the Visium overlay on
exactly that subset. Script: `scripts/posthoc/plot_cosmx_visium_overlays.py`.
Output: `results/cosmx_visium_pseudobulk_overlays/cosmx_*_region_{cells,overlay}.png`.

The hex lattice **overtiles** the tissue: a candidate spot is kept if any
part of its 55 µm circle overlaps the mask (hexes may hang off the blob).
A cell is captured iff it lands in a circle. Remaining grey points are the
true Visium inter-spot gaps (~71-73% of area at 100 µm pitch), not dropped
edge spots.

**Second bug (found after "overtile" alone still looked too sparse):** the
non-synthetic (density-based) tissue-membership fallback in
`_density_tissue_membership_fn` defaults to `radius_um = 1.5 * median NN
distance` of the dataset's own cells. That's tuned for dense, confluent full
tissue sections, but is far too tight for a single sparse/scattered
cell-type region (e.g. neutrophils, B-cells): median NN was ~5-6 µm for
these regions vs. the DBSCAN `eps` (~22-25 µm =
`percentile(nn_dist,90)*3.0`, see `scripts/data_prep/
segment_cosmx_celltype_regions.py`) actually used to define them as one
spatially-contiguous component in the first place. Using the tight default
punched holes through legitimate tissue between same-type cells, so only
~25 of a ~286-candidate lattice passed even after overtiling.

Fix: added `test.visium_tissue_radius_um` (optional override) to
`TestConfig`/`build_visium_pseudobulk_dataset`
(`methods/visium_pseudobulk.py`) that replaces the `1.5*median_nn` default
when set. `scripts/posthoc/plot_cosmx_visium_overlays.py` now computes this
per-region as `percentile(nn_dist,90)*3.0` (matching the DBSCAN segmentation
exactly) and passes it through. Result: hex lattice now fully tiles each
region's bounding box (c88 24 occupied+6 empty, c35 23+9, c22 10+2 — was
2-4 total before either fix).

**Ran the actual visium_pseudobulk existence test** (not just the plotting
dummy) end-to-end on all 3 regions with real counts (957-960 genes),
production settings (`nll_poisson_mse`, 99 perms, 500 epochs, 10 reruns),
matching `visium_tissue_radius_um` per region. New configs:
`configs/cosmx_celltype_regions/cosmx_{luad_9_r2_neutrophil_c88,
luad_9_r1_neutrophil_c35,luad_5_r3_b_cell_c22}_visium_pseudobulk.json`.
Results in `results/cosmx_celltype_regions/cosmx_*_visium_pseudobulk_poisson/`:
c88 p=0.04 (24 pseudospots), c35 p=0.11 (23), c22 p=0.38 (10 — likely
underpowered, too few spots). No GPU in this interactive shell
(`torch.cuda.is_available()` False here even though configs say
`"device": "cuda"`); ran via `python run_permutation.py --config ... --device cpu`.
Each run took ~2m45s-2m55s wall clock on CPU despite `epochs=500` x
`(1+99 perms)*10 reruns` because the trainer vectorizes all
reruns/perms as one batched forward pass (`parallel_permutation` inner
method) — full production settings are cheap enough to smoke-test on CPU
for datasets this small (10-24 pseudospots, <1000 genes), no need to
reduce epochs/perms/reruns first.

## MERFISH nearest-neighbor spacing (2026-08-31)

KDTree k=2 on `obsm["spatial"]` (already µm, `coordinate_um_per_unit=1`) for every
spot.

**JFan Xia2019 U2OS** (`data/h5ad/merfish_xia2019_u2os.h5ad`, n=1368, 3 batches
B1/B2/B3 = 645/400/323 sharing the same ~3.2 mm coordinate frame):
- Pooled all-spots NN: median **45.4 µm**, mean 47.7, Q1–Q3 29.3–64.8, min 2.17,
  max 135.1, std 24.1. This is what the pipeline sees and matches the earlier
  visium-pseudobulk “median NN ~45 µm” note.
- 92% of those nearest neighbors are a cell from a **different batch** (overlay
  artifact). Same-batch NN: median **97.5 µm**, mean 102.5, min 45.9, max 334.
  Per-batch medians: B1 88.5, B2 106.5, B3 112.2.

**Hypothalamus RCTD slice** (`data/h5ad/merfish_hypothalamus_rctd.h5ad`, n=6111,
one ~1.8×1.8 mm slice, no batch overlay): median **12.5 µm**, mean 13.2,
Q1–Q3 9.3–16.4, min 2.78, max 68.8, std 5.18, mode ~10.5 µm. Same coordinates
as `merfish_hypothalamus.h5ad`.

**Scale verification (2026-08-31):** Hypothalamus units are microns with high
confidence. Coords are byte-identical to squidpy/Moffitt 2018 Animal 1 female
naive **Bregma +0.01** (`Centroid_X`/`Centroid_Y`; n=6111). Bbox 1.790×1.791 mm
matches the paper’s 1.8×1.8 mm FOV; no expansion microscopy. JFan
`x_microns`/`obsm["spatial"]` are microns (not pixels/mm): 1368 cells / 3
batches match Xia et al. PNAS 2019; per-batch bbox ~3.16 mm is consistent with
Fig. 6C/D 500 µm scale bars; same-batch min NN 46–56 µm matches U2OS soma
exclusion. Residual caveat: the experiment used ~2× expansion (paper: “expansion
factor is only ∼2 in each dimension”; 4.2 µm expanded z = 2 µm pre-expansion).
Figures and z-depth are reported in pre-expansion µm; treat `x_microns` as
pre-expansion biological microns unless a deposited-file README says otherwise.
Pooled JFan NN (median 45 µm) remains an overlay artifact, not a unit error.

**JFan per-batch existence configs (2026-08-31):** Pooled run used all 1368 cells
(B1+B2+B3 overlaid). Same linear-decoder recipe as `configs/jfanmerfish/jfan_raw_pooled_coord_perm.json`,
one imaging batch each: `configs/jfanmerfish/jfan_raw_B1_coord_perm.json` (n=645), `jfan_raw_B2_coord_perm.json` (n=400), `jfan_raw_B3_coord_perm.json` (n=323). Filter is `data.obs_filters.batch`. Original
99-perm outputs:
`results/jfan_merfish/260831_jfan_merfish_linear_decoder_B{1,2,3}`.

**JFan per-batch 499-perm rerun (2026-09-04):** Same three configs, `test.n_perms`
bumped 99 → 499. New `output.run_name` so they do not overwrite the 99-perm
dirs: `260904_jfan_merfish_linear_decoder_B{1,2,3}_p499`. Submit:
`sbatch run_config.sh configs/jfanmerfish/jfan_raw_B1_coord_perm.json` (and B2/B3).

**JFan 20/50 µm blurred-permutation overlay (2026-09-04):** Post-hoc
effect-size reference on the existing 499-perm folders (does not recompute
p-values). Blurs the first 20 coordinate permutations (`test.seed=42`) with
Gaussian Nadaraya–Watson kernels at **20 µm (orange)** and **50 µm (purple)**,
retrains those 20 slots per bandwidth with the original linear-decoder
hyperparameters, and overlays both on `{run}_metric_distribution.png`.
Sidecar `{run}_blurred_perm_losses.json` stores an `overlays` list (legacy
single-σ files are migrated on read). Original plot copied once to
`{run}_metric_distribution_pre_blur_overlay.png`.
`python scripts/posthoc/overlay_blurred_perm_losses.py` or
`sbatch scripts/posthoc/run_overlay_blurred_perm_losses.sh`. Historical
10/20/50 µm separate-mode runs (hippocampus CA1/CA2/CA3, hypothalamus
separate) rewrite each `{type}_metric_distribution.png` **and**
`{run}_combined_metric_distribution.png`. The combined grid used to be
skipped whenever there was more than one overlay target. Overlay redraws
keep the covariate dual line.

**Visium hex overlay on hypothalamus (2026-08-31):** Same 100 µm pitch / 55 µm
capture geometry as `visium_pseudobulk`, anchored at the tissue lower-left
(`block_jitter=false`). Plots in
`results/merfish_hypothalamus_visium_overlay/`: full-slice
`hypothalamus_visium_hex_overlay.png` and a 500×500 µm zoom. 6111 cells → 357
occupied spots (389 in-tissue hexes), 73% of cells dropped in inter-spot gaps,
median 5 cells/spot (min 1, max 12). Confirms native MERFISH spacing is cell
scale (~12.5 µm), not Visium 100 µm.

**Gaussian null distributions use MSE, not NLL (2026-09-10):** `test.metric:
"nll_gaussian_mse"` still trains with squared error, but evaluation
(`compute_metric`, masked trainers, subsampling) now *reports* MSE instead of
homoscedastic Gaussian NLL `(n/2) log(2π MSE) + n/2`. Ranking and p-values are
unchanged because NLL is strictly increasing in MSE at fixed n. Null-plot
x-labels go through `methods.metrics.metric_axis_label` ("MSE"). Tests:
`tests/test_gaussian_mse_null_ordering.py`. Existing result JSONs written
before this change still store NLL-scale `stat_true`/`stat_perm`; do not
relabel those plots as MSE without converting. Poisson NLL is unchanged.

**Historical hypothalamus/hippocampus 499-perm reruns (2026-09-10):** New
non-overwriting configs, same linear-Gaussian recipe as current CalicoST/JFan
(`n_perms=499`, `epochs=1000`, `n_reruns=30`, `sgd_batch_size=128`,
`decoder=linear`, `patience=0`, CUDA) plus config-driven 10/20/50 µm × 20
blurred-permutation overlays (`blurred_permutations`). Submit with
`sbatch run_config.sh <config>` — `run_config.sh` runs `run_permutation.py`
then `scripts/posthoc/overlay_blurred_perm_losses.py --from-config` (no-op if
the blur block is missing/disabled). Overlay p-values are unchanged; the 60
extra losses are effect-size references only. Together/separate overlay
training matches the main run (shared encoder vs independent per-type
preprocessing). Hippocampus stored `X` is `log1p(CPM-10k)`: set
`input_expression_representation=log1p_normalized`,
`input_normalization_target=10000`, `make_var_names_unique=true`, then the
usual CPM-1e6 → log1p → z-score path.

- `configs/hypothalamus/hypothalamus_midline_together_gaussian_linear_p499.json`
- `configs/hypothalamus/hypothalamus_midline_separate_gaussian_linear_p499.json`
  (drops types with <30 cells)
- `configs/hippocampus/mouse_hippocampus_coordinate_separate_gaussian_linear_p499.json`
  (CA1/CA2/CA3, obs covariate `coordinate`, `coordinate_um_per_unit=0.2738242950835738`)
  This morphogam file is a **thick** CA selection (4860 cells). The cleaner Visium HD
  ribbon is `data/h5ad/hippocampal_pyramidal.h5ad` (5315 cells; used by
  `visium_hd_hippocampal_block_perm_r24um`). Thin sibling:
  `configs/hippocampus/mouse_hippocampus_coordinate_separate_gaussian_linear_p499_thin.json`
  → `data/h5ad/hippocampal_pyramidal_thin_ca_morphogam.h5ad` (same 5315-cell ribbon,
  morphogam `cell_type`/`coordinate` transferred by barcode then spatial KNN).
  Rebuild with `python scripts/data_prep/build_hippocampus_thin_ribbon_h5ad.py`.
  Preview: `configs/hippocampus/thin_ribbon_preview.png`. Do not overwrite the thick
  p499 folder; thin outputs go to
  `results/hippocampus/mouse_hippocampus_coordinate_separate_gaussian_linear_p499_thin/`.

- **New convention: `notebooks/adhoc_experiments/` for interactive diagnostics** (2026-09-23).
  **Notebook execution gotcha**: `%matplotlib inline` must be placed *after* importing
  `experiments.configuration` (or anything importing it), not before — that module chain
  imports `analysis/cosmx_region_context.py`, which calls `matplotlib.use("Agg")` at import
  time (for headless script use), silently clobbering the inline backend set earlier and
  making `plt.show()` produce no output in the executed `.ipynb`. Env setup: `isodepth_env`
  had no jupyter/ipykernel; installed via
  `pip install ipykernel nbclient nbconvert` + `python -m ipykernel install --user --name isodepth_env`,
  then executed with
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=isodepth_env <nb>`.
  Note the repo's real result/notebook paths resolve under `/weka/scratch/jhu/uchitra1/users/ajain71/...`
  even though the login-node path is `/home/jhu/ajain71/scratch_uchitra1/users/ajain71/...` (same
  filesystem via different mount points) — `nbconvert --execute` writing there needed
  full/unsandboxed filesystem permissions in the sandboxed shell tool even though the target
  files were already owned by the same user with `rw` perms (plain `python script.py` runs
  writing the same path did not hit this, only the nbconvert-launched kernel subprocess did).

**spCorr hippocampus rebuild (2026-09-30 failed; continued 2026-10-01 from saved BANKSY):**
`datasets/spatial-RNA/visiumhd/mouse_hippocampus/spcorr_hippocampus_preprocess.ipynb`
starts from the 8 µm matrix (393,543 spots). The local `spcorr_nb_run.sh` run died
in BANKSY neighborhood chunk 1/5 (`Rscript` exit -9). Do not rerun that cell.
`banksy_outputs/` is empty (`savefig=False`). The April 7 executed notebook
`banksy_processing_out.ipynb` already has the full-gene clustering: lambda 0.8,
20 PCs, Leiden resolutions 0.7/0.75/0.8 → 28/29/32 domains, kept 0.8.
Labels are `obs['banksy_cluster']` in `mouse_brain_banksy_ALL_GENES.h5ad`
(393,543 × 19,059, raw counts). That clustering is 32 domains, not 31, and no
domain pair sums to 5,910. Domains 23 (CA1 markers, 5,619) and 28 (CA3 markers,
3,742) are one curve; 86 disconnected spots were dropped.
`continue_spcorr_from_saved_banksy.py` then applies the notebook's CA argmax
(log1p CPM-1e4; Pvrl3 is absent, Nectin3 is used) and MorphoGAM `CurveFinder`
defaults, flipping `t` so mean CA1 < mean CA3.
Outputs, same folder: `spcorr_hippocampus_ca_allgenes.h5ad` (9,275 × 19,059),
`spcorr_hippocampus_ca_counts.h5ad` (9,275 × 641), `spcorr_pyramidal_spots.csv`,
`spcorr_banksy_meta.json`, `spcorr_hippocampus_ca_overview.png`.
This is not the paper object (5,910 spots, 964 genes). Argmax ties go to CA1:
5,281/9,275 spots have all three scores at 0 and are labeled CA1, so the CA1
count (7,437) is mostly empty bins. Nonzero-score calls are CA1 2,156, CA2 626,
CA3 1,212. `t` still tracks the curve (domain 23 median 0.26, domain 28 median 0.74).
The older HVG `mouse_brain_banksy.h5ad` cluster 23 (5,315 spots) sits inside this
pair (`pyrimidal_hippocampus.ipynb`, then DBSCAN). R 4.5.1 needs `module load GCC/13.2.0`
before `R/4.5.1-gfbf-2023b`, plus ImageMagick and GMP. Packages live in
`~/R/x86_64-pc-linux-gnu-library/4.5`. DropletUtils `read10xCounts` needs
`col.names=TRUE`.

**Quadratic OLS Gram singularity (2026-09-29):**
`configs/hypothalamus/hypothalamus_midline_separate_gaussian_linear_p499.json` is named
linear, but the JSON on disk sets `"decoder": "quadratic"`. Cell-type-separate training
of Inhibitory (2357 cells) died around epoch 451/500 with
`torch.linalg.solve: (Batch element 95): The solver failed because the input matrix is singular`
inside `_solve_ols_beta`. Quadratic features are `[z, z**2]` then per-feature standardized
plus an intercept. A collapsed slot (constant `z`) or a slot where `z**2` is an affine
function of `z` makes `XᵀX + 1e-6 I` numerically singular in float32: the absolute ridge
is ~1e-6 while Gram diagonals are ~N, so the condition number sits below cuSOLVER's
rcond cutoff and the whole batched solve aborts. Flooring small diagonals does not fix
the collinear case (diagonals are large; one eigenvalue is ~0). A larger relative ridge
would change healthy slots and the float64 lstsq tests. `_solve_ols_beta` still tries
`torch.linalg.solve` first and, on `torch.linalg.LinAlgError`, returns
`pinv(gram) @ xty` (min-norm, finite). The function is not `torch.compile`d, so the
except path is live. Do not rerun the multi-hour GPU job unless asked.

**JFan B2 linear p499 with cosine decay has a stable p-value (2026-09-29, second pass):**
Same folder, regenerated ~13:48 with `sgd_cosine_lr_decay=true`, `sgd_cosine_eta_min=0`,
`T_max` = epochs × steps/epoch, `lr=0.001`, `sgd_batch_size=128` still on. True best
rerun last improved at epoch 93 (MSE 0.9913) and is flat thereafter; 26/30 true reruns
are flat over the last 50 epochs. Red p-value plateaus at 0.321 from epoch 450–500
(official `(1+#nulls ≤ observed)/(1+n_perms)` = 0.324). Zero rank crossings in the last
50 epochs. Two nulls sit within 1e-4 of the observed loss, so a few more steps could
flip a count (~0.004 in p) but not the call. Late flatness is partly the cosine
schedule driving Adam's step size to 0; the true model stalled while the learning rate
was still near its start. The earlier no-cosine pass in this same folder ended at
p≈0.69 with nulls still descending at epoch 500 — that history was overwritten.
Sept 4 `*_metric_distribution_pre_blur_overlay.png` remains the original NLL histogram
(p=0.038) and does not match this MSE result.

**JFan B1/B2/B3 cosine SGD (2026-09-29):**
`configs/jfanmerfish/jfan_raw_B{1,2,3}_coord_perm.json` set `test.sgd_cosine_lr_decay`
true. `eta_min` stays 0 and `sgd_cosine_t_max_steps` stays unset, so
`CosineAnnealingLR` uses `T_max = epochs * ceil(n_cells / sgd_batch_size)` and
anneals `lr=0.001` to 0 over the full minibatch run. `output.run_name` is still
`260904_jfan_merfish_linear_decoder_B{1,2,3}_p499`, which already holds the
cosine-off results — a resubmit writes into those folders.

**Xia/Fan 2019 clustering preprocessing (2026-09-29, executed):**
`notebooks/adhoc_experiments/jfan_u2os_fan2019_preprocess.ipynb` rebuilt the
pooled (B1+B2+B3) matrix: CPM 1e6, `log1p`, ComBat on `obs['batch']`, GAM
variance~mean (`GAM_DF=5`), top 1598 genes by adjusted variance, PCA to 30.
Output `data/h5ad/merfish_xia2019_u2os_fan2019.h5ad` is 1368 × 1598.
`X` is already `(combat - mean) / sqrt(fitted variance)`. Do not CPM, log1p, or
z-score it again (`standardize_expression` would replace her fitted-SD scale
with unit observed variance). Every source gene was already in ≥10 cells, so
`MIN_CELLS=10` dropped nothing. Chi-square p<0.05 calls 2750 genes and BH
q<0.05 calls 2451; 1598 is the rank cut. New PC1 vs published `X_pca` PC1 has
Pearson **−0.964** (sign arbitrary). 30 new PCs explain 46.9% of OD-gene
variance. Published PCs are in `obsm['X_pca_published']`.

Pooled existence config, matching the cells ComBat and the 1598-gene set were
fit on: `configs/jfanmerfish/jfan_fan2019_pooled_coord_perm.json`
(`260930_jfan_merfish_fan2019_all_batches_p249`, 1368 cells, no `obs_filters`).
Same linear-Gaussian recipe (249 perms, 500 epochs, 30 reruns) and normalization
off. The three batches still share one coordinate frame, so this spatial test
sees the overlaid monolayers. Per-batch siblings remain at
`configs/jfanmerfish/jfan_fan2019_B{1,2,3}_coord_perm.json`.

**Fan2019 per-batch runs are more significant because the loss is no longer an equal vote across 10,050 z-scored genes (2026-09-30):**
`260930_jfan_merfish_fan2019_B{1,2,3}_p249` vs cosine z-score runs
`260904_jfan_merfish_linear_decoder_B{1,2,3}_p499` (B1 old folder was
re-run at 1000 epochs the same morning, p still 0.004).

| batch | z-score p (z) | Fan2019 p (z) |
|---|---|---|
| B1 | 0.004 (8.2) | 0.004 (28.9) |
| B2 | 0.324 (0.05) | 0.008 (3.8) |
| B3 | 0.70 (−0.5) | 0.172 (0.94) |

Fan `X` keeps only the 1598 overdispersed genes and does not z-score, so MSE
weights genes by leftover variance (median var 1.68, max 36; top 50 genes are
18% of total variance). Top isodepth genes become cell-cycle
(CDC6, E2F1, GTSE1, BUB1B, CENPF, MCM10), the sibling-cell spatial structure.
The old equal-weight 10,050-gene average dilutes that. F-test SVG counts went
*down* (B1 1861→610), so the existence-test gap grew while the gene list got
shorter. Absolute MSE is not comparable (Fan stat_true ≈ 2.73 vs z-score ≈ 0.99).

**B1 cell-cycle together config (2026-09-30, not submitted):**
`configs/jfanmerfish/jfan_raw_B1_cellcycle_coord_perm.json`. Same raw
B1 recipe as `jfan_raw_B1_coord_perm.json` (CPM/log1p/z-score, linear, cosine,
249 perms, 1000 epochs) with `cell_type=together` and
`cell_type_key=cell_cycle_phase`. Shared encoder, per-stage linear decoder.
B1 counts: G1 269, G1/S 167, G2/M 130, M 69, S 10. `min_cells_per_celltype=5`
keeps S. Run name `260930_jfan_merfish_B1_cellcycle_together_p249`.

**result.json omits decoder predictions unless `output.save_preds` is true (2026-09-30):**
`save_standardized_outputs` used to embed `pred_true` (and covariate / Freedman–Lane
prediction matrices) even when `save_preds` was false. That was ~6.5e6 floats and
187 MB of the B1 z-score `*_result.json`. `_method_artifact_keys_for_json` now
drops `pred_true`, `pred_true_covariate`, `pred_true_full_iso`, and
`freedman_lane_pred` unless `save_preds` is true. Plots still read those arrays
from the in-memory `TestResult` during the same run. Separate-cell-type
`*_isodepths.npz` still stores `pred_true` for offline plot regen. Already-written
result JSON files are unchanged.

**CalicoST standard existence-permutation montage (2026-10-02):** Same 12 panels, under `existence_permutation/`. `existence_permutation_isodepth_pvalue_summary.png` uses the same tile layout. Panels at or under 0.05: 01, 03, 04, 05, 06 (0.024), 07, 11, 12, 18, 24, all at the 0.004 floor except 06. Not significant: 02 (0.672), 21 (0.208).

## `correlated_null_permutation` variogram-matched null (2026-10-06)

Config key `test.method: correlated_null_permutation`. Candidate blur bandwidths are
`test.variogram_sigma_multiples` times the median nearest-neighbor distance on **raw**
(unstandardized) coordinates, with a plain Gaussian kernel
(`exp(-d²/(2σ²))`, row-normalized — same convention as `methods/spatial_blur.py`, not
the locfit `/2.5` convention from `notebooks/variogram_matching.ipynb`).

The matching loop prints a tqdm bar (`variogram matching`, or
`variogram matching: <cell type>` in separate mode), one tick per permutation.
It follows `test.verbose` the same way training bars do (`--quiet` hides it).

Per perm slot: joint row shuffle of standardized `A`, blur, pick σ̂ by minimizing OLS
variogram fit SSE `γ_X ≈ a γ_Y + b` over geoR-style bins (Q25 max.dist, 13 bins default),
then `X' = sqrt(|a|) Y_{σ̂} + sqrt(|b|) R` with `R = W X / sqrt(N)` and `W` i.i.d.
`N(0,1)` so each row of `R` has covariance `XᵀX/N` (unit diagonal when `A` is
gene-z-scored). Re-z-score columns before GASTON. Negative OLS `a`/`b` use `|·|`
(STCompare-style). Training uses fixed `S` and `a_batched` nulls,
including loss-difference `fixed_covariate_values`. Memory: `(n_perms+1)×N×G` float32
for `a_batched`. Diagnostics plots land in `variogram-matching/` under the run output
dir (not serialized in `result.json`).

**First hypothalamus separate run (2026-10-06)** —
`results/hypothalamus/hypothalamus_correlated_null_separate_gaussian_linear_p249`. Name says
p249 but `test.n_perms=99` (p floor 0.01). Only the three densest types reject (Astrocytes
1081, Excitatory 1322, Inhibitory 2357; z ≈ −5 to −6). Sparse types skew to p≈1
(Fibroblast and Mature oligodendrocyte 1.00, Unstable 0.97, Endothelial 0.93): the null
fits *better* than the data. Cause: σ grid is a multiple of the per-type median NN distance
(17 µm Inhibitory vs 50–124 µm sparse types), so 12×NN reaches 600–1500 µm, close to the
1.8 mm tissue size. Small-n variograms are noisy (selection SSE 1–5 vs ~1e-3 for dense
types). The OLS fit then returns tiny or negative intercepts and slopes up to 1e4, and the
`|a|`/`|b|` fallback hides this. The surrogate becomes a near-global smooth field (in
`Mature_oligodendrocyte/variogram-matching/variogram_best_match.png` the surrogate
semivariance is ~0.1 while the real one is ~1), which a linear isodepth fits trivially. For
dense types the opposite holds: the largest blur (~200–280 µm) plus the Q25 (~540 µm)
variogram window cannot produce tissue-scale trends, so the null may be too easy to reject.
p-values are not comparable across cell densities until the null is calibrated.

**JFan Fan2019 B1/B2/B3 adaptive correlated null (2026-10-06, not submitted):**
Latest per-batch matrix is `data/h5ad/merfish_xia2019_u2os_fan2019.h5ad` filtered by
`obs['batch']` (B1 645, B2 400, B3 323). Expression is already CPM 1e6 / log1p /
ComBat / fitted-SD scaled on the 1598 overdispersed genes, so
`normalize_total`, `log1p`, and `standardize_expression` stay false. Each config
is one replicate, so coordinates are that monolayer only. Method matches the
hypothalamus adaptive config (`correlated_null_adaptive_permutation`, nn
candidates 0–0.9, 13 bins). Training matches the Fan2019 existence runs
(linear Gaussian, cosine, 249 perms, 500 epochs, 30 reruns).
Configs: `configs/jfanmerfish/jfan_fan2019_B{1,2,3}_correlated_null.json`.
Submit each with `sbatch run_config.sh <config>`.

**Batched variogram matching (2026-10-08):** `test.variogram_backend` defaults to
`cuda`. `sequential` is the reference loop and `cpu` is the CPU batch. A config
that omits the key uses the GPU. A process that has already started keeps the backend it loaded.
`test.variogram_perm_batch_size` overrides the batch. The library entry points take
the same arguments (`backend`, `perm_batch_size`). Sequential code stays in
`methods/correlated_null_permutation.py`. The batched path is
`methods/variogram_matching_accel.py`.

The fit is unchanged: each permutation × gene still picks its own bandwidth and its
own nonnegative `a`, `b`, including the nn=0 identity score. Draws stay on the NumPy
generator, one row permutation then one N×N Gaussian matrix, then the deterministic
work is batched. Adaptive weights are built once per candidate (about 47 MB for B1's 14
positive nn values). If that cache would exceed 512 MB they are built in cell blocks,
applied to the whole permutation batch, and dropped. Candidates are scored one at a
time so the full candidates × permutations × cells × genes array is never stored.
CUDA products are float64. Surrogates stay float32.

On CPU, the default batch size is 1. SciPy's sparse variogram products got slower
as the column count grew (B1, 32 perms: 1.02 s/perm at batch 1, 2.57 s at batch 8,
1.94 s at batch 32). Weight reuse is the CPU win, about 1.2× versus the reference.
CUDA is flat from batch 8 through 32, so the GPU default is the 256 MB expression
budget (32 permutations on B1).

Fan2019 B1 check (645 cells, 1598 genes, 15 nn candidates, 13 bins, seed 42, L40S),
8 permutations at batch 8, against the sequential path: permutations and noise
identical, 0 bandwidth mismatches out of 12,784, candidate variograms and fit SSE
within 6e-15, float32 surrogates identical. Same result on an earlier 2-permutation
check. Component times, per permutation: sequential (8 perms) smooth 0.14 s,
variogram 1.03 s, fit 0.004 s, total 1.23 s. CUDA batch 32 (32 perms) smooth 0.018 s,
variogram 0.025 s, fit 0.004 s, total 0.076 s, about 16×. A 249-permutation surrogate
build is about 5 minutes sequential and about 20 seconds on this GPU. Float64 GPU
speed is hardware-dependent; this L40S was faster than the CPU. Bit-for-bit identity
is not guaranteed when the product order or the device changes. The three Fan2019
adaptive configs set `variogram_backend` to `cuda`. Set it back to `sequential` to
compare.

**`correlated_null_smooth_noise_study` FPR diagnosis (2026-10-07):** Results
(`results/experiments/correlated_null_smooth_noise_study`, run 2026-10-06 13:25, before
the NNLS variogram fit landed at 17:37, so they use the old `|a|`/`|b|` mixing). FPR is
5/10, 2/10, and 0/10 at blur σ 15/30/45 µm. The selected σ matches on average (median NN
≈7 µm, so multiples 2/4/6–8). Null mean minus true stat is the same +0.0043 at 15 and 30 µm,
but the null SD grows with σ (0.0027 / 0.0084 / 0.0140). The number of independent
patches scales as ~(L/σ)², so the null gets tighter at short σ and a fixed small
roughness bias in the surrogate shows up as rejections. Sources of that bias: the
grid point under the true σ (2×NN≈14 µm) is picked almost always at 15 µm; negative OLS
intercepts turned into extra nugget noise via `|b|` (86% of genes at 15 µm, 67% at 30 µm,
32% at 45 µm); and only ~1–2 of 13 variogram bins (Q25 ≈ 300+ µm window) resolve a 15 µm
correlation. At 45 µm, about a third of picks are 8×NN (smoother), so the null fits better
and p-values skew high. Re-run with the NNLS fit and a finer or log-spaced σ grid.

**Maynard DLPFC 151673 adaptive correlated null (2026-10-07, config only, not submitted):**
All 12 Visium sections are downloaded as separate raw-count h5ads
(`data/h5ad/151507.h5ad` … `151676.h5ad`). The only config kept is the canonical slide
**151673** (`obs['Region']`, unlabeled spots are real NaNs). `151507` is the odd file:
its layer column is `sce.layer_guess`, not `Region`. `151669`–`151672` have no Layer1 or Layer2.
`cell_type: "separate"` runs each present layer on its own.
Config: `configs/maynard_dlpfc/maynard_dlpfc_151673_correlated_null_adaptive_separate_gaussian_linear_p249.json`.
Count recipe matches the other Visium DLPFC runs (1000 HVGs, library-size, log1p, z-score).
Method matches the hypothalamus separate adaptive config (linear Gaussian, 249 perms,
500 epochs, 10 reruns, nn candidates 0–0.9). Submit with `sbatch run_config.sh <config>`.

