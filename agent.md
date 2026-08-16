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

Run: `results/jfan_merfish/260810_jfan_merfish_linear_decoder/` (`configs/jfan_merfish.json`,
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
`python -m scripts.posthoc.refit_linear_decoder_analyses configs/jfan_merfish.json <result.json> --gmt data/gmt/h.all.v2026.1.Hs.symbols.gmt`
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

Math note: for a 1-D **linear** decoder, closed-form-refit decoder score is an exact monotonic
transform of raw `Pearson(obs, isodepth)` (verified rank-corr=1.0) — i.e. once GD noise is
removed, "decoder-based" ranking *is* direct-correlation ranking. It is not identical to
`--score-method spearman` (rank-corr≈0.877 vs Pearson here), and in this weak/borderline run
that Pearson-vs-Spearman gap is enough to flip FDR significance (Spearman's original 8
q<0.05 Hallmark hits, e.g. E2F NES=-1.94/q=0.033, drop to q≥0.199 under the
decoder-consistent/Pearson view) — read the Spearman-based hit list as a rank-metric-sensitivity
artifact in this weak-signal regime, not a robust finding.

**Expression variogram true vs coordinate-permutation null (2026-08-11):**
`python -m scripts.posthoc.plot_expression_variogram_true_vs_perm configs/jfan_merfish.json <result.json>`
→ `{stem}_expression_variogram_true_vs_perm.{png,csv}` in the run dir. Gene-pooled empirical
semivariogram γ(d)=1−c(d) on z-scored A, all unordered cell pairs, shared distance bins; null uses
the same `_build_permuted_coordinate_batch` shuffles as the existence test (`test.seed`). For this
U2OS MERFISH run: true has a clear short-range excess covariance (peak c≈0.023 / Δγ≈−0.023 at
~140 µm) that decays to the flat perm null by ~600 µm; 28/40 bins fall outside the perm ±2 SD
band. So there *is* real (if weak, gene-pooled) spatial structure vs label-shuffle — enough for the
NLL test to reject — but the absolute effect is small and short-range, consistent with local
technical/packing structure rather than a tissue-scale expression gradient.
