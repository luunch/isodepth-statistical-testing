# Isodepth Statistical Testing

This repository runs simulation-based spatial testing experiments for isodepth models.

## Structure

- `isodepth/`: shared runtime, metrics, and training utilities.
- `models.py`: neural model definitions (`IsoDepthNet`, `GastonMixNet`, `ParallelIsoDepthNet`).
- `data_manager.py`: synthetic spatial data generation and visualization helpers.
- Experiment entry points:
  - `permutation_frozen_encoder.py`
  - `full_retraining_gaston.py`
  - `parallel_full_retraining.py`
  - `gaston_mix_frozen_encoder.py`
- Comparison and benchmark scripts:
  - `comparison_qq_plot.py`
  - `benchmark_retraining.py`
  - `scaling_benchmark.py`
- Smoothness diagnostics:
  - `test_laplacian_smoothness.py`
  - `test_permuted_smoothness.py`

## Refactor Notes

Shared code that used to be repeated in many scripts now lives in `isodepth/`:

- Device selection: `isodepth.runtime.choose_device`
- Seed setup: `isodepth.runtime.set_global_seed`
- Results directory creation: `isodepth.runtime.ensure_results_dir`
- Gaussian NLL and empirical p-value calculations: `isodepth.metrics`
- Laplacian smoothness metric: `isodepth.metrics.laplacian_smoothness`
- Early-stopping training loop and module reset helper: `isodepth.training`

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```
