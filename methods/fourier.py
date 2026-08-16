"""Fourier spectral randomization nulls for regular square grids."""
from __future__ import annotations

import time

import numpy as np


def _grid_shape_from_meta(meta: dict, n_cells: int) -> tuple[int, int]:
    height = meta.get("grid_height")
    width = meta.get("grid_width")
    if height is None or width is None:
        side = int(round(np.sqrt(int(n_cells))))
        if side * side != int(n_cells):
            raise ValueError(
                "fourier_spectral_randomization requires a regular square grid. "
                "Dataset metadata must include grid_height/grid_width, or n_cells must be a perfect square."
            )
        return side, side
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0 or height * width != int(n_cells):
        raise ValueError(
            "Invalid grid metadata for fourier_spectral_randomization: "
            f"grid_height={height}, grid_width={width}, n_cells={n_cells}."
        )
    if height != width:
        raise ValueError(
            "fourier_spectral_randomization currently supports square grids only; "
            f"got grid_height={height}, grid_width={width}."
        )
    return height, width


def build_fourier_spectral_randomization_surrogates(
    A: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    """Build coherent Fourier phase-randomized expression surrogates.

    The same random phase field is applied to every gene for a surrogate. This
    preserves each gene's Fourier power spectrum and preserves cross-gene phase
    relationships at each Fourier mode, instead of independently scrambling genes.
    """
    A = np.asarray(A, dtype=np.float32)
    n_cells, n_genes = A.shape
    h = int(grid_height)
    w = int(grid_width)
    if h <= 0 or w <= 0 or h * w != n_cells:
        raise ValueError(
            "A cannot be reshaped to the requested Fourier grid: "
            f"A has {n_cells} rows, grid is {h}x{w}."
        )
    if int(n_surrogates) <= 0:
        raise ValueError("n_surrogates must be > 0")

    print(
        f"    Fourier spectral randomization: grid={h}x{w}, G={n_genes}, "
        f"surrogates={int(n_surrogates)}, shared_phase=True",
        flush=True,
    )
    t0 = time.time()

    fields = A.reshape(h, w, n_genes)
    spectrum = np.fft.fft2(fields, axes=(0, 1))
    amplitude = np.abs(spectrum)
    rng = np.random.default_rng(int(seed))
    surrogates = np.empty((int(n_surrogates), n_cells, n_genes), dtype=np.float32)

    for i in range(int(n_surrogates)):
        # FFT phases from a real white-noise image have the needed Hermitian symmetry,
        # so inverse transforms remain real while randomizing spatial phase.
        noise = rng.normal(size=(h, w))
        noise_spectrum = np.fft.fft2(noise)
        noise_abs = np.abs(noise_spectrum)
        phase = np.divide(
            noise_spectrum,
            noise_abs,
            out=np.ones_like(noise_spectrum, dtype=np.complex128),
            where=noise_abs > 1e-12,
        )
        phase[0, 0] = 1.0 + 0.0j  # preserve per-gene means exactly.
        randomized = np.fft.ifft2(spectrum * phase[:, :, np.newaxis], axes=(0, 1)).real
        surrogates[i] = randomized.reshape(n_cells, n_genes).astype(np.float32)

    print(f"    Fourier surrogates: {time.time() - t0:.2f}s", flush=True)
    if n_genes <= 300:
        orig_power = float(np.mean(np.abs(spectrum) ** 2))
        surr_power = float(np.mean(np.abs(np.fft.fft2(surrogates[0].reshape(h, w, n_genes), axes=(0, 1))) ** 2))
        print(
            f"    Fourier power check: orig={orig_power:.4f} surrogate[0]={surr_power:.4f}",
            flush=True,
        )
    return surrogates


def build_fourier_spectral_randomization_surrogates_from_meta(
    A: np.ndarray,
    meta: dict,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    h, w = _grid_shape_from_meta(meta, np.asarray(A).shape[0])
    return build_fourier_spectral_randomization_surrogates(
        A,
        n_surrogates,
        seed,
        grid_height=h,
        grid_width=w,
    )
