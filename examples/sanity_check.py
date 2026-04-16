"""Sanity check: run relin on a real 4096x4096x29 lab ramp.

Loads the example NPZ, applies a standard photodiode correction
(illumination-drift normalization to the first read), fits, saves FITS,
reloads, applies, and reports residuals.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import relin
from relin.loaders import loadNpz
from relin.types import Ramp


def _t(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"  [{now - t0:7.2f}s] {label}", flush=True)
    return now


def main() -> None:
    dataPath = Path("examples/linearity/18734/18734_164220.npz")
    fitsPath = Path("examples/linearity/18734/correction.fits")

    print(f"Loading {dataPath} ...", flush=True)
    t0 = time.perf_counter()
    ramp, photodiode = loadNpz(dataPath)
    t1 = _t("loadNpz", t0)

    print(
        f"  deltas: shape={ramp.deltas.shape} dtype={ramp.deltas.dtype} "
        f"min={ramp.deltas.min():.4g} max={ramp.deltas.max():.4g}",
        flush=True,
    )
    print(
        f"  photodiode[0..2]={photodiode[:3]} photodiode[-1]={photodiode[-1]}",
        flush=True,
    )

    # Standard illumination-drift correction: scale every read so the reported
    # delta reflects the same incident flux as read 0. Read 0 is unchanged.
    scale = (photodiode[0] / photodiode).astype(np.float32)  # (N,)
    print(
        f"  photodiode scale range: min={scale.min():.6f} max={scale.max():.6f}",
        flush=True,
    )
    correctedDeltas = ramp.deltas * scale[:, None, None]
    correctedRamp = Ramp(deltas=correctedDeltas)
    t1 = _t("photodiode correction applied", t1)

    print("Fitting (blockSize=(512, 512)) ...", flush=True)
    correction = relin.fit([correctedRamp], blockSize=(512, 512))
    t1 = _t("relin.fit", t1)

    print("Summary diagnostics:", flush=True)
    for k, v in correction.diagnostics.summary.items():
        print(f"  {k:40s} {v}", flush=True)

    print(
        f"  fitMin: min={correction.fitMin.min():.3g} "
        f"max={correction.fitMin.max():.3g}",
        flush=True,
    )
    print(
        f"  fitMax: min={correction.fitMax.min():.3g} "
        f"max={correction.fitMax.max():.3g}",
        flush=True,
    )

    print(f"Saving to {fitsPath} ...", flush=True)
    relin.saveFits(fitsPath, correction)
    t1 = _t("relin.saveFits", t1)
    print(f"  FITS size: {fitsPath.stat().st_size / 1e6:.1f} MB", flush=True)

    print("Reloading FITS ...", flush=True)
    loaded = relin.loadFits(fitsPath)
    t1 = _t("relin.loadFits", t1)

    # Sanity: round-trip preserves coefficients bit-for-bit (ImageHDUs).
    coefMatch = np.array_equal(loaded.coefficients, correction.coefficients)
    bpMatch = np.array_equal(loaded.badPixelMask, correction.badPixelMask)
    print(f"  coefficients bitwise-equal:   {coefMatch}", flush=True)
    print(f"  badPixelMask bitwise-equal:   {bpMatch}", flush=True)

    print("Applying correction to input ramp ...", flush=True)
    result = relin.apply(loaded, correctedRamp)
    t1 = _t("relin.apply", t1)

    # Linearity check: for each good pixel, cumulativeLinear should be a
    # straight line in n. Fit a least-squares line per pixel and report the
    # deviation from linearity (worst and median residual RMS).
    good = loaded.badPixelMask == 0
    nGood = int(good.sum())
    print(
        f"  good pixels: {nGood}/{good.size} "
        f"({100 * nGood / good.size:.2f}%)",
        flush=True,
    )

    # Take a random sample of 10000 good pixels to keep memory bounded while
    # checking linearity.
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(good.ravel())
    sampleIdx = rng.choice(idx, size=min(10000, nGood), replace=False)
    H, W = good.shape
    rows = sampleIdx // W
    cols = sampleIdx % W

    trajectories = result.cumulativeLinear[:, rows, cols]  # (N, K)
    n = np.arange(1, trajectories.shape[0] + 1, dtype=np.float64)

    # LS line y = a*n: slope a_k = sum(n*y) / sum(n*n)
    slopes = np.einsum("n,nk->k", n, trajectories) / np.sum(n * n)
    residuals = trajectories - slopes[None, :] * n[:, None]
    rms = np.sqrt(np.mean(residuals**2, axis=0))
    print(
        f"  per-pixel linearity RMS over {len(sampleIdx)} sampled pixels:",
        flush=True,
    )
    print(f"    median={np.median(rms):.4f}  p95={np.percentile(rms, 95):.4f}  max={rms.max():.4f}", flush=True)
    print(
        f"  per-pixel slope over sampled pixels: "
        f"median={np.median(slopes):.3f} "
        f"p95={np.percentile(slopes, 95):.3f}",
        flush=True,
    )

    oorFrac = float(result.outOfRangeMask.sum()) / result.outOfRangeMask.size
    print(f"  out-of-range sample fraction: {oorFrac:.4f}", flush=True)

    t1 = _t("done", t0)


if __name__ == "__main__":
    main()
