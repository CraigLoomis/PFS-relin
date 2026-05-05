"""Synthetic ramp generator shared by the example benchmarks.

Produces a 4096x4096x30 cumulative-DN cube with plausible IR-detector
behavior, so benchmarks can run without the lab NPZ being on disk.
"""

from __future__ import annotations

import numpy as np

from nirLinearity.types import Ramp


def syntheticRamp(
    N: int = 30, H: int = 4096, W: int = 4096,
    seed: int = 0, deadFraction: float = 0.01,
) -> Ramp:
    """Generate a synthetic ramp for benchmarking when no real data is given.

    Per-pixel: linear with ~10% rate variation, mild quadratic droop
    (~5% at the last read, simulating detector saturation onset), small
    read noise. About `deadFraction` of pixels (default 1%) are "dead":
    rate well below the lowFluxFraction cutoff but cumulatives still
    span a wide range, so the per-pixel matrix is well-conditioned —
    the fit rejects them cleanly via the lowFlux mask rather than
    FIT_FAILED. The resulting goodPixelFraction is ~98%, close to
    typical lab data.

    Generates ~2 GB of float32 (the (30, 4096, 4096) reads array) plus a
    couple of (4096, 4096) per-pixel parameter arrays. Read noise is
    added per-read to keep peak memory bounded.
    """
    rng = np.random.default_rng(seed)
    # Per-pixel rate: mean 2000 DN/read, 10% sigma. The natural tail
    # stays well above the lowFluxFraction=0.5 cutoff (1000 DN/read).
    rate = (2000.0 + 200.0 * rng.standard_normal((H, W))).astype(np.float32)
    # Dead pixels: rate ~100 DN/read (well below the 1000 cutoff). Cumulatives
    # still span 100..3000 so the polynomial fit is well-conditioned even
    # though the lowFlux mask rejects them.
    deadMask = rng.random((H, W)) < deadFraction
    rate[deadMask] = 100.0
    # Per-pixel droop amplitude: 5% nominal, 1% sigma. cumRaw =
    # cumLinear * (1 - droopAmp * (cumLinear / fullScale)^2).
    droopAmp = (0.05 + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
    fullScale = np.float32(N * 2000.0)

    reads = np.empty((N, H, W), dtype=np.float32)
    for k in range(N):
        cumLinear = rate * np.float32(k + 1)
        x = cumLinear / fullScale
        reads[k] = cumLinear * (1.0 - droopAmp * x * x)
        reads[k] += 10.0 * rng.standard_normal((H, W), dtype=np.float32)

    return Ramp(reads=reads)
