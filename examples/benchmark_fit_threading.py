"""Benchmark: fit() wall-clock across worker counts.

Reads the example lab ramp, applies the standard illumination-drift
photodiode correction, then runs `relin.fit` for several worker counts
and prints a timing table. Not a test — no pass/fail criteria.

Usage:
    uv run python examples/benchmark_fit_threading.py

The dataset is not in the repo (see .gitignore); if it's missing, the
script prints a pointer and exits cleanly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import relin
from relin.loaders import loadNpz
from relin.types import Ramp


def main() -> int:
    dataPath = Path("examples/linearity/18734/18734_164220.npz")
    if not dataPath.exists():
        print(
            f"Data file missing: {dataPath}\n"
            "Place the lab NPZ at that path (see .gitignore)."
        )
        return 1

    print(f"Loading {dataPath} ...", flush=True)
    ramp, photodiode = loadNpz(dataPath)
    scale = (photodiode[0] / photodiode).astype(np.float32)
    correctedDeltas = ramp.deltas * scale[:, None, None]
    correctedRamp = Ramp(deltas=correctedDeltas)
    print(
        f"  shape={correctedDeltas.shape} dtype={correctedDeltas.dtype}",
        flush=True,
    )

    workerCounts = [1, 2, 4, 8]
    results = []

    # Warm-up call so the first measurement isn't paying first-touch costs
    # on memory allocations / imported BLAS libraries.
    print("Warm-up run (workers=1) ...", flush=True)
    _ = relin.fit([correctedRamp], blockSize=(512, 512), workers=1)

    for w in workerCounts:
        t0 = time.perf_counter()
        correction = relin.fit(
            [correctedRamp], blockSize=(512, 512), workers=w
        )
        elapsed = time.perf_counter() - t0
        goodFrac = correction.diagnostics.summary["goodPixelFraction"]
        results.append((w, elapsed, goodFrac))
        print(
            f"  workers={w:>2d}  elapsed={elapsed:7.2f}s  "
            f"goodPixelFraction={goodFrac:.4f}",
            flush=True,
        )

    baseline = results[0][1]
    print("\nSummary (speedup vs workers=1):")
    print(f"  {'workers':>8s}  {'elapsed (s)':>12s}  {'speedup':>8s}")
    for w, elapsed, _ in results:
        speedup = baseline / elapsed if elapsed > 0 else float("nan")
        print(f"  {w:>8d}  {elapsed:>12.2f}  {speedup:>7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
