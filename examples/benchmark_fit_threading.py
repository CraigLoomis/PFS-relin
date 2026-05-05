"""Benchmark: fit() wall-clock across worker counts.

Runs `nirLinearity.fit` for several worker counts at fixed
``blockSize=(512, 512)`` and prints a timing table. Not a test — no
pass/fail criteria.

By default the input is a synthetic ramp (linear with mild per-pixel
saturation and read noise) so the script is self-contained and
portable. Pass ``--data-path`` to benchmark on a real lab NPZ instead;
the standard illumination-drift photodiode correction is applied.

Usage:
    uv run python examples/benchmark_fit_threading.py
    uv run python examples/benchmark_fit_threading.py --data-path path/to.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import nirLinearity
from nirLinearity.loaders import loadNpz
from nirLinearity.types import Ramp

from syntheticRamp import syntheticRamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 "
             "ramp is generated",
    )
    args = parser.parse_args()

    if args.data_path is None:
        print("No --data-path provided; generating synthetic ramp ...", flush=True)
        correctedRamp = syntheticRamp()
    else:
        dataPath = Path(args.data_path)
        if not dataPath.exists():
            print(f"Data file missing: {dataPath}")
            return 1
        print(f"Loading {dataPath} ...", flush=True)
        ramp, photodiode = loadNpz(dataPath)
        scale = (photodiode[0] / photodiode).astype(np.float32)
        deltas = np.empty_like(ramp.reads)
        deltas[0] = ramp.reads[0]
        deltas[1:] = np.diff(ramp.reads, axis=0)
        correctedReads = np.cumsum(deltas * scale[:, None, None], axis=0)
        correctedRamp = Ramp(reads=correctedReads)
    print(
        f"  shape={correctedRamp.reads.shape} dtype={correctedRamp.reads.dtype}",
        flush=True,
    )

    workerCounts = [1, 2, 4, 8]
    results = []

    # Warm-up call so the first measurement isn't paying first-touch costs
    # on memory allocations / imported BLAS libraries.
    print("Warm-up run (workers=1) ...", flush=True)
    _ = nirLinearity.fit([correctedRamp], blockSize=(512, 512), workers=1)

    for w in workerCounts:
        t0 = time.perf_counter()
        correction = nirLinearity.fit(
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
