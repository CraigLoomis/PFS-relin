"""Benchmark: fit() wall-clock across blockSize values.

Runs `nirLinearity.fit` on a 4096x4096x30 ramp at each candidate
blockSize and prints a timing table plus a recommended default for this
machine. Not a test — no pass/fail criteria.

By default the input is a synthetic ramp (linear with mild per-pixel
saturation and read noise) so the script is self-contained and
portable. Pass `--data-path` to benchmark on a real lab NPZ instead.
"""

from __future__ import annotations

import time
from pathlib import Path

import lsst.obs.pfs.h4Linearity as nirLinearity

from fitLinearity.loader import loadCorrectedRamp
from fitLinearity.syntheticRamp import syntheticRamp


def runBenchmark(
    dataPath: Path | None,
    sizes: list[int],
    trials: int,
    workers: int | None,
    basinTolerance: float = 0.02,
) -> int:
    if dataPath is None:
        print("No --data-path provided; generating synthetic ramp ...", flush=True)
        correctedRamp = syntheticRamp()
    else:
        dataPath = Path(dataPath)
        if not dataPath.exists():
            print(f"Data file missing: {dataPath}")
            return 1
        print(f"Loading {dataPath} ...", flush=True)
        correctedRamp, _ = loadCorrectedRamp(dataPath)
    print(
        f"  shape={correctedRamp.reads.shape} dtype={correctedRamp.reads.dtype}",
        flush=True,
    )
    print(
        f"  sweep: sizes={sizes} trials={trials} workers={workers}",
        flush=True,
    )

    # Warm-up call so the first measurement isn't paying first-touch costs
    # on memory allocations / imported BLAS libraries. Use the median of
    # the candidate sizes so the warm-up is representative.
    warmSize = sorted(sizes)[len(sizes) // 2]
    print(f"Warm-up run (blockSize=({warmSize}, {warmSize})) ...", flush=True)
    _ = nirLinearity.fit(
        [correctedRamp], blockSize=(warmSize, warmSize), workers=workers
    )

    results: list[tuple[int, float, float, float]] = []
    for bs in sizes:
        trialTimes = []
        goodFrac = float("nan")
        for _ in range(trials):
            t0 = time.perf_counter()
            correction = nirLinearity.fit(
                [correctedRamp], blockSize=(bs, bs), workers=workers
            )
            elapsed = time.perf_counter() - t0
            trialTimes.append(elapsed)
            goodFrac = correction.diagnostics.summary["goodPixelFraction"]
        best = min(trialTimes)
        mean = sum(trialTimes) / len(trialTimes)
        results.append((bs, best, mean, goodFrac))
        print(
            f"  blockSize=({bs:>4d},{bs:>4d})  "
            f"best={best:6.2f}s  mean={mean:6.2f}s  "
            f"goodPixelFraction={goodFrac:.4f}",
            flush=True,
        )

    print("\nSummary (sorted by best time):")
    print(f"  {'blockSize':>10s}  {'best (s)':>10s}  {'mean (s)':>10s}")
    for bs, best, mean, _ in sorted(results, key=lambda r: r[1]):
        print(f"  {bs:>4d}x{bs:<5d}  {best:>10.2f}  {mean:>10.2f}")

    # Recommendation. The minimum time defines the basin; any size within
    # `basinTolerance` of it is statistically tied. Among ties we prefer
    # the largest, which trades a hair of throughput for fewer tiles
    # (lower scheduling overhead, less peak memory in the as_completed
    # queue) and is more robust if workers/BLAS settings shift.
    bestTime = min(r[1] for r in results)
    threshold = bestTime * (1.0 + basinTolerance)
    basin = [r for r in results if r[1] <= threshold]
    chosen = max(basin, key=lambda r: r[0])

    print(
        f"\nFastest: blockSize=({min(results, key=lambda r: r[1])[0]}, "
        f"{min(results, key=lambda r: r[1])[0]}) at {bestTime:.2f}s",
        flush=True,
    )
    if len(basin) > 1:
        basinSizes = sorted(r[0] for r in basin)
        print(
            f"Basin (within {basinTolerance:.0%} of fastest): "
            f"{basinSizes}",
            flush=True,
        )
    print(
        f"\nRecommended: blockSize=({chosen[0]}, {chosen[0]})  "
        f"(best={chosen[1]:.2f}s, "
        f"{100 * (chosen[1] / bestTime - 1):.1f}% slower than fastest)",
        flush=True,
    )
    return 0
