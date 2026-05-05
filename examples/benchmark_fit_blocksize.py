"""Benchmark: fit() wall-clock across blockSize values.

Runs `nirLinearity.fit` on a 4096x4096x30 ramp at each candidate
blockSize and prints a timing table plus a recommended default for this
machine. Not a test — no pass/fail criteria.

By default the input is a synthetic ramp (linear with mild per-pixel
saturation and read noise) so the script is self-contained and
portable. Pass `--data-path` to benchmark on a real lab NPZ instead.

Usage:
    uv run python examples/benchmark_fit_blocksize.py
    uv run python examples/benchmark_fit_blocksize.py --sizes 32,64,128,256,512
    uv run python examples/benchmark_fit_blocksize.py --trials 3 --workers 8
    uv run python examples/benchmark_fit_blocksize.py --data-path path/to.npz

Defaults probe a geometric sweep [32, 64, 128, 256, 512] with 2 trials,
which takes 5–10 minutes on a typical workstation and is enough to pick
a reasonable blockSize. Tune with the same `workers` value you'll use
in production — the optimum shifts with thread count.
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


def _syntheticRamp(
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


def _parseSizes(s: str) -> list[int]:
    sizes = [int(x) for x in s.split(",") if x.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("--sizes must list at least one integer")
    if any(b < 1 for b in sizes):
        raise argparse.ArgumentTypeError("--sizes values must be >= 1")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes", type=_parseSizes, default=[32, 64, 128, 256, 512],
        help="Comma-separated square blockSize values to probe "
             "(default: 32,64,128,256,512)",
    )
    parser.add_argument(
        "--trials", type=int, default=2,
        help="Trials per blockSize; the minimum is reported (default: 2)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker count passed to fit(); None uses the auto default",
    )
    parser.add_argument(
        "--basin-tolerance", type=float, default=0.02,
        help="Sizes within this fraction of the best time are considered "
             "tied for the recommendation (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 "
             "ramp is generated",
    )
    args = parser.parse_args()

    if args.data_path is None:
        print("No --data-path provided; generating synthetic ramp ...", flush=True)
        correctedRamp = _syntheticRamp()
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
    print(
        f"  sweep: sizes={args.sizes} trials={args.trials} "
        f"workers={args.workers}",
        flush=True,
    )

    # Warm-up call so the first measurement isn't paying first-touch costs
    # on memory allocations / imported BLAS libraries. Use the median of
    # the candidate sizes so the warm-up is representative.
    warmSize = sorted(args.sizes)[len(args.sizes) // 2]
    print(f"Warm-up run (blockSize=({warmSize}, {warmSize})) ...", flush=True)
    _ = nirLinearity.fit(
        [correctedRamp], blockSize=(warmSize, warmSize), workers=args.workers
    )

    results: list[tuple[int, float, float, float]] = []
    for bs in args.sizes:
        trials = []
        goodFrac = float("nan")
        for _ in range(args.trials):
            t0 = time.perf_counter()
            correction = nirLinearity.fit(
                [correctedRamp], blockSize=(bs, bs), workers=args.workers
            )
            elapsed = time.perf_counter() - t0
            trials.append(elapsed)
            goodFrac = correction.diagnostics.summary["goodPixelFraction"]
        best = min(trials)
        mean = sum(trials) / len(trials)
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
    # `basin-tolerance` of it is statistically tied. Among ties we prefer
    # the largest, which trades a hair of throughput for fewer tiles
    # (lower scheduling overhead, less peak memory in the as_completed
    # queue) and is more robust if workers/BLAS settings shift.
    bestTime = min(r[1] for r in results)
    threshold = bestTime * (1.0 + args.basin_tolerance)
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
            f"Basin (within {args.basin_tolerance:.0%} of fastest): "
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


if __name__ == "__main__":
    sys.exit(main())
