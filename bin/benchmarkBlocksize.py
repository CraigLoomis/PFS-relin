#!/usr/bin/env python
"""Benchmark fit() wall-clock across blockSize values.

Defaults probe a geometric sweep [32, 64, 128, 256, 512] with 2 trials, which
takes 5-10 minutes on a typical workstation and is enough to pick a reasonable
blockSize. Tune with the same worker count you will use in production — the
optimum shifts with thread count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity.benchmarks.fitBlocksize import runBenchmark  # noqa: E402


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
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 ramp is generated",
    )
    parser.add_argument("--sizes", type=_parseSizes, default=[32, 64, 128, 256, 512],
                        help="Comma-separated blockSize values (default: 32,64,128,256,512)")
    parser.add_argument("--trials", type=int, default=2,
                        help="Trials per blockSize (default: 2)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker count passed to fit(); None uses the auto default")
    parser.add_argument(
        "--basin-tolerance", type=float, default=0.02,
        help="Sizes within this fraction of the best time are considered "
             "tied for the recommendation (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()
    return runBenchmark(
        Path(args.data_path) if args.data_path else None,
        sizes=args.sizes, trials=args.trials, workers=args.workers,
        basinTolerance=args.basin_tolerance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
