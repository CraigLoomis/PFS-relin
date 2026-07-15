#!/usr/bin/env python
"""Benchmark fit() wall-clock across worker counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity.benchmarks.fitThreading import runBenchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 ramp is generated",
    )
    args = parser.parse_args()
    return runBenchmark(Path(args.data_path) if args.data_path else None)


if __name__ == "__main__":
    raise SystemExit(main())
