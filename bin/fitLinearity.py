#!/usr/bin/env python
"""Fit a per-pixel linearity correction for one detector's lab ramp."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cap BLAS/LAPACK to one thread per process. The fit parallelizes over tiles, so
# an uncapped BLAS thread pool oversubscribes (tile workers x BLAS threads);
# single-threaded BLAS is fastest at every core budget on a many-core host. Must
# run before numpy is imported (via the fitLinearity imports below); setdefault
# lets an explicit environment value win.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity import paths  # noqa: E402
from fitLinearity.fitLinearity import SanityCheckConfig, cliTag, runFit, runPlot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", type=str, required=True,
                        help="Detector id, naming the input subdirectory (e.g. 18734)")
    parser.add_argument("--data-root", type=str, default=None,
                        help=f"Root holding the per-detector ramp dirs "
                             f"(default: {paths.dataRoot()})")
    parser.add_argument("--out-root", type=str, default=None,
                        help=f"Root for output artifacts (default: {paths.DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--fit", action="store_true",
                        help="Fit and save a linearity correction FITS file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots (reads existing FITS + input data)")
    parser.add_argument("--plot-format", type=str, default="png",
                        help="Plot file format: png, pdf, svg (default: png)")
    parser.add_argument("--nplot", type=int, default=1000,
                        help="Number of pixels to plot (default: 1000)")
    parser.add_argument("--deviation-limit", type=float, default=None,
                        help="Fractional deviation limit for the fit range (default: None)")
    parser.add_argument("--deviation-start", type=float, default=0.5,
                        help="Fraction of the ramp at which deviation is measured (default: 0.5)")
    parser.add_argument("--order", type=int, default=4,
                        help="Polynomial order (default: 4)")
    parser.add_argument("--saturation-level", type=float, default=None,
                        help="Absolute saturation level in DN (default: None)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for pixel sampling (default: 0)")
    parser.add_argument("--no-photodiode", action="store_true",
                        help="Skip the illumination-drift photodiode correction")
    parser.add_argument("--low-flux-fraction", type=float, default=0.5,
                        help="Low-flux rejection fraction (default: 0.5)")
    parser.add_argument("--saturation-knee", type=float, default=0.5,
                        help="Per-pixel saturation knee (default: 0.5; -1 to disable)")
    parser.add_argument("--bad-linearity-multiplier", type=float, default=5.0,
                        help="Flag HIGH_FIT_RESIDUAL when residualRms > multiplier × median(good "
                             "residualRms) (default: 5.0; -1 to disable)")
    parser.add_argument("--rate-stability", action="store_true",
                        help="Run the split-half rate-stability gate after the fit and "
                             "fold RATE_UNSTABLE into the saved correction")
    parser.add_argument("--rate-stability-threshold", type=float, default=0.20,
                        help="Fractional half-vs-half rate-disagreement threshold (default: 0.20)")
    parser.add_argument("--rate-stability-floor", type=float, default=5.0,
                        help="Rate-floor DN in the disagreement denominator (default: 5.0)")
    parser.add_argument("--fitrange-min", type=float, default=None,
                        help="Lower bound (DN) for the fit-range histogram (default: auto p0.05)")
    parser.add_argument("--fitrange-max", type=float, default=None,
                        help="Upper bound (DN) for the fit-range histogram (default: auto p99.95)")
    args = parser.parse_args()

    if not args.fit and not args.plot:
        args.fit = True

    config = SanityCheckConfig(
        order=args.order,
        deviationLimit=args.deviation_limit,
        deviationStart=args.deviation_start,
        saturationLevel=args.saturation_level,
        lowFluxFraction=args.low_flux_fraction,
        # CLI value -1 disables the gate; the API spells that None.
        saturationKnee=None if args.saturation_knee < 0 else args.saturation_knee,
        badLinearityMultiplier=(
            None if args.bad_linearity_multiplier < 0 else args.bad_linearity_multiplier
        ),
        noPhotodiode=args.no_photodiode,
        seed=args.seed,
        nplot=args.nplot,
        plotFormat=args.plot_format,
        fitrangeMin=args.fitrange_min,
        fitrangeMax=args.fitrange_max,
        rateStability=args.rate_stability,
        rateStabilityThreshold=args.rate_stability_threshold,
        rateStabilityFloor=args.rate_stability_floor,
    )

    dataRoot = Path(args.data_root) if args.data_root else None
    outRoot = Path(args.out_root) if args.out_root else None
    inputDir = paths.inputDir(args.det, root=dataRoot)
    if not inputDir.is_dir():
        parser.error(f"input directory does not exist: {inputDir}")
    outDir = paths.outputDir(args.det, cliTag(config), root=outRoot)

    print(f"  in : {inputDir}", flush=True)
    print(f"  out: {outDir}", flush=True)

    if args.fit:
        runFit(config, inputDir, outDir)
    if args.plot:
        runPlot(config, inputDir, outDir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
