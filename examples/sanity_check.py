"""Sanity check: run relin on a real 4096x4096x29 lab ramp.

Loads the example NPZ, applies a standard photodiode correction
(illumination-drift normalization to the first read), fits, saves FITS,
reloads, applies, and reports residuals.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import relin
from relin.loaders import loadNpz
from relin.types import (
    BORDER_PIX,
    FIT_FAILED,
    INSUFFICIENT_POINTS,
    MASKED_BY_INPUT,
    NON_MONOTONIC,
    Ramp,
)


def _t(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"  [{now - t0:7.2f}s] {label}", flush=True)
    return now


def _plotDiagnostic(
    rawCum: np.ndarray,
    linCum: np.ndarray,
    fitMin: np.ndarray,
    fitMax: np.ndarray,
    badPixelMask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    rate: float,
    outPath: Path,
    nPlot: int = 1000,
    order: int = 4,
    deviationLimit: float | None = None,
    nRefReads: int = 5,
    saturationLevel: float | None = None,
    seed: int = 0,
    brightFraction: float = 0.5,
    darkFraction: float = -0.05,
    detector: str = "",
    visit: str = "",
) -> None:
    """Combined diagnostic: good-pixel before/after (rows 1-2) + rejected traces (row 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = badPixelMask.shape
    N = rawCum.shape[0]
    reads = np.arange(1, N + 1)

    fig = plt.figure(figsize=(16, 15), layout="constrained")
    gs = fig.add_gridspec(3, 2, hspace=0.1)
    # Rows 1-2 share x; row 3 is independent.
    axRawCum = fig.add_subplot(gs[0, 0])
    axLinCum = fig.add_subplot(gs[0, 1], sharex=axRawCum)
    axRawDelta = fig.add_subplot(gs[1, 0], sharex=axRawCum)
    axLinDelta = fig.add_subplot(gs[1, 1], sharex=axRawCum)
    axRejCum = fig.add_subplot(gs[2, 0])
    axRejBar = fig.add_subplot(gs[2, 1])

    # --- Rows 1-2: good-pixel before/after ---
    isGood = badPixelMask[rows, cols] == 0
    goodRows, goodCols = rows[isGood], cols[isGood]

    rng = np.random.default_rng(seed)
    K = min(nPlot, len(goodRows))
    pick = rng.choice(len(goodRows), size=K, replace=False)
    pRows, pCols = goodRows[pick], goodCols[pick]

    for k in range(K):
        r, c = pRows[k], pCols[k]
        mTrace = rawCum[:, r, c]
        tTrace = linCum[:, r, c]
        fMin, fMax = fitMin[r, c], fitMax[r, c]
        inRange = (mTrace >= fMin) & (mTrace <= fMax)

        rawDelta = np.diff(mTrace, prepend=0.0)
        rawDelta[0] = mTrace[0]
        linDelta = np.diff(tTrace, prepend=0.0)
        linDelta[0] = tTrace[0]

        for seg, color in _segments(reads, mTrace, inRange, "C0", "red"):
            axRawCum.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(reads, tTrace, inRange, "C1", "red"):
            axLinCum.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(reads, rawDelta, inRange, "C0", "red"):
            axRawDelta.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(reads, linDelta, inRange, "C1", "red"):
            axLinDelta.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)

    idealCum = rate * reads
    for ax in (axRawCum, axLinCum):
        ax.plot(reads, idealCum, "k--", linewidth=1, label=f"ideal (rate={rate:.1f} DN/read)")
        ax.legend(loc="upper left")
        ax.set_ylim(0, rate * N * 1.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Cumulative DN")
    for ax in (axRawDelta, axLinDelta):
        ax.axhline(rate, color="k", linestyle="--", linewidth=1, label=f"median rate={rate:.1f} DN/read")
        ax.legend(loc="upper right")
        ax.set_ylim(-rate * 0.5, rate * 2.0)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Delta DN")
        ax.set_xlabel("Read number")

    axRawCum.set_title(f"Uncorrected — accumulated flux ({K} pixels)")
    axLinCum.set_title(f"Linearized — accumulated flux ({K} pixels)")
    axRawDelta.set_title(f"Uncorrected — delta flux ({K} pixels)")
    axLinDelta.set_title(f"Linearized — delta flux ({K} pixels)")

    # --- Row 3 left: rejected pixel traces ---
    good = badPixelMask == 0
    if good.any():
        goodIdx = np.flatnonzero(good.ravel())
        rngGood = np.random.default_rng(seed)
        gSample = rngGood.choice(goodIdx, size=min(10000, len(goodIdx)), replace=False)
        gRows, gCols = gSample // W, gSample % W
        medianTrace = np.median(rawCum[:, gRows, gCols], axis=1)
    else:
        medianTrace = None

    notBorder = (badPixelMask & BORDER_PIX) == 0
    rngRej = np.random.default_rng(seed)

    medMax = float(medianTrace[-1]) if medianTrace is not None else 0.0
    brightThresh = brightFraction * medMax
    darkThresh = darkFraction * medMax

    # Non-monotonic
    nmMask = ((badPixelMask & NON_MONOTONIC) != 0) & notBorder
    nmIdx = np.flatnonzero(nmMask.ravel())
    nNM = len(nmIdx)
    if nNM > nPlot:
        nmIdx = rngRej.choice(nmIdx, size=nPlot, replace=False)
    nmRows, nmCols = nmIdx // W, nmIdx % W

    # Bright insufficient-points
    ipMask = ((badPixelMask & INSUFFICIENT_POINTS) != 0) & notBorder
    ipIdx = np.flatnonzero(ipMask.ravel())
    if len(ipIdx) > 0:
        ipR, ipC = ipIdx // W, ipIdx % W
        ipTraces = rawCum[:, ipR, ipC]
        keep = (ipTraces.max(axis=0) > brightThresh) | (ipTraces.min(axis=0) < darkThresh)
        ipIdx = ipIdx[keep]
        if len(ipIdx) > nPlot:
            ipIdx = rngRej.choice(ipIdx, size=nPlot, replace=False)
    ipRows, ipCols = (ipIdx // W, ipIdx % W) if len(ipIdx) > 0 else (np.array([], int), np.array([], int))

    # Other failed
    failFlags = INSUFFICIENT_POINTS | FIT_FAILED | NON_MONOTONIC
    otherMask = ((badPixelMask & failFlags) != 0) & notBorder & ~nmMask
    otherIdx = np.flatnonzero(otherMask.ravel())
    otherIdx = np.setdiff1d(otherIdx, ipIdx)
    if len(otherIdx) > nPlot:
        otherIdx = rngRej.choice(otherIdx, size=nPlot, replace=False)
    otherRows, otherCols = (otherIdx // W, otherIdx % W) if len(otherIdx) > 0 else (np.array([], int), np.array([], int))

    totalPlotted = len(nmRows) + len(ipRows) + len(otherRows)
    if totalPlotted == 0:
        axRejCum.text(0.5, 0.5, "No failed pixels", transform=axRejCum.transAxes,
                      ha="center", va="center", fontsize=14)
    else:
        for k in range(len(otherRows)):
            trace = rawCum[:, otherRows[k], otherCols[k]]
            lbl = "other failed" if k == 0 else None
            axRejCum.plot(reads, trace, color="C7", alpha=0.1, linewidth=0.4, label=lbl)
        for k in range(len(ipRows)):
            trace = rawCum[:, ipRows[k], ipCols[k]]
            lbl = f"INSUFF_PTS (max>{brightFraction:.0%} or all<{darkFraction:.0%} of median, N={len(ipRows)})" if k == 0 else None
            axRejCum.plot(reads, trace, color="C3", alpha=0.08, linewidth=0.5, label=lbl)
        for k in range(len(nmRows)):
            trace = rawCum[:, nmRows[k], nmCols[k]]
            lbl = f"NON_MONOTONIC (N={nNM})" if k == 0 else None
            axRejCum.plot(reads, trace, color="C0", alpha=0.3, linewidth=1.2, label=lbl)

    if medianTrace is not None:
        axRejCum.plot(reads, medianTrace, "k-", linewidth=1.5, label="median good pixel")
    axRejCum.legend(loc="upper left")
    axRejCum.set_xlabel("Read number")
    axRejCum.set_ylabel("Cumulative DN")
    axRejCum.set_title(f"Rejected pixel traces (N={totalPlotted})")
    axRejCum.grid(True, alpha=0.3)

    # --- Row 3 right: rejection bar chart ---
    totalPixels = H * W
    categories = [
        ("BORDER_PIX", (badPixelMask & BORDER_PIX > 0).sum(), "C7"),
        ("MASKED_BY_INPUT", (badPixelMask & MASKED_BY_INPUT > 0).sum(), "C8"),
        ("INSUFF_PTS", (badPixelMask & INSUFFICIENT_POINTS > 0).sum(), "C3"),
        ("FIT_FAILED", (badPixelMask & FIT_FAILED > 0).sum(), "C1"),
        ("NON_MONOTONIC", (badPixelMask & NON_MONOTONIC > 0).sum(), "C0"),
    ]
    labels = [c[0] for c in categories]
    counts = [int(c[1]) for c in categories]
    colors = [c[2] for c in categories]

    bars = axRejBar.bar(labels, counts, color=colors)
    axRejBar.set_ylabel("Pixel count")
    axRejBar.set_title("Rejection categories")
    axRejBar.tick_params(axis="x", rotation=30)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / totalPixels
        axRejBar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=8,
        )

    titleParts = []
    if detector:
        titleParts.append(f"det {detector}")
    if visit:
        titleParts.append(f"visit {visit}")
    titleParts.append(f"order={order}")
    if deviationLimit is not None:
        titleParts.append(f"deviationLimit={deviationLimit}, nRefReads={nRefReads}")
    if saturationLevel is not None:
        titleParts.append(f"saturationLevel={saturationLevel:.0f}")
    if deviationLimit is None and saturationLevel is None:
        titleParts.append("no clipping")
    fig.suptitle(", ".join(titleParts), fontsize=11)
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outPath}", flush=True)


def _segments(
    x: np.ndarray, y: np.ndarray, inRange: np.ndarray,
    colorIn: str, colorOut: str,
) -> list[tuple[tuple[np.ndarray, np.ndarray], str]]:
    """Split a trace into contiguous in-range and out-of-range segments.

    Returns a list of ((xSeg, ySeg), color) pairs. Adjacent segments share
    their boundary point so the plotted line has no gaps.
    """
    result = []
    n = len(x)
    if n == 0:
        return result
    i = 0
    while i < n:
        cur = bool(inRange[i])
        j = i + 1
        while j < n and bool(inRange[j]) == cur:
            j += 1
        # Include one extra point at each end to connect segments
        lo = i
        hi = min(j, n)  # exclusive
        if lo > 0:
            lo -= 1
        if hi < n:
            hi += 1
        color = colorIn if cur else colorOut
        result.append(((x[lo:hi], y[lo:hi]), color))
        i = j
    return result



def _plotFitRange(
    fitMin: np.ndarray,
    fitMax: np.ndarray,
    badPixelMask: np.ndarray,
    outPath: Path,
) -> None:
    """Plot 3: distributions of fitMin and fitMax, highlighting failed pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = badPixelMask == 0
    failFlags = INSUFFICIENT_POINTS | FIT_FAILED | NON_MONOTONIC
    failed = (badPixelMask & failFlags) != 0

    goodMin = fitMin[good].ravel()
    goodMax = fitMax[good].ravel()
    failMin = fitMin[failed].ravel()
    failMax = fitMax[failed].ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- Top: fitMin distribution ---
    allMin = fitMin.ravel()
    lo, hi = np.percentile(allMin[np.isfinite(allMin)], [0.5, 99.5])
    bins = np.linspace(lo, hi, 200)

    ax1.hist(goodMin, bins=bins, alpha=0.7, color="C0", label="good pixels")
    if len(failMin) > 0:
        ax1.hist(failMin, bins=bins, alpha=0.7, color="red", label="failed pixels")
    ax1.set_xlabel("fitMin (DN)")
    ax1.set_ylabel("Pixel count")
    ax1.set_title("Distribution of fitMin (lower bound of fitting range)")
    ax1.legend()
    ax1.set_yscale("log")

    # --- Bottom: fitMax distribution ---
    allMax = fitMax.ravel()
    lo, hi = np.percentile(allMax[np.isfinite(allMax)], [0.5, 99.5])
    bins = np.linspace(lo, hi, 200)

    ax2.hist(goodMax, bins=bins, alpha=0.7, color="C0", label="good pixels")
    if len(failMax) > 0:
        ax2.hist(failMax, bins=bins, alpha=0.7, color="red", label="failed pixels")
    ax2.set_xlabel("fitMax (DN)")
    ax2.set_ylabel("Pixel count")
    ax2.set_title("Distribution of fitMax (upper bound of fitting range)")
    ax2.legend()
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outPath}", flush=True)


def _loadInputRamp(dataDir: Path, noPhotodiode: bool) -> tuple[Ramp, Path]:
    """Load the .npz ramp and apply photodiode correction.

    Returns (correctedRamp, dataPath).
    """
    npzFiles = sorted(dataDir.glob("*.npz"))
    if not npzFiles:
        raise FileNotFoundError(f"No .npz files found in {dataDir}")
    dataPath = npzFiles[0]

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

    if noPhotodiode:
        print("  photodiode correction: DISABLED", flush=True)
        correctedDeltas = ramp.deltas
    else:
        scale = (photodiode[0] / photodiode).astype(np.float32)
        print(
            f"  photodiode scale range: min={scale.min():.6f} max={scale.max():.6f}",
            flush=True,
        )
        correctedDeltas = ramp.deltas * scale[:, None, None]

    correctedRamp = Ramp(deltas=correctedDeltas)
    _t("photodiode correction applied" if not noPhotodiode else "photodiode correction skipped", t1)

    return correctedRamp, dataPath


def _buildTag(summary: dict, pdTag: str) -> str:
    """Build a filename tag from the fit parameters stored in the FITS header."""
    order = summary.get("order", "?")
    tag = f"o{order}{pdTag}"
    devLimit = summary.get("deviationLimit")
    if devLimit is not None:
        tag += f"_dev{devLimit}"
    satLevel = summary.get("saturationLevel")
    if satLevel is not None:
        tag += f"_sat{int(satLevel)}"
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check relin on a real ramp")
    parser.add_argument("--fit", action="store_true",
                        help="Fit and save a linearity correction FITS file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots (reads existing FITS + input data)")
    parser.add_argument("--plot-format", type=str, default="png",
                        help="Plot file format: png, pdf, svg (default: png)")
    parser.add_argument("--nplot", type=int, default=1000,
                        help="Number of pixels to plot (default: 1000)")
    parser.add_argument("--deviation-limit", type=float, default=None,
                        help="Fractional deviation threshold for fit range clipping (default: None)")
    parser.add_argument("--deviation-start", type=float, default=0.5,
                        help="Fraction of reads before deviation limit is applied (default: 0.5)")
    parser.add_argument("--order", type=int, default=4,
                        help="Polynomial order (default: 4)")
    parser.add_argument("--saturation-level", type=float, default=None,
                        help="Saturation level in DN (default: None)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for pixel sampling (default: 0)")
    parser.add_argument("--no-photodiode", action="store_true",
                        help="Skip photodiode illumination-drift correction")
    parser.add_argument("--low-flux-fraction", type=float, default=0.5,
                        help="Reject pixels with refDelta below this fraction of median rate (default: 0.5)")
    parser.add_argument("--data-dir", type=str, default="examples/linearity/18734",
                        help="Directory containing the .npz ramp file (default: examples/linearity/18734)")
    args = parser.parse_args()

    if not args.fit and not args.plot:
        args.fit = True

    dataDir = Path(args.data_dir)
    detName = dataDir.name
    pdTag = "_pdOff" if args.no_photodiode else ""

    # Build config tag from CLI args for directory naming.
    cliTag = f"o{args.order}{pdTag}"
    if args.deviation_limit is not None:
        cliTag += f"_dev{args.deviation_limit}"
    if args.deviation_start != 0.5:
        cliTag += f"_ds{args.deviation_start}"
    if args.saturation_level is not None:
        cliTag += f"_sat{int(args.saturation_level)}"
    if args.low_flux_fraction != 0.5:
        cliTag += f"_lff{args.low_flux_fraction}"
    outDir = dataDir / cliTag
    fitsPath = outDir / f"{detName}_linearity.fits"

    # ---- Fit ----
    if args.fit:
        t0 = time.perf_counter()
        correctedRamp, dataPath = _loadInputRamp(dataDir, args.no_photodiode)
        H, W = correctedRamp.deltas.shape[1:]

        # Sample pixel indices from all interior pixels BEFORE fitting, so
        # the same pixels are reported regardless of clipping parameters.
        interior = np.ones((H, W), dtype=bool)
        interior[:4, :] = False
        interior[-4:, :] = False
        interior[:, :4] = False
        interior[:, -4:] = False
        rng = np.random.default_rng(args.seed)
        interiorIdx = np.flatnonzero(interior.ravel())
        sampleIdx = rng.choice(interiorIdx, size=min(10000, len(interiorIdx)), replace=False)
        rows = sampleIdx // W
        cols = sampleIdx % W

        from relin.models import PolynomialModel
        model = PolynomialModel(order=args.order)
        print(f"Fitting (blockSize=(512, 512), order={args.order}) ...", flush=True)
        correction = relin.fit(
            [correctedRamp], blockSize=(512, 512),
            model=model,
            deviationLimit=args.deviation_limit,
            deviationStart=args.deviation_start,
            saturationLevel=args.saturation_level,
            lowFluxFraction=args.low_flux_fraction,
        )
        t1 = _t("relin.fit", t0)

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

        # Extract visit number from filename (e.g. 18734_164220.npz -> 164220)
        parts = dataPath.stem.split("_", 1)
        if len(parts) == 2:
            correction.diagnostics.summary["visit"] = parts[1]
            correction.diagnostics.summary["detector"] = parts[0]

        outDir.mkdir(exist_ok=True)
        print(f"Saving to {fitsPath} ...", flush=True)
        relin.saveFits(fitsPath, correction)
        t1 = _t("relin.saveFits", t1)
        print(f"  FITS size: {fitsPath.stat().st_size / 1e6:.1f} MB", flush=True)

        print("Reloading FITS ...", flush=True)
        loaded = relin.loadFits(fitsPath)
        t1 = _t("relin.loadFits", t1)

        coefMatch = np.array_equal(loaded.coefficients, correction.coefficients)
        bpMatch = np.array_equal(loaded.badPixelMask, correction.badPixelMask)
        print(f"  coefficients bitwise-equal:   {coefMatch}", flush=True)
        print(f"  badPixelMask bitwise-equal:   {bpMatch}", flush=True)

        print("Applying correction to input ramp ...", flush=True)
        result = relin.apply(loaded, correctedRamp)
        t1 = _t("relin.apply", t1)

        good = loaded.badPixelMask == 0
        nGood = int(good.sum())
        print(
            f"  good pixels: {nGood}/{good.size} "
            f"({100 * nGood / good.size:.2f}%)",
            flush=True,
        )

        sampleGood = good[rows, cols]
        goodRows = rows[sampleGood]
        goodCols = cols[sampleGood]
        nGoodSampled = len(goodRows)

        trajectories = result.cumulativeLinear[:, goodRows, goodCols]
        n = np.arange(1, trajectories.shape[0] + 1, dtype=np.float64)

        slopes = np.einsum("n,nk->k", n, trajectories) / np.sum(n * n)
        residuals = trajectories - slopes[None, :] * n[:, None]
        rms = np.sqrt(np.mean(residuals**2, axis=0))
        print(
            f"  per-pixel linearity RMS over {nGoodSampled} sampled good pixels:",
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
        _t("done (fit)", t0)

    # ---- Plot ----
    if args.plot:
        t0 = time.perf_counter()
        if not fitsPath.exists():
            raise FileNotFoundError(
                f"No FITS file at {fitsPath}; run with --fit first"
            )

        correctedRamp, _ = _loadInputRamp(dataDir, args.no_photodiode)
        H, W = correctedRamp.deltas.shape[1:]

        interior = np.ones((H, W), dtype=bool)
        interior[:4, :] = False
        interior[-4:, :] = False
        interior[:, :4] = False
        interior[:, -4:] = False
        rng = np.random.default_rng(args.seed)
        interiorIdx = np.flatnonzero(interior.ravel())
        sampleIdx = rng.choice(interiorIdx, size=min(10000, len(interiorIdx)), replace=False)
        rows = sampleIdx // W
        cols = sampleIdx % W

        print(f"Loading {fitsPath} ...", flush=True)
        loaded = relin.loadFits(fitsPath)
        t1 = _t("relin.loadFits", t0)

        summary = loaded.diagnostics.summary
        order = summary.get("order", "?")
        deviationLimit = summary.get("deviationLimit")
        saturationLevel = summary.get("saturationLevel")
        nRefReads = summary.get("nRefReads", 5)

        print("Applying correction to input ramp ...", flush=True)
        result = relin.apply(loaded, correctedRamp)
        t1 = _t("relin.apply", t1)

        rawCum = np.cumsum(correctedRamp.deltas.astype(np.float32), axis=0)

        print("Generating diagnostic plots ...", flush=True)
        _plotDiagnostic(
            rawCum=rawCum,
            linCum=result.cumulativeLinear,
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            rows=rows,
            cols=cols,
            rate=float(np.median(correctedRamp.deltas[0])),
            outPath=outDir / f"diagnostic.{args.plot_format}",
            nPlot=args.nplot,
            order=order,
            deviationLimit=deviationLimit,
            nRefReads=nRefReads,
            saturationLevel=saturationLevel,
            seed=args.seed,
            detector=str(summary.get("detector", detName)),
            visit=str(summary.get("visit", "")),
        )
        _plotFitRange(
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            outPath=outDir / f"diagnostic_fit_range.{args.plot_format}",
        )
        _t("done (plot)", t0)


if __name__ == "__main__":
    main()
