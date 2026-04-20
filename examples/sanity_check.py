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


def _plotBeforeAfter(
    rawCum: np.ndarray,
    linCum: np.ndarray,
    fitMin: np.ndarray,
    fitMax: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    rate: float,
    outPath: Path,
    nPlot: int = 1000,
) -> None:
    """Plot 1: before/after linearization for random good pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    K = min(nPlot, len(rows))
    pick = rng.choice(len(rows), size=K, replace=False)
    pRows, pCols = rows[pick], cols[pick]

    N = rawCum.shape[0]
    reads = np.arange(1, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axRawCum, axLinCum = axes[0, 0], axes[0, 1]
    axRawDelta, axLinDelta = axes[1, 0], axes[1, 1]

    # Precompute all traces for consistent plotting
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

    # Reference lines
    idealCum = rate * reads
    for ax in (axRawCum, axLinCum):
        ax.plot(reads, idealCum, "k--", linewidth=1, label="ideal (rate * n)")
        ax.legend(loc="upper left")
        ax.set_ylim(0, rate * N * 1.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Cumulative DN")
    for ax in (axRawDelta, axLinDelta):
        ax.axhline(rate, color="k", linestyle="--", linewidth=1, label="median rate")
        ax.legend(loc="upper right")
        ax.set_ylim(-rate * 0.5, rate * 2.0)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Delta DN")
        ax.set_xlabel("Read number")

    axRawCum.set_title(f"Uncorrected — accumulated flux ({K} pixels)")
    axLinCum.set_title(f"Linearized — accumulated flux ({K} pixels)")
    axRawDelta.set_title(f"Uncorrected — delta flux ({K} pixels)")
    axLinDelta.set_title(f"Linearized — delta flux ({K} pixels)")

    fig.tight_layout()
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


def _plotRejections(
    rawCum: np.ndarray,
    badPixelMask: np.ndarray,
    outPath: Path,
    nPlot: int = 1000,
) -> None:
    """Plot 2: rejection bar chart and failed-pixel traces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = badPixelMask.shape
    totalPixels = H * W

    # --- Top: bar chart ---
    categories = [
        ("Good", (badPixelMask == 0).sum()),
        ("MASKED_BY_INPUT", (badPixelMask & MASKED_BY_INPUT > 0).sum()),
        ("INSUFFICIENT_POINTS", (badPixelMask & INSUFFICIENT_POINTS > 0).sum()),
        ("FIT_FAILED", (badPixelMask & FIT_FAILED > 0).sum()),
        ("NON_MONOTONIC", (badPixelMask & NON_MONOTONIC > 0).sum()),
    ]
    labels = [c[0] for c in categories]
    counts = [int(c[1]) for c in categories]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    bars = ax1.bar(labels, counts, color=["C2", "C7", "C3", "C1", "C0"])
    ax1.set_ylabel("Pixel count")
    ax1.set_title("Fit rejection categories (flags are not mutually exclusive)")
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / totalPixels
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=9,
        )

    # --- Bottom: failed pixel traces ---
    # Exclude MASKED_BY_INPUT (no useful signal)
    failFlags = INSUFFICIENT_POINTS | FIT_FAILED | NON_MONOTONIC
    failed = (badPixelMask & failFlags) != 0
    failIdx = np.flatnonzero(failed.ravel())

    if len(failIdx) == 0:
        ax2.text(0.5, 0.5, "No failed pixels", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14)
    else:
        rng = np.random.default_rng(42)
        K = min(nPlot, len(failIdx))
        pick = rng.choice(len(failIdx), size=K, replace=False)
        sampleIdx = failIdx[pick]
        fRows = sampleIdx // W
        fCols = sampleIdx % W
        flags = badPixelMask[fRows, fCols]

        N = rawCum.shape[0]
        reads = np.arange(1, N + 1)

        flagColors = {
            FIT_FAILED: ("C1", "FIT_FAILED"),
            NON_MONOTONIC: ("C0", "NON_MONOTONIC"),
            INSUFFICIENT_POINTS: ("C3", "INSUFFICIENT_POINTS"),
        }
        plotted = set()
        for k in range(K):
            trace = rawCum[:, fRows[k], fCols[k]]
            # Use highest-priority flag for color
            for flag, (color, label) in flagColors.items():
                if flags[k] & flag:
                    lbl = label if label not in plotted else None
                    ax2.plot(reads, trace, color=color, alpha=0.1,
                             linewidth=0.5, label=lbl)
                    plotted.add(label)
                    break

        ax2.legend(loc="upper left")

    ax2.set_xlabel("Read number")
    ax2.set_ylabel("Cumulative DN")
    ax2.set_title(f"Raw cumulative flux for failed pixels (N={K if len(failIdx) > 0 else 0})")

    fig.tight_layout()
    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outPath}", flush=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check relin on a real ramp")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic PNGs")
    parser.add_argument("--nplot", type=int, default=1000,
                        help="Number of pixels to plot (default: 1000)")
    parser.add_argument("--deviation-limit", type=float, default=0.10,
                        help="Fractional deviation threshold for fit range clipping (default: 0.10)")
    args = parser.parse_args()

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
    correction = relin.fit(
        [correctedRamp], blockSize=(512, 512),
        deviationLimit=args.deviation_limit,
    )
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

    if args.plot:
        print("Generating diagnostic plots ...", flush=True)
        rawCum = np.cumsum(correctedRamp.deltas.astype(np.float32), axis=0)
        plotDir = fitsPath.parent

        _plotBeforeAfter(
            rawCum=rawCum,
            linCum=result.cumulativeLinear,
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            rows=rows,
            cols=cols,
            rate=float(np.median(correctedRamp.deltas[0])),
            outPath=plotDir / "diagnostic_before_after.png",
            nPlot=args.nplot,
        )
        _plotRejections(
            rawCum=rawCum,
            badPixelMask=loaded.badPixelMask,
            outPath=plotDir / "diagnostic_rejections.png",
            nPlot=args.nplot,
        )
        _plotFitRange(
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            outPath=plotDir / "diagnostic_fit_range.png",
        )
        t1 = _t("plots", t1)

    t1 = _t("done", t0)


if __name__ == "__main__":
    main()
