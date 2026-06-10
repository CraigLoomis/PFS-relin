"""Sanity check: run nirLinearity on a real 4096x4096x29 lab ramp.

Loads the example NPZ, applies a standard photodiode correction
(illumination-drift normalization to the first read), fits, saves FITS,
reloads, applies, and reports residuals.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import nirLinearity
from nirLinearity.loaders import loadNpz
from nirLinearity.types import (
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
    detector: str = "",
    visit: str = "",
) -> None:
    """Combined diagnostic: good-pixel before/after (rows 1-2) + rejected traces (row 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    H, W = badPixelMask.shape
    N = rawCum.shape[0]
    reads = np.arange(N)

    fig = plt.figure(figsize=(16, 20), layout="constrained")
    gs = fig.add_gridspec(4, 2, hspace=0.1)
    # Rows 1-3 share x; row 4 (rejected/good-pop) is independent.
    axRawCum = fig.add_subplot(gs[0, 0])
    axLinCum = fig.add_subplot(gs[0, 1], sharex=axRawCum)
    axRawDelta = fig.add_subplot(gs[1, 0], sharex=axRawCum)
    axLinDelta = fig.add_subplot(gs[1, 1], sharex=axRawCum)
    axGoodPopRaw = fig.add_subplot(gs[2, 0], sharex=axRawCum)
    axGoodPopLin = fig.add_subplot(gs[2, 1], sharex=axRawCum)
    axRejCum = fig.add_subplot(gs[3, 0])
    axRejBar = fig.add_subplot(gs[3, 1])

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

        # Deltas: skip read 0 (implicit zero); deltas[i] is read i+1 minus i.
        rawDelta = np.diff(mTrace)
        linDelta = np.diff(tTrace)
        deltaReads = reads[1:]
        deltaInRange = inRange[1:]

        for seg, color in _segments(reads, mTrace, inRange, "C0", "red"):
            axRawCum.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(reads, tTrace, inRange, "C1", "red"):
            axLinCum.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(deltaReads, rawDelta, deltaInRange, "C0", "red"):
            axRawDelta.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(deltaReads, linDelta, deltaInRange, "C1", "red"):
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
    notBorder = (badPixelMask & BORDER_PIX) == 0
    rngRej = np.random.default_rng(seed)

    # Median good-pixel trace + its initial rate (reference for "hot" threshold).
    # Use first 3 deltas — short enough to catch hot pixels before they saturate.
    nHotRef = min(3, rawCum.shape[0] - 1)
    medianTrace = None
    goodInitRate = 0.0
    if good.any():
        goodIdx = np.flatnonzero(good.ravel())
        rngGood = np.random.default_rng(seed)
        gSample = rngGood.choice(goodIdx, size=min(10000, len(goodIdx)), replace=False)
        gRows, gCols = gSample // W, gSample % W
        medianTrace = np.median(rawCum[:, gRows, gCols], axis=1)
        if nHotRef > 0:
            goodInitRate = float(np.median(np.diff(medianTrace[:nHotRef + 1])))
    medMax = float(medianTrace[-1]) if medianTrace is not None else 0.0

    # Candidate set: every non-good, non-border pixel.
    rejMask = ~good & notBorder
    rejIdx = np.flatnonzero(rejMask.ravel())
    nRej = len(rejIdx)

    # Precompute per-pixel peak, initial rate, and RMS distance from median good.
    if nRej > 0:
        rejR = rejIdx // W
        rejC = rejIdx % W
        rejTraces = rawCum[:, rejR, rejC].astype(np.float64, copy=False)  # (N, nRej)
        rejPeak = rejTraces.max(axis=0)
        if nHotRef > 0:
            rejInitRate = np.median(np.diff(rejTraces[:nHotRef + 1], axis=0), axis=0)
        else:
            rejInitRate = np.zeros(nRej)
        if medianTrace is not None:
            rejDist = np.sqrt(((rejTraces - medianTrace[:, None]) ** 2).mean(axis=0))
        else:
            rejDist = np.full(nRej, np.inf)
    else:
        rejTraces = np.empty((rawCum.shape[0], 0), dtype=np.float64)
        rejPeak = np.array([])
        rejInitRate = np.array([])
        rejDist = np.array([])

    # Per-category plot specs: (label, color, alpha, linewidth, selectionIntoRej, foundCount)
    categoryPlot: list[tuple[str, str, float, float, np.ndarray, int]] = []

    # (a) Random sample of all rejected — background context.
    if nRej > 0:
        if nRej > nPlot:
            bgSel = rngRej.choice(nRej, size=nPlot, replace=False)
        else:
            bgSel = np.arange(nRej)
        categoryPlot.append((
            "all rejected (random sample)", "0.5", 0.12, 0.4, bgSel, nRej,
        ))

    # (b) Closest to the median good pixel — top 200 by smallest RMS distance.
    closestN = 200
    if nRej > 0 and medianTrace is not None and np.isfinite(rejDist).any():
        K = min(closestN, nRej)
        partIdx = np.argpartition(rejDist, K - 1)[:K]
        closeSel = partIdx[np.argsort(rejDist[partIdx])]
    else:
        closeSel = np.array([], dtype=int)
    categoryPlot.append((
        f"closest to median good (top {closestN} by RMS distance)",
        "C2", 0.15, 0.8, closeSel, len(closeSel),
    ))

    # (c) Dead — peak signal < 10% of median good endpoint.
    deadThresh = 0.1 * medMax
    deadSel = np.flatnonzero(rejPeak < deadThresh)
    nDead = len(deadSel)
    if nDead > nPlot:
        deadSel = rngRej.choice(deadSel, size=nPlot, replace=False)
    categoryPlot.append((
        f"dead (peak < {deadThresh:,.0f} DN, 10% of good endpoint {medMax:,.0f})",
        "C4", 0.12, 0.4, deadSel, nDead,
    ))

    # (d) Hot — initial rate over the first nHotRef deltas > 2× good initial rate.
    hotThresh = 2.0 * goodInitRate
    hotSel = np.flatnonzero(rejInitRate > hotThresh)
    nHot = len(hotSel)
    if nHot > nPlot:
        hotSel = rngRej.choice(hotSel, size=nPlot, replace=False)
    categoryPlot.append((
        f"hot (first-{nHotRef} rate > {hotThresh:,.0f} DN/read, 2× good {goodInitRate:,.0f})",
        "C3", 0.5, 1.0, hotSel, nHot,
    ))

    # (e) NON_MONOTONIC — only if any qualifying pixels exist.
    nmMask = ((badPixelMask & NON_MONOTONIC) != 0) & notBorder
    if nmMask.any():
        nmGlobal = np.flatnonzero(nmMask.ravel())
        nNM = len(nmGlobal)
        # NM pixels are a subset of rejIdx — map global → position in rejIdx.
        nmInRej = np.searchsorted(rejIdx, nmGlobal)
        if nNM > nPlot:
            nmInRej = rngRej.choice(nmInRej, size=nPlot, replace=False)
        categoryPlot.append((
            "NON_MONOTONIC", "magenta", 0.6, 1.2, nmInRej, nNM,
        ))

    # --- Plot, in list order so emphasized categories overlay the background. ---
    totalPlotted = sum(len(c[4]) for c in categoryPlot)
    if totalPlotted == 0 and medianTrace is None:
        axRejCum.text(0.5, 0.5, "No rejected pixels", transform=axRejCum.transAxes,
                      ha="center", va="center", fontsize=14)
    else:
        for _label, color, alpha, lw, sel, _total in categoryPlot:
            for k in sel:
                axRejCum.plot(reads, rejTraces[:, k], color=color, alpha=alpha, linewidth=lw)

    if medianTrace is not None:
        axRejCum.plot(reads, medianTrace, "k-", linewidth=1.5)

    # Legend with proxy handles so swatches are visible at full saturation.
    legendHandles = []
    for label, color, _alpha, _lw, sel, total in categoryPlot:
        legendHandles.append(Line2D(
            [0], [0], color=color, linewidth=2.0,
            label=f"{label}: plotted {len(sel):,} / found {total:,}",
        ))
    if medianTrace is not None:
        legendHandles.append(Line2D(
            [0], [0], color="k", linewidth=1.5, label="median good pixel",
        ))
    axRejCum.legend(handles=legendHandles, loc="upper left", framealpha=0.9, fontsize=8)
    axRejCum.set_xlabel("Read number")
    axRejCum.set_ylabel("Cumulative DN")
    axRejCum.set_title(f"Rejected pixel traces — plotted {totalPlotted:,} of {nRej:,} rejected")
    axRejCum.grid(True, alpha=0.3)

    # --- Row 3 right: non-rejected (good) pixel populations — raw traces. ---
    goodCategoryPlot: list[tuple[str, str, float, float, np.ndarray, np.ndarray, int]] = []

    if good.any():
        nGood = int(good.sum())
        endpoints = rawCum[-1]  # (H, W) view of final cumulative read
        if nHotRef > 0:
            firstRate = (rawCum[nHotRef] - rawCum[0]) / nHotRef  # (H, W)
        else:
            firstRate = np.zeros_like(endpoints)

        # (a) Faint — good with endpoint < 50% of median good endpoint.
        faintThresh = 0.5 * medMax
        faintFlat = np.flatnonzero((good & (endpoints < faintThresh)).ravel())
        nFaint = len(faintFlat)
        if nFaint > nPlot:
            faintFlat = rngRej.choice(faintFlat, size=nPlot, replace=False)
        goodCategoryPlot.append((
            f"faint (endpoint < {faintThresh:,.0f} DN, 50% of median {medMax:,.0f})",
            "C0", 0.04, 0.4, faintFlat // W, faintFlat % W, nFaint,
        ))

        # (c) Bright — good with endpoint > 150% of median good endpoint.
        brightThresh = 1.5 * medMax
        brightFlat = np.flatnonzero((good & (endpoints > brightThresh)).ravel())
        nBright = len(brightFlat)
        if nBright > nPlot:
            brightFlat = rngRej.choice(brightFlat, size=nPlot, replace=False)
        goodCategoryPlot.append((
            f"bright (endpoint > {brightThresh:,.0f} DN, 150% of median {medMax:,.0f})",
            "C1", 0.4, 0.8, brightFlat // W, brightFlat % W, nBright,
        ))

        # (d) Hot — good with first-nHotRef rate > 1.5× the median good initial rate.
        hotGoodThresh = 1.5 * goodInitRate
        hotFlat = np.flatnonzero((good & (firstRate > hotGoodThresh)).ravel())
        nHotGood = len(hotFlat)
        if nHotGood > nPlot:
            hotFlat = rngRej.choice(hotFlat, size=nPlot, replace=False)
        goodCategoryPlot.append((
            f"hot (first-{nHotRef} rate > {hotGoodThresh:,.0f} DN/read, 1.5× median {goodInitRate:,.0f})",
            "C2", 0.15, 0.8, hotFlat // W, hotFlat % W, nHotGood,
        ))

    totalGoodPlotted = sum(len(c[4]) for c in goodCategoryPlot)
    if totalGoodPlotted == 0 and medianTrace is None:
        axRejBar.text(0.5, 0.5, "No good pixels", transform=axRejBar.transAxes,
                      ha="center", va="center", fontsize=14)
    else:
        for _label, color, alpha, lw, rrows, rcols, _total in goodCategoryPlot:
            if len(rrows) == 0:
                continue
            sub = rawCum[:, rrows, rcols]  # (N, k)
            axRejBar.plot(reads, sub, color=color, alpha=alpha, linewidth=lw)

    if medianTrace is not None:
        axRejBar.plot(reads, medianTrace, "k-", linewidth=1.5)

    legendHandles2 = []
    for label, color, _alpha, _lw, rrows, _rcols, total in goodCategoryPlot:
        legendHandles2.append(Line2D(
            [0], [0], color=color, linewidth=2.0,
            label=f"{label}: plotted {len(rrows):,} / found {total:,}",
        ))
    if medianTrace is not None:
        legendHandles2.append(Line2D(
            [0], [0], color="k", linewidth=1.5, label="median good pixel",
        ))
    axRejBar.legend(handles=legendHandles2, loc="upper left", framealpha=0.9, fontsize=8)
    axRejBar.set_xlabel("Read number")
    axRejBar.set_ylabel("Cumulative DN")
    nGoodTotal = int(good.sum())
    axRejBar.set_title(f"Good pixel traces — plotted {totalGoodPlotted:,} of {nGoodTotal:,} good")
    axRejBar.grid(True, alpha=0.3)

    # --- Row 3: same good-population pixels, plotted in row-1 ("accumulated flux") style. ---
    # Combine all good-population pixels (faint+bright+hot) and render each pixel
    # exactly like row 1: in-range segments in C0/C1, out-of-range in red.
    popRows: list[np.ndarray] = []
    popCols: list[np.ndarray] = []
    for _label, _color, _alpha, _lw, rrows, rcols, _total in goodCategoryPlot:
        if len(rrows) > 0:
            popRows.append(np.asarray(rrows))
            popCols.append(np.asarray(rcols))
    if popRows:
        popR = np.concatenate(popRows)
        popC = np.concatenate(popCols)
    else:
        popR = np.array([], dtype=int)
        popC = np.array([], dtype=int)

    for r, c in zip(popR, popC):
        mTrace = rawCum[:, r, c]
        tTrace = linCum[:, r, c]
        fMin, fMax = fitMin[r, c], fitMax[r, c]
        inRange = (mTrace >= fMin) & (mTrace <= fMax)
        for seg, color in _segments(reads, mTrace, inRange, "C0", "red"):
            axGoodPopRaw.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)
        for seg, color in _segments(reads, tTrace, inRange, "C1", "red"):
            axGoodPopLin.plot(seg[0], seg[1], color=color, alpha=0.08, linewidth=0.5)

    for ax in (axGoodPopRaw, axGoodPopLin):
        ax.plot(reads, idealCum, "k--", linewidth=1, label=f"ideal (rate={rate:.1f} DN/read)")
        ax.legend(loc="upper left")
        ax.set_ylim(0, rate * N * 1.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Cumulative DN")
        ax.set_xlabel("Read number")
    axGoodPopRaw.set_title(f"Uncorrected — accumulated flux ({len(popR)} good-population pixels)")
    axGoodPopLin.set_title(f"Linearized — accumulated flux ({len(popR)} good-population pixels)")

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
    fitMax: np.ndarray,
    badPixelMask: np.ndarray,
    rawCum: np.ndarray,
    outPath: Path,
    loOverride: float | None = None,
    hiOverride: float | None = None,
) -> None:
    """Plot 3: stacked histograms — good-pixel fitMax (top, with per-detector
    range overrides) and rejected-pixel raw cumulative max (bottom, anchored at
    zero so the floor is visible). Pixels outside each range appear in single
    overflow columns at the edges.

    fitMax in the FITS is unreliable for failed fits, so rejected pixels use
    ``rawCum.max(axis=0)`` instead — their actual max signal level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = badPixelMask == 0
    notBorder = (badPixelMask & BORDER_PIX) == 0
    rejected = ~good & notBorder

    goodMax = fitMax[good].ravel()
    rawMax = rawCum.max(axis=0)
    rejMax = rawMax[rejected].ravel()

    fig, (axGood, axRej) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    nMainBins = 200
    overflowColor = "C2"

    # --- Top: good pixels (fitMax) ---
    autoLoG, autoHiG = np.percentile(goodMax, [0.5, 99.5]) if len(goodMax) else (0.0, 1.0)
    loG = float(loOverride) if loOverride is not None else float(autoLoG)
    hiG = float(hiOverride) if hiOverride is not None else float(autoHiG)
    edgesG = np.linspace(loG, hiG, nMainBins + 1)
    binWG = edgesG[1] - edgesG[0]

    axGood.hist(goodMax, bins=edgesG, alpha=0.7, color="C0", label="good pixels (fitMax)")
    nBelowG = int((goodMax < loG).sum())
    nAboveG = int((goodMax >= hiG).sum())
    if nBelowG > 0:
        axGood.bar(loG - binWG, nBelowG, width=binWG, color=overflowColor, alpha=0.85,
                   align="edge", label=f"clipped < {loG:,.0f} ({nBelowG:,})")
    if nAboveG > 0:
        axGood.bar(hiG, nAboveG, width=binWG, color=overflowColor, alpha=0.85,
                   align="edge", label=f"clipped ≥ {hiG:,.0f} ({nAboveG:,})")
    axGood.axvline(loG, color=overflowColor, linewidth=0.6, linestyle=":", alpha=0.8)
    axGood.axvline(hiG, color=overflowColor, linewidth=0.6, linestyle=":", alpha=0.8)
    axGood.set_xlabel("fitMax (DN)")
    axGood.set_ylabel("Pixel count")
    axGood.set_title(f"Good pixels — fitMax distribution (n={len(goodMax):,})")
    axGood.legend()
    axGood.set_yscale("log")

    # --- Bottom: rejected pixels (raw cumulative max), anchored at 0 ---
    loR = 0.0
    if len(rejMax) > 0:
        autoHiR = float(np.percentile(rejMax, 99.5))
    else:
        autoHiR = float(hiOverride) if hiOverride is not None else float(hiG)
    hiR = float(hiOverride) if hiOverride is not None else autoHiR
    edgesR = np.linspace(loR, hiR, nMainBins + 1)
    binWR = edgesR[1] - edgesR[0]

    if len(rejMax) > 0:
        axRej.hist(rejMax, bins=edgesR, alpha=0.7, color="red", label="rejected pixels (raw max)")
        nAboveR = int((rejMax >= hiR).sum())
        if nAboveR > 0:
            axRej.bar(hiR, nAboveR, width=binWR, color=overflowColor, alpha=0.85,
                      align="edge", label=f"clipped ≥ {hiR:,.0f} ({nAboveR:,})")
        axRej.axvline(hiR, color=overflowColor, linewidth=0.6, linestyle=":", alpha=0.8)
    else:
        axRej.text(0.5, 0.5, "No rejected pixels", transform=axRej.transAxes,
                   ha="center", va="center", fontsize=14)
    axRej.set_xlabel("max cumulative read (DN)")
    axRej.set_ylabel("Pixel count")
    axRej.set_title(f"Rejected (non-border bad) pixels — raw max (n={len(rejMax):,})")
    axRej.set_xlim(0, hiR + binWR)
    axRej.legend()
    axRej.set_yscale("log")

    fig.savefig(outPath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outPath}", flush=True)


def _plotFitRangeSpatial(
    fitMax: np.ndarray,
    badPixelMask: np.ndarray,
    outPath: Path,
    loOverride: float | None = None,
    hiOverride: float | None = None,
    detector: str = "",
    visit: str = "",
) -> None:
    """Spatial companion to _plotFitRange: fitMax heatmap + rejected-pixel
    density map. Uses origin='lower' (pixel (0,0) = lower-left of detector)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = badPixelMask == 0
    notBorder = (badPixelMask & BORDER_PIX) == 0
    rejected = ~good & notBorder
    H, W = badPixelMask.shape

    fig, (axMap, axDensity) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    extent = [0, W, 0, H]

    fmShow = np.where(good, fitMax, np.nan)
    if loOverride is not None:
        vmin = float(loOverride)
    else:
        vmin = float(np.nanpercentile(fmShow, 0.5))
    if hiOverride is not None:
        vmax = float(hiOverride)
    else:
        vmax = float(np.nanpercentile(fmShow, 99.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("red")
    im0 = axMap.imshow(fmShow, origin="lower", cmap=cmap, aspect="equal",
                       vmin=vmin, vmax=vmax, extent=extent)
    axMap.set_title("fitMax (good pixels colored; bad/rejected in red)")
    axMap.set_xlabel("column")
    axMap.set_ylabel("row")
    plt.colorbar(im0, ax=axMap, label="fitMax (DN)")

    bs = 8
    H_b, W_b = H // bs, W // bs
    density = rejected.reshape(H_b, bs, W_b, bs).sum(axis=(1, 3)).astype(np.int32)
    im1 = axDensity.imshow(density, origin="lower", cmap="hot", aspect="equal",
                           extent=extent)
    axDensity.set_title(f"Rejected (non-border bad) pixel density\n"
                        f"{bs}×{bs} blocks; total {int(rejected.sum()):,}")
    axDensity.set_xlabel("column")
    axDensity.set_ylabel("row")
    plt.colorbar(im1, ax=axDensity, label=f"count per {bs}×{bs} block")

    titleParts = []
    if detector:
        titleParts.append(f"det {detector}")
    if visit:
        titleParts.append(f"visit {visit}")
    if titleParts:
        fig.suptitle(", ".join(titleParts), fontsize=11)

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
        f"  reads: shape={ramp.reads.shape} dtype={ramp.reads.dtype} "
        f"min={ramp.reads.min():.4g} max={ramp.reads.max():.4g}",
        flush=True,
    )
    if photodiode.size == 0:
        print("  photodiode: (not recorded for this ramp)", flush=True)
        noPhotodiode = True
    else:
        print(
            f"  photodiode[0..2]={photodiode[:3]} photodiode[-1]={photodiode[-1]}",
            flush=True,
        )

    if noPhotodiode:
        print("  photodiode correction: DISABLED", flush=True)
        correctedReads = ramp.reads
    else:
        # Scale each read's increment by the photodiode ratio, then re-accumulate.
        scale = (photodiode[0] / photodiode).astype(np.float32)  # (N,)
        print(
            f"  photodiode scale range: min={scale.min():.6f} max={scale.max():.6f}",
            flush=True,
        )
        deltas = np.diff(ramp.reads, axis=0)  # (N, H, W)
        correctedReads = np.empty_like(ramp.reads)
        correctedReads[0] = 0.0
        np.cumsum(deltas * scale[:, None, None], axis=0, out=correctedReads[1:])

    correctedRamp = Ramp(reads=correctedReads)
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
    parser = argparse.ArgumentParser(description="Sanity check nirLinearity on a real ramp")
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
    parser.add_argument("--fitrange-min", type=float, default=None,
                        help="Lower bound (DN) for the fit-range histogram (default: auto p0.5)")
    parser.add_argument("--fitrange-max", type=float, default=None,
                        help="Upper bound (DN) for the fit-range histogram (default: auto p99.5)")
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
        H, W = correctedRamp.reads.shape[1:]

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

        from nirLinearity.models import PolynomialModel
        model = PolynomialModel(order=args.order)
        print(f"Fitting (order={args.order}) ...", flush=True)
        correction = nirLinearity.fit(
            [correctedRamp],
            model=model,
            deviationLimit=args.deviation_limit,
            deviationStart=args.deviation_start,
            saturationLevel=args.saturation_level,
            lowFluxFraction=args.low_flux_fraction,
        )
        t1 = _t("nirLinearity.fit", t0)

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
        nirLinearity.saveFits(fitsPath, correction)
        t1 = _t("nirLinearity.saveFits", t1)
        print(f"  FITS size: {fitsPath.stat().st_size / 1e6:.1f} MB", flush=True)

        print("Reloading FITS ...", flush=True)
        loaded = nirLinearity.loadFits(fitsPath)
        t1 = _t("nirLinearity.loadFits", t1)

        coefMatch = np.array_equal(loaded.coefficients, correction.coefficients)
        bpMatch = np.array_equal(loaded.badPixelMask, correction.badPixelMask)
        print(f"  coefficients bitwise-equal:   {coefMatch}", flush=True)
        print(f"  badPixelMask bitwise-equal:   {bpMatch}", flush=True)

        print("Applying correction to input ramp ...", flush=True)
        result = nirLinearity.apply(loaded, correctedRamp)
        t1 = _t("nirLinearity.apply", t1)

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

        from nirLinearity.types import ABOVE_VALID_RANGE, BELOW_VALID_RANGE
        belowCount = int((result.badPixelMask & BELOW_VALID_RANGE > 0).sum())
        aboveCount = int((result.badPixelMask & ABOVE_VALID_RANGE > 0).sum())
        nPix = result.badPixelMask.size
        print(f"  below-range pixel fraction: {belowCount / nPix:.4f}", flush=True)
        print(f"  above-range pixel fraction: {aboveCount / nPix:.4f}", flush=True)
        _t("done (fit)", t0)

    # ---- Plot ----
    if args.plot:
        t0 = time.perf_counter()
        if not fitsPath.exists():
            raise FileNotFoundError(
                f"No FITS file at {fitsPath}; run with --fit first"
            )

        correctedRamp, _ = _loadInputRamp(dataDir, args.no_photodiode)
        H, W = correctedRamp.reads.shape[1:]

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
        loaded = nirLinearity.loadFits(fitsPath)
        t1 = _t("nirLinearity.loadFits", t0)

        summary = loaded.diagnostics.summary
        order = summary.get("order", "?")
        deviationLimit = summary.get("deviationLimit")
        saturationLevel = summary.get("saturationLevel")
        nRefReads = summary.get("nRefReads", 5)

        print("Applying correction to input ramp ...", flush=True)
        result = nirLinearity.apply(loaded, correctedRamp)
        t1 = _t("nirLinearity.apply", t1)

        rawCum = correctedRamp.reads.astype(np.float32)

        print("Generating diagnostic plots ...", flush=True)
        _plotDiagnostic(
            rawCum=rawCum,
            linCum=result.cumulativeLinear,
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            rows=rows,
            cols=cols,
            rate=float(np.median(correctedRamp.reads[2] - correctedRamp.reads[1])),
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
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            rawCum=rawCum,
            outPath=outDir / f"diagnostic_fit_range.{args.plot_format}",
            loOverride=args.fitrange_min,
            hiOverride=args.fitrange_max,
        )
        _plotFitRangeSpatial(
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            outPath=outDir / f"diagnostic_fit_range_spatial.{args.plot_format}",
            loOverride=args.fitrange_min,
            hiOverride=args.fitrange_max,
            detector=str(summary.get("detector", detName)),
            visit=str(summary.get("visit", "")),
        )
        _t("done (plot)", t0)


if __name__ == "__main__":
    main()
