"""Fit, save, reload, apply, and diagnose a linearity correction on a lab ramp.

Runs the full chain on a real 4096x4096 up-the-ramp exposure and reports
residuals and per-population diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lsst.obs.pfs.h4Linearity as nirLinearity
import numpy as np
from lsst.obs.pfs.h4Linearity.types import (
    BORDER_PIX,
    FIT_FAILED,
    HIGH_FIT_RESIDUAL,
    INSUFFICIENT_POINTS,
    MASKED_BY_INPUT,
    NON_MONOTONIC,
    Ramp,
)

from fitLinearity.loader import loadCorrectedRamp


@dataclass
class SanityCheckConfig:
    """Fit and plot parameters for one sanity-check run."""

    order: int
    deviationLimit: float | None
    deviationStart: float
    saturationLevel: float | None
    lowFluxFraction: float
    saturationKnee: float | None
    badLinearityMultiplier: float | None
    noPhotodiode: bool
    seed: int
    nplot: int
    plotFormat: str
    fitrangeMin: float | None
    fitrangeMax: float | None


def cliTag(config: SanityCheckConfig) -> str:
    """Build the output-directory name describing this run's fit configuration.

    Only non-default parameters appear, so a default run tags as ``o4``.
    """
    tag = f"o{config.order}"
    if config.noPhotodiode:
        tag += "_pdOff"
    if config.deviationLimit is not None:
        tag += f"_dev{config.deviationLimit}"
    if config.deviationStart != 0.5:
        tag += f"_ds{config.deviationStart}"
    if config.saturationLevel is not None:
        tag += f"_sat{int(config.saturationLevel)}"
    if config.lowFluxFraction != 0.5:
        tag += f"_lff{config.lowFluxFraction}"
    if config.saturationKnee != 0.5:
        tag += f"_knee{config.saturationKnee}" if config.saturationKnee is not None else "_kneeOff"
    if config.badLinearityMultiplier != 5.0:
        tag += (
            f"_blm{config.badLinearityMultiplier}"
            if config.badLinearityMultiplier is not None
            else "_blmOff"
        )
    return tag


def _t(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"  [{now - t0:7.2f}s] {label}", flush=True)
    return now


def _saveOrAddPage(fig, target):
    """Save ``fig`` to ``target``: either to a path on disk, or as the next
    page of an open ``PdfPages``. Closes ``fig`` after writing."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if isinstance(target, PdfPages):
        target.savefig(fig, dpi=150)
    else:
        fig.savefig(target, dpi=150)
        print(f"  Saved {target}", flush=True)
    plt.close(fig)


def _applyBridge(correction, ramp):
    """Bridge for the half-done PIPE2D-1843 transpose.

    Upstream's ``apply.py`` / ``Ramp.reads`` / ``LinearizedRamp.cumulativeLinear``
    are still documented and implemented as ``(H, W, N)``, but our examples
    (and the new upstream ``fit.py``) use ``(N, H, W)``. Transpose around the
    apply call so downstream fitLinearity code stays in ``(N, H, W)``. Remove
    this wrapper once the transpose is finished upstream.
    """
    from types import SimpleNamespace

    readsHWN = np.moveaxis(ramp.reads, 0, -1).astype(np.float32, copy=True)
    res = nirLinearity.apply(correction, Ramp(reads=readsHWN, validMask=ramp.validMask))
    return SimpleNamespace(
        cumulativeLinear=np.moveaxis(res.cumulativeLinear, -1, 0),
        badPixelMask=res.badPixelMask,
    )


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
    saturationKnee: float | None = None,
    badLinearityMedianMultiplier: float | None = None,
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
        "C3", 0.2, 1.0, hotSel, nHot,
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

    # --- Row 4 right: rejected pixels grouped by named failure flag. ---
    # One color per flag bit. BORDER_PIX is excluded (per established
    # convention); MASKED_BY_INPUT appears if the caller supplied a mask.
    failureFlags = [
        ("HIGH_FIT_RESIDUAL", HIGH_FIT_RESIDUAL, "C3"),    # red
        ("INSUFFICIENT_POINTS", INSUFFICIENT_POINTS, "C1"),  # orange
        ("FIT_FAILED", FIT_FAILED, "C4"),                  # purple
        ("NON_MONOTONIC", NON_MONOTONIC, "magenta"),
        ("MASKED_BY_INPUT", MASKED_BY_INPUT, "C7"),        # gray
    ]
    failureCategoryPlot: list[tuple[str, str, float, float, np.ndarray, np.ndarray, int]] = []
    for fname, fbit, fcolor in failureFlags:
        fmask = ((badPixelMask & fbit) != 0) & notBorder
        fidx = np.flatnonzero(fmask.ravel())
        ftotal = len(fidx)
        if ftotal == 0:
            continue
        if ftotal > nPlot:
            fidx = rngRej.choice(fidx, size=nPlot, replace=False)
        failureCategoryPlot.append(
            (fname, fcolor, 0.15, 0.8, fidx // W, fidx % W, ftotal)
        )

    totalFailurePlotted = sum(len(c[4]) for c in failureCategoryPlot)
    totalFailureFound = sum(c[6] for c in failureCategoryPlot)
    if totalFailurePlotted == 0 and medianTrace is None:
        axRejBar.text(0.5, 0.5, "No rejected pixels", transform=axRejBar.transAxes,
                      ha="center", va="center", fontsize=14)
    else:
        for _flabel, fcolor, falpha, flw, frrows, frcols, _ftotal in failureCategoryPlot:
            if len(frrows) == 0:
                continue
            sub = rawCum[:, frrows, frcols]
            axRejBar.plot(reads, sub, color=fcolor, alpha=falpha, linewidth=flw)
    if medianTrace is not None:
        axRejBar.plot(reads, medianTrace, "k-", linewidth=1.5)

    legendHandles2 = []
    for flabel, fcolor, _falpha, _flw, frrows, _frcols, ftotal in failureCategoryPlot:
        legendHandles2.append(Line2D(
            [0], [0], color=fcolor, linewidth=2.0,
            label=f"{flabel}: plotted {len(frrows):,} / found {ftotal:,}",
        ))
    if medianTrace is not None:
        legendHandles2.append(Line2D(
            [0], [0], color="k", linewidth=1.5, label="median good pixel",
        ))
    axRejBar.legend(handles=legendHandles2, loc="upper left", framealpha=0.9, fontsize=8)
    axRejBar.set_xlabel("Read number")
    axRejBar.set_ylabel("Cumulative DN")
    axRejBar.set_title(
        f"Rejected pixels by named failure flag — "
        f"plotted {totalFailurePlotted:,} of {totalFailureFound:,}"
    )
    axRejBar.grid(True, alpha=0.3)

    # --- Row 3: extreme good pixels — uncorrected / linearized accumulated flux.
    # Each pixel is rendered in its category color (faint=C0, bright=C1, hot=C2)
    # for the in-range portion of the trace, with red overlaying any read where
    # the trace exits ``[fitMin, fitMax]``. Same y-scale as row 1.
    totalGoodPlotted = sum(len(c[4]) for c in goodCategoryPlot)
    nGoodTotal = int(good.sum())
    for label, color, alpha, lw, rrows, rcols, _total in goodCategoryPlot:
        for k in range(len(rrows)):
            r, c = rrows[k], rcols[k]
            mTrace = rawCum[:, r, c]
            tTrace = linCum[:, r, c]
            fMin, fMax = fitMin[r, c], fitMax[r, c]
            inRange = (mTrace >= fMin) & (mTrace <= fMax)
            for seg, segColor in _segments(reads, mTrace, inRange, color, "red"):
                axGoodPopRaw.plot(seg[0], seg[1], color=segColor, alpha=alpha, linewidth=lw)
            for seg, segColor in _segments(reads, tTrace, inRange, color, "red"):
                axGoodPopLin.plot(seg[0], seg[1], color=segColor, alpha=alpha, linewidth=lw)

    # Median good-pixel reference (raw side already has medianTrace; lin side
    # gets its own median over the same sample indices).
    if medianTrace is not None:
        axGoodPopRaw.plot(reads, medianTrace, "k-", linewidth=1.5)
        medianTraceLin = np.median(linCum[:, gRows, gCols], axis=1)
        axGoodPopLin.plot(reads, medianTraceLin, "k-", linewidth=1.5)

    for ax in (axGoodPopRaw, axGoodPopLin):
        ax.plot(reads, idealCum, "k--", linewidth=1, alpha=0.5)
        ax.set_ylim(0, rate * N * 1.3)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Cumulative DN")
        ax.set_xlabel("Read number")

    # Legend: per-category proxies + median + red-overlay + ideal line.
    legendHandles3 = []
    for label, color, _alpha, _lw, rrows, _rcols, total in goodCategoryPlot:
        legendHandles3.append(Line2D(
            [0], [0], color=color, linewidth=2.0,
            label=f"{label}: plotted {len(rrows):,} / found {total:,}",
        ))
    if medianTrace is not None:
        legendHandles3.append(Line2D(
            [0], [0], color="k", linewidth=1.5, label="median good pixel",
        ))
    legendHandles3.append(Line2D(
        [0], [0], color="red", linewidth=2.0, label="out of fit range",
    ))
    legendHandles3.append(Line2D(
        [0], [0], color="k", linestyle="--", linewidth=1,
        label=f"ideal (rate={rate:.1f} DN/read)",
    ))
    axGoodPopRaw.legend(handles=legendHandles3, loc="upper left", framealpha=0.9, fontsize=8)
    axGoodPopLin.legend(handles=legendHandles3, loc="upper left", framealpha=0.9, fontsize=8)
    axGoodPopRaw.set_title(
        f"Selection of extreme good pixels — uncorrected accumulated flux "
        f"(plotted {totalGoodPlotted:,})"
    )
    axGoodPopLin.set_title(
        f"Selection of extreme good pixels — linearized accumulated flux "
        f"(plotted {totalGoodPlotted:,})"
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
    if saturationKnee is not None:
        titleParts.append(f"saturationKnee={saturationKnee}")
    else:
        titleParts.append("saturationKnee=off")
    if badLinearityMedianMultiplier is not None:
        titleParts.append(f"badLinearityMedianMultiplier={badLinearityMedianMultiplier}")
    else:
        titleParts.append("badLinearityMedianMultiplier=off")
    if (
        deviationLimit is None
        and saturationLevel is None
        and saturationKnee is None
    ):
        titleParts.append("no clipping")
    fig.suptitle(", ".join(titleParts), fontsize=11)
    _saveOrAddPage(fig, outPath)


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
    autoLoG, autoHiG = np.percentile(goodMax, [0.05, 99.95]) if len(goodMax) else (0.0, 1.0)
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
        autoHiR = float(np.percentile(rejMax, 99.95))
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

    _saveOrAddPage(fig, outPath)


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

    _saveOrAddPage(fig, outPath)


def _loadInputRamp(inputDir: Path, noPhotodiode: bool) -> tuple[Ramp, Path]:
    """Load the single ``.npz`` ramp in ``inputDir``, photodiode-corrected."""
    npzFiles = sorted(inputDir.glob("*.npz"))
    if not npzFiles:
        raise FileNotFoundError(f"No .npz files found in {inputDir}")
    dataPath = npzFiles[0]
    t0 = time.perf_counter()
    print(f"Loading {dataPath} ...", flush=True)
    correctedRamp, photodiode = loadCorrectedRamp(dataPath, noPhotodiode=noPhotodiode)
    if photodiode.shape[0] and not noPhotodiode:
        scale = photodiode[0] / photodiode
        print(
            f"  photodiode scale range: min={scale.min():.6f} max={scale.max():.6f}",
            flush=True,
        )
    _t("load + photodiode correction", t0)
    return correctedRamp, dataPath


def runFit(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None:
    """Fit the ramp in ``inputDir`` and write the correction FITS to ``outDir``."""
    det = outDir.parent.name
    fitsPath = outDir / f"{det}_linearity.fits"

    t0 = time.perf_counter()
    correctedRamp, dataPath = _loadInputRamp(inputDir, config.noPhotodiode)
    H, W = correctedRamp.reads.shape[1:]

    # Sample pixel indices from all interior pixels BEFORE fitting, so
    # the same pixels are reported regardless of clipping parameters.
    interior = np.ones((H, W), dtype=bool)
    interior[:4, :] = False
    interior[-4:, :] = False
    interior[:, :4] = False
    interior[:, -4:] = False
    rng = np.random.default_rng(config.seed)
    interiorIdx = np.flatnonzero(interior.ravel())
    sampleIdx = rng.choice(interiorIdx, size=min(10000, len(interiorIdx)), replace=False)
    rows = sampleIdx // W
    cols = sampleIdx % W

    from lsst.obs.pfs.h4Linearity.models import PolynomialModel
    model = PolynomialModel(order=config.order)
    print(f"Fitting (order={config.order}) ...", flush=True)
    correction = nirLinearity.fit(
        [correctedRamp],
        model=model,
        deviationLimit=config.deviationLimit,
        deviationStart=config.deviationStart,
        saturationLevel=config.saturationLevel,
        lowFluxFraction=config.lowFluxFraction,
        saturationKnee=config.saturationKnee,
        badLinearityMedianMultiplier=config.badLinearityMultiplier,
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
    result = _applyBridge(loaded, correctedRamp)
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

    from lsst.obs.pfs.h4Linearity.types import ABOVE_VALID_RANGE, BELOW_VALID_RANGE
    belowCount = int((result.badPixelMask & BELOW_VALID_RANGE > 0).sum())
    aboveCount = int((result.badPixelMask & ABOVE_VALID_RANGE > 0).sum())
    nPix = result.badPixelMask.size
    print(f"  below-range pixel fraction: {belowCount / nPix:.4f}", flush=True)
    print(f"  above-range pixel fraction: {aboveCount / nPix:.4f}", flush=True)
    _t("done (fit)", t0)


def runPlot(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None:
    """Plot diagnostics for the correction already written to ``outDir``."""
    det = outDir.parent.name
    fitsPath = outDir / f"{det}_linearity.fits"

    t0 = time.perf_counter()
    if not fitsPath.exists():
        raise FileNotFoundError(
            f"No FITS file at {fitsPath}; run with --fit first"
        )

    correctedRamp, _ = _loadInputRamp(inputDir, config.noPhotodiode)
    H, W = correctedRamp.reads.shape[1:]

    interior = np.ones((H, W), dtype=bool)
    interior[:4, :] = False
    interior[-4:, :] = False
    interior[:, :4] = False
    interior[:, -4:] = False
    rng = np.random.default_rng(config.seed)
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
    saturationKneeFromFits = summary.get("saturationKnee")
    badLinearityMultiplierFromFits = summary.get("badLinearityMedianMultiplier")
    # Detector ID for filenames: prefer the FITS header, fall back to the
    # output directory's detector name. Keeps plot files self-identifying
    # when copied around.
    det = str(summary.get("detector", det))

    print("Applying correction to input ramp ...", flush=True)
    result = _applyBridge(loaded, correctedRamp)
    t1 = _t("nirLinearity.apply", t1)

    rawCum = correctedRamp.reads.astype(np.float32)

    print("Generating diagnostic plots ...", flush=True)
    rate_ = float(np.median(correctedRamp.reads[2] - correctedRamp.reads[1]))
    detector_ = str(summary.get("detector", det))
    visit_ = str(summary.get("visit", ""))

    def _plotAll(target, fitRangeTarget, spatialTarget):
        _plotDiagnostic(
            rawCum=rawCum,
            linCum=result.cumulativeLinear,
            fitMin=loaded.fitMin,
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            rows=rows,
            cols=cols,
            rate=rate_,
            outPath=target,
            nPlot=config.nplot,
            order=order,
            deviationLimit=deviationLimit,
            nRefReads=nRefReads,
            saturationLevel=saturationLevel,
            saturationKnee=saturationKneeFromFits,
            badLinearityMedianMultiplier=badLinearityMultiplierFromFits,
            seed=config.seed,
            detector=detector_,
            visit=visit_,
        )
        _plotFitRange(
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            rawCum=rawCum,
            outPath=fitRangeTarget,
            loOverride=config.fitrangeMin,
            hiOverride=config.fitrangeMax,
        )
        _plotFitRangeSpatial(
            fitMax=loaded.fitMax,
            badPixelMask=loaded.badPixelMask,
            outPath=spatialTarget,
            loOverride=config.fitrangeMin,
            hiOverride=config.fitrangeMax,
            detector=detector_,
            visit=visit_,
        )

    if config.plotFormat == "pdf":
        # Multi-page PDF — all three diagnostics in one file per detector.
        from matplotlib.backends.backend_pdf import PdfPages
        combinedPath = outDir / f"diagnostic_{det}.pdf"
        with PdfPages(combinedPath) as pdf:
            _plotAll(pdf, pdf, pdf)
        print(f"  Saved {combinedPath}", flush=True)
    else:
        _plotAll(
            outDir / f"diagnostic_{det}.{config.plotFormat}",
            outDir / f"diagnostic_fit_range_{det}.{config.plotFormat}",
            outDir / f"diagnostic_fit_range_spatial_{det}.{config.plotFormat}",
        )
    _t("done (plot)", t0)
