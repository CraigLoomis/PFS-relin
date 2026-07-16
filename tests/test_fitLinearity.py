"""Tests for the sanity check's configuration tag."""

from __future__ import annotations

import numpy as np
from fitLinearity.fitLinearity import SanityCheckConfig, _foldRateStability, cliTag
from lsst.obs.pfs.h4Linearity.types import RATE_UNSTABLE


def _config(**overrides) -> SanityCheckConfig:
    defaults = dict(
        order=4, deviationLimit=None, deviationStart=0.5, saturationLevel=None,
        lowFluxFraction=0.5, saturationKnee=0.5, badLinearityMultiplier=5.0,
        noPhotodiode=False, seed=0, nplot=1000, plotFormat="png",
        fitrangeMin=None, fitrangeMax=None,
    )
    defaults.update(overrides)
    return SanityCheckConfig(**defaults)


def testDefaultsTagIsOrderOnly():
    assert cliTag(_config()) == "o4"


def testNoPhotodiodeTag():
    assert cliTag(_config(noPhotodiode=True)) == "o4_pdOff"


def testDeviationLimitTag():
    assert cliTag(_config(order=5, deviationLimit=0.45)) == "o5_dev0.45"


def testNonDefaultDeviationStartTag():
    assert cliTag(_config(deviationStart=0.3)) == "o4_ds0.3"


def testSaturationLevelTagIsInteger():
    assert cliTag(_config(saturationLevel=45000.0)) == "o4_sat45000"


def testDisabledGatesTag():
    tag = cliTag(_config(saturationKnee=None, badLinearityMultiplier=None))
    assert tag == "o4_kneeOff_blmOff"


def testEveryFieldTogether():
    tag = cliTag(_config(
        order=5, noPhotodiode=True, deviationLimit=0.45, deviationStart=0.3,
        saturationLevel=45000.0, lowFluxFraction=0.7, saturationKnee=0.6,
        badLinearityMultiplier=3.0,
    ))
    assert tag == "o5_pdOff_dev0.45_ds0.3_sat45000_lff0.7_knee0.6_blm3.0"


def testRateStabilityOffAddsNoTag():
    assert cliTag(_config(rateStability=False)) == "o4"


def testRateStabilityEnabledDefaultTag():
    assert cliTag(_config(rateStability=True)) == "o4_rs0.2"


def testRateStabilityNonDefaultThresholdTag():
    assert cliTag(_config(rateStability=True, rateStabilityThreshold=0.3)) == "o4_rs0.3"


def testRateStabilityNonDefaultFloorTag():
    assert cliTag(_config(rateStability=True, rateStabilityFloor=8.0)) == "o4_rs0.2_rsf8.0"


def testRateStabilityFloorIgnoredWhenDisabled():
    # Floor only appears when the gate is on.
    assert cliTag(_config(rateStability=False, rateStabilityFloor=8.0)) == "o4"


def _cumulativeFromDeltas(deltaRows):
    """Build a (N, H, W) linearized cumulative cube from per-pixel delta lists.

    deltaRows[i] is the length-(N-1) delta sequence for pixel (0, i); read 0 is 0.
    """
    deltas = np.array(deltaRows, dtype=np.float32).T          # (N-1, npix)
    nDeltas, npix = deltas.shape
    cube = np.zeros((nDeltas + 1, 1, npix), dtype=np.float32)
    np.cumsum(deltas, axis=0, out=cube[1:, 0, :])
    return cube


def _openRange(cube):
    """Per-pixel ``(fitMin, fitMax)`` wide enough to keep every read in range."""
    hw = cube.shape[1:]
    return np.full(hw, -1e30, dtype=np.float32), np.full(hw, 1e30, dtype=np.float32)


def testFoldRateStabilityFlagsOnlyTheUnstablePixel():
    # 3 pixels, 7 deltas each. Halves split [0:3] / [3:7].
    #   pixel 0: constant rate 10 -> stable
    #   pixel 1: 10 in first half, 100 in second -> unstable
    #   pixel 2: constant, but pre-masked (excluded from the gate)
    cube = _cumulativeFromDeltas([
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 100, 100, 100, 100],
        [10, 10, 10, 10, 10, 10, 10],
    ])
    fitMin, fitMax = _openRange(cube)
    badPixelMask = np.array([[0, 0, 0x0001]], dtype=np.int32)
    folded, result = _foldRateStability(cube, cube, fitMin, fitMax, badPixelMask,
                                        threshold=0.20, rateFloorADU=5.0)

    assert folded[0, 0] == 0                       # stable good pixel: untouched
    assert folded[0, 1] == RATE_UNSTABLE           # unstable good pixel: flagged
    assert folded[0, 2] == 0x0001                  # pre-masked pixel: preserved, not gated
    assert result is not None and result.nRejected == 1
    assert badPixelMask[0, 1] == 0                 # input mask not mutated in place


def testFoldRateStabilityShortRampSkips():
    # 4 deltas < 2*minDeltasPerSegment (6): gate cannot form two halves.
    cube = _cumulativeFromDeltas([[10, 10, 10, 10]])
    fitMin, fitMax = _openRange(cube)
    badPixelMask = np.array([[0]], dtype=np.int32)
    folded, result = _foldRateStability(cube, cube, fitMin, fitMax, badPixelMask,
                                        threshold=0.20, rateFloorADU=5.0)
    assert result is None
    assert folded[0, 0] == 0                       # nothing flagged


def testFoldRateStabilityClipsAboveFitMax():
    # One pixel: linearized rate is a constant 10 while the raw signal is in
    # range, then crashes to -50 on the last two reads, whose raw signal exceeds
    # fitMax. That crash is the only thing that would trip the gate, and it sits
    # out of range.
    linCube = _cumulativeFromDeltas([[10] * 11 + [-50, -50]])   # N = 14
    rawCube = _cumulativeFromDeltas([[10] * 13])                # raw rises linearly to 130
    fitMin = np.full((1, 1), -1e30, dtype=np.float32)
    badPixelMask = np.array([[0]], dtype=np.int32)

    # Clipped at fitMax=115: the last two reads (raw 120, 130) are excluded, so
    # both halves see only the constant-rate region -> stable -> not flagged.
    fitMaxClip = np.full((1, 1), 115.0, dtype=np.float32)
    folded, result = _foldRateStability(linCube, rawCube, fitMin, fitMaxClip, badPixelMask,
                                        threshold=0.20, rateFloorADU=5.0)
    assert folded[0, 0] == 0
    assert result is not None and result.nRejected == 0

    # Without the clip (fitMax wide open) the same ramp's out-of-range droop
    # enters the second half and the pixel is rejected -- confirming the clip,
    # not the data, is what spares it.
    fitMaxOpen = np.full((1, 1), 1e30, dtype=np.float32)
    foldedOpen, _ = _foldRateStability(linCube, rawCube, fitMin, fitMaxOpen, badPixelMask,
                                       threshold=0.20, rateFloorADU=5.0)
    assert foldedOpen[0, 0] == RATE_UNSTABLE
