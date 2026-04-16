"""Tests for apply() and applyFrame()."""

from __future__ import annotations

import numpy as np
import pytest

from relin.apply import apply, applyFrame
from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import Ramp


def test_applyOnFittedRampYieldsTarget(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    result = apply(correction, ramp)
    # cumulativeLinear should closely match the true target at every read.
    expected = np.broadcast_to(
        truth["target"][:, None, None], ramp.deltas.shape
    )
    np.testing.assert_allclose(
        result.cumulativeLinear, expected, rtol=1e-3, atol=1e-1
    )
    # No pixel should be out-of-range on the same data it was fit on.
    assert not result.outOfRangeMask.any()


def test_applyFlagsOutOfRangeForExtrapolation():
    rng = np.random.default_rng(0)
    H, W = 2, 3
    # Linear ramp: t = m.
    deltas = np.ones((5, H, W), dtype=np.float32)
    correction = fit(
        [Ramp(deltas=deltas)], model=PolynomialModel(order=1)
    )
    # Build a ramp whose cumulative values exceed fit_max.
    extrapRamp = Ramp(
        deltas=np.ones((20, H, W), dtype=np.float32)
    )
    result = apply(correction, extrapRamp)
    # Reads beyond index 4 are extrapolated (m > fit_max of original fit).
    assert result.outOfRangeMask[10:].all()


def test_applyLeavesBadPixelsUntouched(tinyLinearRamp):
    ramp, truth = tinyLinearRamp
    mask = np.zeros(ramp.deltas.shape[1:], dtype=np.uint8)
    mask[0, 0] = 1
    correction = fit(
        [Ramp(deltas=ramp.deltas, validMask=mask)],
        model=PolynomialModel(order=1),
    )
    result = apply(correction, Ramp(deltas=ramp.deltas))
    # Pixel (0, 0) is bad: output equals cumulative of input (unchanged).
    inputCum = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    np.testing.assert_allclose(
        result.cumulativeLinear[:, 0, 0], inputCum[:, 0, 0]
    )
    # Pixel (0, 1) is good: output is linearized (differs from raw cumulative).
    assert not np.allclose(
        result.cumulativeLinear[:, 0, 1], inputCum[:, 0, 1]
    )


def test_applyFrameMatchesApplyOnSingleRead(smallSyntheticRamp):
    ramp, _ = smallSyntheticRamp
    correction = fit([ramp])
    fullResult = apply(correction, ramp)

    mSingle = np.cumsum(ramp.deltas, axis=0)[-1]  # last read's cumulative
    linFrame, oorFrame = applyFrame(correction, mSingle)

    np.testing.assert_allclose(
        linFrame, fullResult.cumulativeLinear[-1], rtol=1e-6
    )
    np.testing.assert_array_equal(oorFrame, fullResult.outOfRangeMask[-1])


def test_applyRejectsShapeMismatch(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    correction = fit([ramp], model=PolynomialModel(order=1))
    wrongRamp = Ramp(deltas=np.zeros((3, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        apply(correction, wrongRamp)


def test_applyEmptyRampRaises(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    correction = fit([ramp], model=PolynomialModel(order=1))
    empty = Ramp(
        deltas=np.zeros((0, *ramp.deltas.shape[1:]), dtype=np.float32)
    )
    with pytest.raises(ValueError):
        apply(correction, empty)
