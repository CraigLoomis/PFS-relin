"""Tests for the top-level fit() function."""

from __future__ import annotations

import numpy as np

from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import MASKED_BY_INPUT, Ramp


def test_fitSingleRampRecoversTarget(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    assert correction.coefficients.shape == (5, 4, 5)
    assert (correction.badPixelMask == 0).all()
    # Evaluate at the fit points: map m → x, then evaluate.
    m = ramp.reads.astype(np.float32)
    denom = correction.fitMax - correction.fitMin
    denom = np.where(denom > 0, denom, 1.0)
    x = 2.0 * (m - correction.fitMin[None]) / denom[None] - 1.0
    tPred = correction.model.evaluate(correction.coefficients, x)
    fitRate = float(np.median(ramp.reads[0]))
    N = ramp.reads.shape[0]
    expected = fitRate * np.arange(1, N + 1, dtype=np.float32)
    expectedBroad = np.broadcast_to(expected[:, None, None], tPred.shape)
    np.testing.assert_allclose(tPred, expectedBroad, rtol=1e-3, atol=1.0)
    # Summary should carry percentiles.
    assert "residualRmsP50" in correction.diagnostics.summary
    assert "residualRmsP95" in correction.diagnostics.summary
    assert "residualRmsP99" in correction.diagnostics.summary


def test_fitTilingIsDeterministic(smallSyntheticRamp):
    """Fitting with different block sizes must yield identical coefficients
    (within float32 precision)."""
    ramp, _ = smallSyntheticRamp
    c1 = fit([ramp], blockSize=(4, 5)).coefficients
    c2 = fit([ramp], blockSize=(2, 3)).coefficients
    np.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_fitPropagatesInputMask(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    mask = np.zeros(ramp.reads.shape[1:], dtype=np.uint8)
    mask[0, 0] = 1  # Mark pixel (0, 0) as invalid
    maskedRamp = Ramp(reads=ramp.reads, validMask=mask)
    correction = fit([maskedRamp], model=PolynomialModel(order=1))
    assert correction.badPixelMask[0, 0] & MASKED_BY_INPUT
    assert correction.badPixelMask[0, 1] == 0


def test_fitMultipleRampsConcatenates():
    """Two ramps of different lengths combine per-pixel."""
    H, W = 3, 4
    # Pixel-linear: t = m for every pixel.
    # Ramp 1: 8 reads, rate 100.
    # Ramp 2: 12 reads, rate 200.
    rate1 = 100.0
    rate2 = 200.0
    reads1 = np.full((1, H, W), rate1, dtype=np.float32) * np.arange(1, 9, dtype=np.float32)[:, None, None]
    reads2 = np.full((1, H, W), rate2, dtype=np.float32) * np.arange(1, 13, dtype=np.float32)[:, None, None]
    correction = fit(
        [Ramp(reads=reads1), Ramp(reads=reads2)],
        model=PolynomialModel(order=2),
    )
    # Verify via evaluation: for ramp1, evaluate at its m values → should match targets.
    m1 = reads1
    denom = correction.fitMax - correction.fitMin
    denom = np.where(denom > 0, denom, 1.0)
    x1 = 2.0 * (m1 - correction.fitMin[None]) / denom[None] - 1.0
    tPred = correction.model.evaluate(correction.coefficients, x1)
    # Target for ramp1: rate1 * (n+1)
    expected1 = rate1 * np.arange(1, 9, dtype=np.float32)
    expected1Broad = np.broadcast_to(expected1[:, None, None], tPred.shape)
    np.testing.assert_allclose(tPred, expected1Broad, rtol=1e-4, atol=1e-2)
    # nPointsUsed should be 8 + 12 = 20 everywhere.
    assert (correction.diagnostics.nPointsUsed == 20).all()


def test_fitEmptyRampListRaises():
    import pytest
    with pytest.raises(ValueError):
        fit([])
