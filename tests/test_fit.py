"""Tests for the top-level fit() function."""

from __future__ import annotations

import numpy as np

from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import MASKED_BY_INPUT, Ramp


def test_fitSingleRampRecoversCoefficients(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    assert correction.coefficients.shape == (5, 4, 5)
    # Leading coefficients should roughly match truth.
    np.testing.assert_allclose(
        correction.coefficients[1], truth["c1"], rtol=1e-3, atol=1e-3
    )
    assert (correction.badPixelMask == 0).all()
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
    mask = np.zeros(ramp.deltas.shape[1:], dtype=np.uint8)
    mask[0, 0] = 1  # Mark pixel (0, 0) as invalid
    maskedRamp = Ramp(deltas=ramp.deltas, validMask=mask)
    correction = fit([maskedRamp], model=PolynomialModel(order=1))
    assert correction.badPixelMask[0, 0] & MASKED_BY_INPUT
    assert correction.badPixelMask[0, 1] == 0


def test_fitMultipleRampsConcatenates():
    """Two ramps of different lengths combine per-pixel."""
    rng = np.random.default_rng(0)
    H, W = 3, 4
    # Pixel-linear: t = m for every pixel.
    # Ramp 1: 8 reads, rate 100.
    # Ramp 2: 12 reads, rate 200.
    rate1 = 100.0
    rate2 = 200.0
    deltas1 = np.full((8, H, W), rate1, dtype=np.float32)
    deltas2 = np.full((12, H, W), rate2, dtype=np.float32)
    correction = fit(
        [Ramp(deltas=deltas1), Ramp(deltas=deltas2)],
        model=PolynomialModel(order=2),
    )
    # Expected: t = m identically, so c0 ≈ 0, c1 ≈ 1, c2 ≈ 0.
    np.testing.assert_allclose(correction.coefficients[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(correction.coefficients[1], 1.0, rtol=1e-4)
    np.testing.assert_allclose(correction.coefficients[2], 0.0, atol=1e-6)
    # nPointsUsed should be 8 + 12 = 20 everywhere.
    assert (correction.diagnostics.nPointsUsed == 20).all()


def test_fitEmptyRampListRaises():
    import pytest
    with pytest.raises(ValueError):
        fit([])
