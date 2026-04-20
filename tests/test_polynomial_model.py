"""Tests for PolynomialModel: evaluate and isMonotonic."""

from __future__ import annotations

import numpy as np
import pytest

from relin.models.polynomial import PolynomialModel


def test_defaultConstructor():
    m = PolynomialModel()
    assert m.order == 4
    assert m.modelName == "CHEBYSHEV"


def test_customOrder():
    m = PolynomialModel(order=3)
    assert m.order == 3


def test_rejectsNonIntegerOrder():
    with pytest.raises(ValueError):
        PolynomialModel(order=4.5)  # type: ignore[arg-type]


def test_rejectsZeroOrder():
    with pytest.raises(ValueError):
        PolynomialModel(order=0)


def test_evaluateIdentity():
    """t = m (c0=0, c1=1, rest 0) should return m unchanged."""
    model = PolynomialModel(order=4)
    H, W, N = 4, 5, 6
    coeffs = np.zeros((5, H, W), dtype=np.float32)
    coeffs[1] = 1.0  # c1 = 1 everywhere
    m = np.linspace(0, 10, N * H * W, dtype=np.float32).reshape(N, H, W)
    t = model.evaluate(coeffs, m)
    np.testing.assert_allclose(t, m, rtol=1e-6)


def test_evaluateKnownPolynomial():
    """t = 2 + 3m + 0.5 m^2, pixel-constant coefficients."""
    model = PolynomialModel(order=2)
    H, W, N = 2, 3, 5
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[0] = 2.0
    coeffs[1] = 3.0
    coeffs[2] = 0.5
    m = np.tile(np.linspace(0, 4, N, dtype=np.float32)[:, None, None], (1, H, W))
    t = model.evaluate(coeffs, m)
    expected = 2.0 + 3.0 * m + 0.5 * m ** 2
    np.testing.assert_allclose(t, expected, rtol=1e-5)


def test_evaluateSingleFrameShape():
    """evaluate must also accept (H, W) input."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    m = np.arange(H * W, dtype=np.float32).reshape(H, W)
    t = model.evaluate(coeffs, m)
    np.testing.assert_allclose(t, m, rtol=1e-6)


def test_isMonotonicOnLinearCoefficients():
    """t = m is monotonic everywhere."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    mMin = np.zeros((H, W), dtype=np.float32)
    mMax = np.full((H, W), 10.0, dtype=np.float32)
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert result.shape == (H, W)
    assert result.all()


def test_isMonotonicDetectsNonMonotonicQuadratic():
    """t = -m^2 + 2m has derivative -2m + 2, which flips sign at m=1."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 2.0
    coeffs[2] = -1.0
    mMin = np.zeros((H, W), dtype=np.float32)
    mMax = np.full((H, W), 2.0, dtype=np.float32)  # spans the sign flip
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert not result.any()


# ---------------------------------------------------------------------------
# fitBlock tests
# ---------------------------------------------------------------------------

from relin.types import (
    MASKED_BY_INPUT,  # noqa: F401  (used transitively; referenced below)
    INSUFFICIENT_POINTS,
    FIT_FAILED,
)


def test_fitBlockRecoversLinearCoefficients(tinyLinearRamp):
    """With perfectly linear data, recovered coefficients should have c0 ≈ 0
    and c1 exactly capturing the scaling needed to hit the target rate."""
    ramp, truth = tinyLinearRamp
    N, H, W = ramp.deltas.shape
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    t = truth["target"].astype(np.float32)
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (3, H, W)
    # For pixel (h, w): m[n] = rate * pixelScale[h,w] * (n+1), t[n] = rate * (n+1)
    # so t = m / pixelScale[h,w]. Expect c0 ≈ 0, c1 ≈ 1/pixelScale, c2 ≈ 0.
    np.testing.assert_allclose(result.coefficients[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(
        result.coefficients[1], 1.0 / truth["pixelScale"], rtol=1e-4
    )
    np.testing.assert_allclose(result.coefficients[2], 0.0, atol=1e-6)
    assert (result.badPixelMask == 0).all()
    assert (result.nPointsUsed == N).all()


def test_fitBlockRecoversPolynomialCoefficients(smallSyntheticRamp):
    """Fit the known 4th-order synthetic ramp and verify coefficients are recovered."""
    ramp, truth = smallSyntheticRamp
    N, H, W = ramp.deltas.shape
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    t = truth["target"].astype(np.float32)
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=4)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (5, H, W)
    # Coefficients should match within a loose tolerance — synthetic data is
    # constructed exactly but Newton-iteration residuals and float32 arithmetic
    # limit the recovery precision.
    np.testing.assert_allclose(result.coefficients[0], truth["c0"], atol=1.0)
    np.testing.assert_allclose(
        result.coefficients[1], truth["c1"], rtol=1e-3, atol=1e-3
    )
    assert (result.badPixelMask == 0).all()
    # The residuals should be small: each pixel's target-minus-prediction is
    # close to zero.
    assert result.residualRms.max() < 1.0


def test_fitBlockFlagsInsufficientPoints():
    """A pixel with only 3 valid reads cannot fit a 4th-order polynomial
    (needs >= 6 points)."""
    N, H, W = 29, 2, 2
    m = np.tile(np.arange(1, N + 1, dtype=np.float32)[:, None, None], (1, H, W))
    t = np.arange(1, N + 1, dtype=np.float32)
    valid = np.ones((N, H, W), dtype=bool)
    valid[3:, 0, 0] = False  # Pixel (0, 0) has only 3 valid reads

    model = PolynomialModel(order=4)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.badPixelMask[0, 0] & INSUFFICIENT_POINTS
    assert result.badPixelMask[0, 1] == 0
    assert result.badPixelMask[1, 0] == 0
    assert result.badPixelMask[1, 1] == 0
    # Insufficient-points pixel's coefficient row should be zeroed
    np.testing.assert_array_equal(result.coefficients[:, 0, 0], 0.0)


def test_fitBlockFlagsFitFailedWhenIllConditioned():
    """A pixel where m is constant across all reads yields a singular normal
    equations matrix and should be flagged FIT_FAILED."""
    N, H, W = 29, 2, 2
    m = np.ones((N, H, W), dtype=np.float32) * 100.0  # constant m across reads
    t = np.arange(1, N + 1, dtype=np.float32)  # varying t
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=4)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e10)

    # All pixels degenerate — all flagged FIT_FAILED (or INSUFFICIENT_POINTS
    # would not apply since nPoints == N). FIT_FAILED set.
    assert (result.badPixelMask & FIT_FAILED).all()


def test_fitBlockFlagsNonMonotonicFit():
    """A fit that produces a non-monotonic polynomial on its domain must set
    NON_MONOTONIC and must not set INSUFFICIENT_POINTS or FIT_FAILED."""
    from relin.types import NON_MONOTONIC
    # Construct t vs m such that the best-fit quadratic peaks inside the range:
    # target is a down-opening parabola in m. m spans [0, 2], t = -m^2 + 2m
    # (peak at m=1). Use a 2x2 pixel block with identical data everywhere.
    N, H, W = 13, 2, 2
    mVec = np.linspace(0.0, 2.0, N, dtype=np.float32)
    tVec = (-mVec ** 2 + 2.0 * mVec).astype(np.float32)
    m = np.tile(mVec[:, None, None], (1, H, W))
    t = tVec
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    # Every pixel: not skipped, non-monotonic -> NON_MONOTONIC flag set,
    # no INSUFFICIENT_POINTS, no FIT_FAILED.
    assert (result.badPixelMask & NON_MONOTONIC).all()
    assert not (result.badPixelMask & INSUFFICIENT_POINTS).any()
    assert not (result.badPixelMask & FIT_FAILED).any()
    assert not result.monotonic.any()


from astropy.io import fits


def test_polynomialModelFitsRoundTrip():
    H, W = 2, 3
    coeffs = np.arange(5 * H * W, dtype=np.float32).reshape(5, H, W)
    model = PolynomialModel(order=4)

    # Build a minimal correction-like object with just what to/fromFitsHdus read.
    class Stub:
        def __init__(self, c):
            self.coefficients = c
    hdus = model.toFitsHdus(Stub(coeffs))

    # Expect exactly one ImageHDU named "COEFFS".
    assert len(hdus) == 1
    assert hdus[0].name == "COEFFS"
    np.testing.assert_array_equal(hdus[0].data, coeffs)

    # Header must record ORDER.
    assert hdus[0].header["ORDER"] == 4

    # Round-trip through fromFitsHdus.
    loadedModel, loadedCoeffs = PolynomialModel.fromFitsHdus(hdus)
    assert loadedModel.order == 4
    np.testing.assert_array_equal(loadedCoeffs, coeffs)


