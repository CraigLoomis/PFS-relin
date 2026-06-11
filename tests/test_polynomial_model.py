"""Tests for PolynomialModel: evaluate and isMonotonic."""

from __future__ import annotations

import numpy as np
import pytest

from lsst.obs.pfs.h4Linearity.models.polynomial import PolynomialModel


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


def test_evaluateChebyshevIdentity():
    """t = T_1(x) = x. With coeffs c0=0, c1=1, rest=0, evaluate(x) = x."""
    model = PolynomialModel(order=4)
    H, W, N = 4, 5, 6
    coeffs = np.zeros((5, H, W), dtype=np.float32)
    coeffs[1] = 1.0  # c1 = 1 → T_1(x) = x
    x = np.linspace(-1, 1, N * H * W, dtype=np.float32).reshape(N, H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, x, rtol=1e-6)


def test_evaluateChebyshevConstant():
    """t = 3*T_0(x) = 3. Constant everywhere."""
    model = PolynomialModel(order=2)
    H, W, N = 2, 3, 5
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[0] = 3.0
    x = np.linspace(-1, 1, N * H * W, dtype=np.float32).reshape(N, H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, 3.0, atol=1e-6)


def test_evaluateChebyshevKnownSeries():
    """t = 2*T_0(x) + 3*T_1(x) + 0.5*T_2(x).
    T_0=1, T_1=x, T_2=2x^2-1. So t = 2 + 3x + 0.5(2x^2-1) = 1.5 + 3x + x^2."""
    model = PolynomialModel(order=2)
    H, W, N = 2, 3, 10
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[0] = 2.0
    coeffs[1] = 3.0
    coeffs[2] = 0.5
    x = np.tile(np.linspace(-1, 1, N, dtype=np.float32)[:, None, None], (1, H, W))
    t = model.evaluate(coeffs, x)
    expected = 1.5 + 3.0 * x + x ** 2
    np.testing.assert_allclose(t, expected, rtol=1e-5)


def test_evaluateChebyshevSingleFrameShape():
    """evaluate must accept (H, W) input."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    x = np.linspace(-1, 1, H * W, dtype=np.float32).reshape(H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, x, rtol=1e-6)


def test_isMonotonicChebyshevLinear():
    """t = T_1(x) = x is monotonically increasing on [-1, 1]."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    mMin = np.full((H, W), -1.0, dtype=np.float32)
    mMax = np.full((H, W), 1.0, dtype=np.float32)
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert result.shape == (H, W)
    assert result.all()


def test_isMonotonicChebyshevDetectsNonMonotonic():
    """t = T_2(x) = 2x^2 - 1 has derivative 4x, which is negative for x < 0.
    On [-1, 1] this is non-monotonic."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[2] = 1.0  # T_2(x) = 2x^2 - 1
    mMin = np.full((H, W), -1.0, dtype=np.float32)
    mMax = np.full((H, W), 1.0, dtype=np.float32)
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert not result.any()


# ---------------------------------------------------------------------------
# fitBlock tests
# ---------------------------------------------------------------------------

from lsst.obs.pfs.h4Linearity.types import (
    MASKED_BY_INPUT,  # noqa: F401  (used transitively; referenced below)
    INSUFFICIENT_POINTS,
    FIT_FAILED,
)


def test_fitBlockChebyshevRecoversLinear():
    """Fit t = m (identity) and verify evaluate(coeffs, x) reproduces t."""
    N, H, W = 20, 2, 3
    mVals = np.linspace(100.0, 500.0, N, dtype=np.float32)
    m = np.tile(mVals[:, None, None], (1, H, W))
    t = mVals.copy()
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (3, H, W)
    fitMin = result.fitMin
    fitMax = result.fitMax
    x = 2.0 * (m - fitMin[None]) / (fitMax - fitMin)[None] - 1.0
    tPred = model.evaluate(result.coefficients, x.astype(np.float32))
    tExpected = np.tile(t[:, None, None], (1, H, W))
    np.testing.assert_allclose(tPred, tExpected, rtol=1e-4, atol=1e-2)
    assert (result.badPixelMask == 0).all()


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
    from lsst.obs.pfs.h4Linearity.types import NON_MONOTONIC
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


