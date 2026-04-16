"""Tests for PolynomialModel: evaluate and isMonotonic."""

from __future__ import annotations

import numpy as np
import pytest

from relin.models.polynomial import PolynomialModel


def test_defaultConstructor():
    m = PolynomialModel()
    assert m.order == 4
    assert m.forceThroughOrigin is False
    assert m.modelName == "POLYNOMIAL"


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
