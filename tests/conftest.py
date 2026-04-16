"""Shared pytest fixtures for relin tests."""

from __future__ import annotations

import numpy as np
import pytest

from relin.types import Ramp


def _buildSyntheticDeltas(
    H: int,
    W: int,
    N: int,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    rate: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Construct ``(deltas, trueCoeffs)`` such that for each pixel::

        t[n] = rate * (n + 1)                 # the linearization target
        t[n] = c0 + c1*m[n] + c2*m[n]^2
               + c3*m[n]^3 + c4*m[n]^4        # the per-pixel nonlinearity
        m[n] = cumsum(deltas[:, h, w])[n]

    Strategy: choose ``m[n]`` by inverting the polynomial for the given ``t[n]``
    values (numerically via ``np.roots`` — picking the physically sensible real
    root in range), then differentiate to get ``deltas[n]``.

    For a well-posed test, construct ``m[n]`` directly by solving
    ``t[n] = Σ c_i m[n]^i`` with a bisection on a chosen invertible branch.
    """
    # Target at each read, same for all pixels
    t = rate * np.arange(1, N + 1, dtype=np.float64)  # (N,)

    # Solve t[n] = c0 + c1 m + c2 m^2 + c3 m^3 + c4 m^4 for each pixel & read
    # via Newton's method, starting from m ≈ (t - c0) / c1.
    m = np.empty((N, H, W), dtype=np.float64)
    for n in range(N):
        mGuess = (t[n] - c0) / np.where(c1 != 0, c1, 1.0)
        for _ in range(50):
            pVal = c0 + c1 * mGuess + c2 * mGuess**2 + c3 * mGuess**3 + c4 * mGuess**4
            pPrime = c1 + 2 * c2 * mGuess + 3 * c3 * mGuess**2 + 4 * c4 * mGuess**3
            step = (pVal - t[n]) / np.where(pPrime != 0, pPrime, 1.0)
            mGuess = mGuess - step
            if np.max(np.abs(step)) < 1e-10:
                break
        m[n] = mGuess

    # deltas[0] = m[0]; deltas[n>0] = m[n] - m[n-1]
    deltas = np.empty_like(m)
    deltas[0] = m[0]
    deltas[1:] = np.diff(m, axis=0)

    trueCoeffs = {
        "c0": c0.astype(np.float32),
        "c1": c1.astype(np.float32),
        "c2": c2.astype(np.float32),
        "c3": c3.astype(np.float32),
        "c4": c4.astype(np.float32),
        "targetRate": float(rate),
        "target": t.astype(np.float32),
        "mTrue": m.astype(np.float32),
    }
    return deltas.astype(np.float32), trueCoeffs


@pytest.fixture
def smallSyntheticRamp():
    """A 29-read 4x5 ramp with spatially-varying polynomial coefficients.

    The target rate ``R`` is chosen so that ``median(deltas[0]) == R`` by
    construction (all pixels' first deltas equal the rate).
    """
    rng = np.random.default_rng(seed=42)
    H, W, N = 4, 5, 29
    rate = 1000.0  # DN per read

    # Per-pixel polynomial coefficients: mostly-linear with small higher-order terms.
    c0 = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float64)
    c1 = np.full((H, W), 1.0, dtype=np.float64) + rng.normal(0.0, 1e-3, size=(H, W))
    c2 = rng.normal(0.0, 1e-7, size=(H, W)).astype(np.float64)
    c3 = rng.normal(0.0, 1e-11, size=(H, W)).astype(np.float64)
    c4 = rng.normal(0.0, 1e-15, size=(H, W)).astype(np.float64)

    deltas, trueCoeffs = _buildSyntheticDeltas(H, W, N, c0, c1, c2, c3, c4, rate)
    return Ramp(deltas=deltas), trueCoeffs


@pytest.fixture
def tinyLinearRamp():
    """A 6-read 2x3 ramp where every pixel is perfectly linear: t = m * pixelScale.

    Each pixel's per-read delta is a constant, chosen so that
    ``cumsum(deltas)[n] == pixelScale * target[n] / targetRate``. Useful for the
    simplest-possible fit test.
    """
    H, W, N = 2, 3, 6
    rate = 500.0
    # Pixel-scale factor — varies per pixel so the PRNU is non-trivial
    pixelScale = np.array(
        [[1.0, 1.1, 0.9], [0.95, 1.05, 1.0]], dtype=np.float32
    )
    deltas = np.broadcast_to(
        (rate * pixelScale)[None, :, :], (N, H, W)
    ).astype(np.float32).copy()
    target = rate * np.arange(1, N + 1, dtype=np.float32)
    return Ramp(deltas=deltas), {
        "pixelScale": pixelScale,
        "targetRate": rate,
        "target": target,
    }
