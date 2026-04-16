"""Polynomial nonlinearity model: t = c0 + c1*m + ... + cp*m^p (per pixel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relin.models.base import BlockFitResult  # noqa: F401  (used in a later task)


@dataclass(frozen=True)
class PolynomialModel:
    """Pluggable polynomial-fit model. Default 4th order, general form."""

    order: int = 4
    forceThroughOrigin: bool = False

    modelName: str = "POLYNOMIAL"

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise ValueError(f"order must be an int, got {type(self.order).__name__}")
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")

    def evaluate(self, coefficients: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Evaluate the per-pixel polynomial via Horner's method.

        Parameters
        ----------
        coefficients
            Shape ``(order+1, H, W)``, float32. ``coefficients[i]`` is the
            coefficient of m^i (constant term first).
        m
            Shape ``(..., H, W)``, float32. Leading dimensions (e.g. reads) are broadcast.

        Returns
        -------
        t
            Same shape as ``m``.
        """
        coefficients = np.asarray(coefficients)
        m = np.asarray(m)
        order = coefficients.shape[0] - 1
        # Horner: t = ((c_p * m + c_{p-1}) * m + ... + c_1) * m + c_0
        t = np.full_like(m, coefficients[order], dtype=m.dtype)
        for i in range(order - 1, -1, -1):
            t = t * m + coefficients[i]
        return t

    def isMonotonic(
        self,
        coefficients: np.ndarray,
        mMin: np.ndarray,
        mMax: np.ndarray,
        nSamples: int = 32,
    ) -> np.ndarray:
        """Return an ``(H, W)`` boolean map: ``True`` if the fit is monotonically
        increasing on ``[mMin, mMax]`` per pixel.

        Samples the polynomial derivative at ``nSamples`` evenly-spaced points
        per pixel and reports whether all sampled derivatives are non-negative.
        Pixels with ``mMin == mMax`` are considered trivially monotonic.
        """
        coefficients = np.asarray(coefficients, dtype=np.float64)
        order = coefficients.shape[0] - 1
        if order < 1:
            return np.ones(coefficients.shape[1:], dtype=bool)

        # Derivative coefficients: d_i = (i+1) * c_{i+1}, i = 0..order-1
        derivCoefs = (
            coefficients[1:] * np.arange(1, order + 1, dtype=np.float64)[:, None, None]
        )  # shape (order, H, W)

        H, W = mMin.shape
        # Evenly-spaced sample points per pixel: shape (nSamples, H, W)
        fractions = np.linspace(0.0, 1.0, nSamples, dtype=np.float64)
        samplePoints = (
            mMin[None] + (mMax - mMin)[None] * fractions[:, None, None]
        )

        # Evaluate derivative polynomial at samplePoints via Horner.
        d = np.full_like(samplePoints, derivCoefs[order - 1], dtype=np.float64)
        for i in range(order - 2, -1, -1):
            d = d * samplePoints + derivCoefs[i]

        allNonNegative = (d >= 0).all(axis=0)  # (H, W)
        # Treat degenerate [mMin == mMax] pixels as monotonic.
        degenerate = mMax <= mMin
        return allNonNegative | degenerate
