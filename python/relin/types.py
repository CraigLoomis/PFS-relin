"""Data types and bad-pixel flag constants for relin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Bad-pixel bit-flag constants. See spec section 3.
MASKED_BY_INPUT: int = 0x01
INSUFFICIENT_POINTS: int = 0x02
FIT_FAILED: int = 0x04
NON_MONOTONIC: int = 0x08


@dataclass(frozen=True)
class Ramp:
    """A single ramp's photodiode-corrected deltas plus an optional pixel mask.

    Parameters
    ----------
    deltas
        Shape ``(N, H, W)``, float32. Already photodiode-corrected.
    validMask
        Shape ``(H, W)``; 0 means valid. May be ``None`` for "all pixels valid".
    """

    deltas: np.ndarray
    validMask: np.ndarray | None = None


@dataclass(frozen=True)
class LinearizedRamp:
    """Output of :func:`relin.apply.apply` on a :class:`Ramp`."""

    cumulativeLinear: np.ndarray   # (N, H, W) float32
    outOfRangeMask: np.ndarray     # (N, H, W) bool
    badPixelMask: np.ndarray       # (H, W) uint8


@dataclass(frozen=True)
class Diagnostics:
    """Per-pixel fit diagnostics plus a dataset-wide summary."""

    residualRms: np.ndarray        # (H, W) float32
    maxAbsResidual: np.ndarray     # (H, W) float32
    nPointsUsed: np.ndarray        # (H, W) int32
    monotonic: np.ndarray          # (H, W) bool
    conditionNumber: np.ndarray    # (H, W) float32
    summary: dict[str, Any]


@dataclass(frozen=True)
class LinearityCorrection:
    """A fitted per-pixel nonlinearity correction."""

    model: Any                     # Model protocol; runtime-checked to avoid a types <-> models cycle
    coefficients: np.ndarray       # shape depends on model; polynomial: (order+1, H, W) float32
    fitMin: np.ndarray             # (H, W) float32
    fitMax: np.ndarray             # (H, W) float32
    badPixelMask: np.ndarray       # (H, W) uint8
    diagnostics: Diagnostics
