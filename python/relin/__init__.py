"""relin — per-pixel nonlinearity correction for IR detector ramps."""

from __future__ import annotations

from relin.apply import apply, applyFrame
from relin.fit import fit
from relin.io import loadFits, saveFits
from relin.models import Model, PolynomialModel
from relin.types import (
    FIT_FAILED,
    INSUFFICIENT_POINTS,
    MASKED_BY_INPUT,
    NON_MONOTONIC,
    Diagnostics,
    LinearityCorrection,
    LinearizedRamp,
    Ramp,
)

__all__ = [
    "Ramp",
    "LinearizedRamp",
    "Diagnostics",
    "LinearityCorrection",
    "Model",
    "PolynomialModel",
    "fit",
    "apply",
    "applyFrame",
    "saveFits",
    "loadFits",
    "MASKED_BY_INPUT",
    "INSUFFICIENT_POINTS",
    "FIT_FAILED",
    "NON_MONOTONIC",
]
