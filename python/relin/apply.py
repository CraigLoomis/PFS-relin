"""Apply a fitted LinearityCorrection to a new ramp or single cumulative frame."""

from __future__ import annotations

import numpy as np

from relin.types import LinearityCorrection, LinearizedRamp, Ramp


def apply(correction: LinearityCorrection, ramp: Ramp) -> LinearizedRamp:
    """Linearize a full ramp."""
    if ramp.deltas.ndim != 3:
        raise ValueError(
            f"ramp.deltas must be 3-D (N, H, W); got {ramp.deltas.shape}"
        )
    if ramp.deltas.shape[0] == 0:
        raise ValueError("ramp has zero reads")
    if ramp.deltas.shape[1:] != correction.coefficients.shape[1:]:
        raise ValueError(
            f"ramp H,W = {ramp.deltas.shape[1:]} does not match "
            f"correction H,W = {correction.coefficients.shape[1:]}"
        )

    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)  # (N, H, W)

    t = correction.model.evaluate(correction.coefficients, m)
    oor = (m < correction.fitMin[None]) | (m > correction.fitMax[None])

    # Bad-pixel pass-through: copy input m for any pixel with badPixelMask != 0.
    bad = correction.badPixelMask != 0
    if bad.any():
        t = np.where(bad[None], m, t)

    return LinearizedRamp(
        cumulativeLinear=t.astype(np.float32),
        outOfRangeMask=oor,
        badPixelMask=correction.badPixelMask.copy(),
    )


def applyFrame(
    correction: LinearityCorrection, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize a single already-cumulated frame."""
    m = np.asarray(m, dtype=np.float32)
    if m.shape != correction.coefficients.shape[1:]:
        raise ValueError(
            f"m shape {m.shape} does not match correction "
            f"H,W = {correction.coefficients.shape[1:]}"
        )

    t = correction.model.evaluate(correction.coefficients, m)
    oor = (m < correction.fitMin) | (m > correction.fitMax)

    bad = correction.badPixelMask != 0
    if bad.any():
        t = np.where(bad, m, t)

    return t.astype(np.float32), oor
