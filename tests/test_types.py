"""Tests for data types and bad-pixel flag constants."""

from __future__ import annotations

import numpy as np
import pytest

from nirLinearity.types import (
    BORDER_PIX,
    MASKED_BY_INPUT,
    INSUFFICIENT_POINTS,
    FIT_FAILED,
    NON_MONOTONIC,
    Ramp,
    LinearizedRamp,
    Diagnostics,
    LinearityCorrection,
)


def test_badPixelFlagsAreDistinctPowersOfTwo():
    flags = [MASKED_BY_INPUT, INSUFFICIENT_POINTS, FIT_FAILED, NON_MONOTONIC, BORDER_PIX]
    assert flags == [0x01, 0x02, 0x04, 0x08, 0x10]
    # Pairwise AND is zero — independent bits
    for i, a in enumerate(flags):
        for b in flags[i + 1:]:
            assert a & b == 0


def test_rampConstructsWithoutMask():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    ramp = Ramp(reads=deltas)
    assert ramp.reads.shape == (3, 4, 5)
    assert ramp.validMask is None


def test_rampConstructsWithMask():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=np.uint8)
    ramp = Ramp(reads=deltas, validMask=mask)
    assert ramp.validMask is not None
    assert ramp.validMask.shape == (4, 5)


def test_rampIsFrozen():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    ramp = Ramp(reads=deltas)
    with pytest.raises(Exception):  # FrozenInstanceError
        ramp.reads = np.zeros((3, 4, 5), dtype=np.float32)


def test_linearizedRampConstruction():
    lin = LinearizedRamp(
        cumulativeLinear=np.zeros((3, 4, 5), dtype=np.float32),
        outOfRangeMask=np.zeros((3, 4, 5), dtype=bool),
        badPixelMask=np.zeros((4, 5), dtype=np.uint8),
    )
    assert lin.cumulativeLinear.shape == (3, 4, 5)
    assert lin.outOfRangeMask.dtype == bool
    assert lin.badPixelMask.dtype == np.uint8


def test_diagnosticsConstruction():
    H, W = 4, 5
    diag = Diagnostics(
        residualRms=np.zeros((H, W), dtype=np.float32),
        maxAbsResidual=np.zeros((H, W), dtype=np.float32),
        nPointsUsed=np.zeros((H, W), dtype=np.int32),
        monotonic=np.ones((H, W), dtype=bool),
        conditionNumber=np.zeros((H, W), dtype=np.float32),
        summary={"badPixelFraction": 0.0},
    )
    assert diag.summary["badPixelFraction"] == 0.0


def test_linearityCorrectionConstruction():
    # Use a dummy model placeholder — real models come in a later task.
    class DummyModel:
        pass

    H, W = 4, 5
    correction = LinearityCorrection(
        model=DummyModel(),
        coefficients=np.zeros((5, H, W), dtype=np.float32),
        fitMin=np.zeros((H, W), dtype=np.float32),
        fitMax=np.ones((H, W), dtype=np.float32),
        badPixelMask=np.zeros((H, W), dtype=np.uint8),
        diagnostics=Diagnostics(
            residualRms=np.zeros((H, W), dtype=np.float32),
            maxAbsResidual=np.zeros((H, W), dtype=np.float32),
            nPointsUsed=np.zeros((H, W), dtype=np.int32),
            monotonic=np.ones((H, W), dtype=bool),
            conditionNumber=np.zeros((H, W), dtype=np.float32),
            summary={},
        ),
    )
    assert correction.coefficients.shape == (5, H, W)
