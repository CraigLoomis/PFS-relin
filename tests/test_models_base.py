"""Tests for the Model protocol and BlockFitResult dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from lsst.obs.pfs.h4Linearity.models.base import BlockFitResult, Model


def test_blockFitResultConstruction():
    H, W = 4, 5
    result = BlockFitResult(
        coefficients=np.zeros((5, H, W), dtype=np.float32),
        fitMin=np.zeros((H, W), dtype=np.float32),
        fitMax=np.ones((H, W), dtype=np.float32),
        residualRms=np.zeros((H, W), dtype=np.float32),
        maxAbsResidual=np.zeros((H, W), dtype=np.float32),
        nPointsUsed=np.full((H, W), 29, dtype=np.int32),
        conditionNumber=np.ones((H, W), dtype=np.float32),
        monotonic=np.ones((H, W), dtype=bool),
        badPixelMask=np.zeros((H, W), dtype=np.uint8),
    )
    assert result.coefficients.shape == (5, H, W)
    assert result.nPointsUsed.dtype == np.int32


def test_blockFitResultIsFrozen():
    H, W = 4, 5
    result = BlockFitResult(
        coefficients=np.zeros((5, H, W), dtype=np.float32),
        fitMin=np.zeros((H, W), dtype=np.float32),
        fitMax=np.zeros((H, W), dtype=np.float32),
        residualRms=np.zeros((H, W), dtype=np.float32),
        maxAbsResidual=np.zeros((H, W), dtype=np.float32),
        nPointsUsed=np.zeros((H, W), dtype=np.int32),
        conditionNumber=np.zeros((H, W), dtype=np.float32),
        monotonic=np.ones((H, W), dtype=bool),
        badPixelMask=np.zeros((H, W), dtype=np.uint8),
    )
    with pytest.raises(Exception):
        result.coefficients = np.zeros((5, H, W), dtype=np.float32)


def test_modelProtocolAcceptsDuckCompliantClass():
    """A class providing the required methods is recognized as a Model."""

    class FakeModel:
        modelName = "FAKE"

        def fitBlock(self, m, t, valid, conditionNumberLimit):
            raise NotImplementedError

        def evaluate(self, coefficients, m):
            raise NotImplementedError

        def isMonotonic(self, coefficients, mMin, mMax):
            raise NotImplementedError

        def toFitsHdus(self, correction):
            raise NotImplementedError

        @classmethod
        def fromFitsHdus(cls, hdus):
            raise NotImplementedError

    instance: Model = FakeModel()  # type: ignore[assignment]
    # isinstance against a runtime-checkable Protocol:
    assert isinstance(instance, Model)
