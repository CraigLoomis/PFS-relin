"""Tests for the model registry and package exports."""

from __future__ import annotations

from relin.models import MODEL_REGISTRY, PolynomialModel, registerModel


def test_polynomialRegistered():
    assert MODEL_REGISTRY["POLYNOMIAL"] is PolynomialModel


def test_registerModelAddsEntry():
    class DummyModel:
        modelName = "DUMMY"

        def fitBlock(self, m, t, valid, conditionNumberLimit): ...
        def evaluate(self, coefficients, m): ...
        def isMonotonic(self, coefficients, mMin, mMax): ...
        def toFitsHdus(self, correction): ...
        @classmethod
        def fromFitsHdus(cls, hdus): ...

    try:
        registerModel(DummyModel)
        assert MODEL_REGISTRY["DUMMY"] is DummyModel
    finally:
        # Clean up so other tests aren't affected.
        MODEL_REGISTRY.pop("DUMMY", None)


def test_registerModelRejectsDuplicateByDefault():
    import pytest
    with pytest.raises(ValueError):
        registerModel(PolynomialModel)  # POLYNOMIAL already registered
