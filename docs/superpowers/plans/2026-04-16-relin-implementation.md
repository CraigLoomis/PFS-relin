# relin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `relin` package — per-pixel polynomial nonlinearity fitting + application for up-the-ramp IR-detector calibration data — per the spec at `docs/superpowers/specs/2026-04-16-relin-package-design.md`.

**Architecture:** Pluggable `Model` protocol with polynomial MVP; batched-normal-equations fitter tiled over `(H, W)`; FITS persistence; all data flows through frozen dataclasses (`Ramp`, `LinearityCorrection`, etc.). Photodiode correction is the caller's responsibility — the package only sees already-corrected `deltas`.

**Tech Stack:** Python 3.12, `numpy`, `scipy`, `astropy`, `pytest`, managed with `uv`; `uv_build` build backend.

**Style:** camelCase for functions/methods/variables; PascalCase for classes; `UPPER_SNAKE_CASE` for constants. Module filenames stay short/single-word where possible.

**Working directory:** `/Users/cloomis/claude/relin`. All paths below are relative to it.

---

## Task 1: Project scaffolding and `uv` setup

Create the `pyproject.toml`, `README.md`, package directory skeleton, and verify that `uv sync` provisions a working environment that can run `pytest` (with no tests yet).

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `python/relin/__init__.py` (empty for now)
- Create: `python/relin/models/__init__.py` (empty for now)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_smoke.py`

### - [ ] Step 1: Write `pyproject.toml`

```toml
[build-system]
requires = ["uv_build>=0.5"]
build-backend = "uv_build"

[project]
name = "relin"
version = "0.1.0"
description = "Per-pixel nonlinearity correction for IR detector ramps"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "astropy>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.5",
]

[tool.uv.build-backend]
# Our package lives at python/relin/ rather than the default src/relin/.
module-name = "relin"
module-root = "python"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
pythonpath = ["python"]

[tool.ruff]
line-length = 110

[tool.ruff.lint]
# Keep the default E/F/W/I rules; do NOT enable pep8-naming ("N") rules
# because this project uses camelCase for functions/variables.
select = ["E", "F", "W", "I"]
```

### - [ ] Step 2: Write `README.md`

```markdown
# relin

Per-pixel nonlinearity correction for IR detector ramps.

## Install (development)

```bash
uv sync
```

## Run tests

```bash
uv run pytest
```

## Design

See `docs/superpowers/specs/2026-04-16-relin-package-design.md`.
```

### - [ ] Step 3: Create empty package files

```bash
touch python/relin/__init__.py
touch python/relin/models/__init__.py
touch tests/__init__.py
```

### - [ ] Step 4: Write `tests/test_smoke.py` (the failing test)

```python
"""Smoke test: the package imports cleanly."""

def test_packageImports():
    import relin
    assert relin is not None


def test_modelsSubpackageImports():
    import relin.models
    assert relin.models is not None
```

### - [ ] Step 5: Provision the environment and run the smoke test

```bash
uv sync
uv run pytest tests/test_smoke.py -v
```

Expected: both tests PASS. Output includes `2 passed`.

If `uv` is not installed: `brew install uv` (macOS) or see https://github.com/astral-sh/uv.

### - [ ] Step 6: Commit

```bash
git add pyproject.toml README.md python/relin/__init__.py python/relin/models/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "Add project scaffolding with uv and hatchling"
```

---

## Task 2: `types.py` — dataclasses and bad-pixel flag constants

Define `Ramp`, `LinearizedRamp`, `Diagnostics`, `LinearityCorrection`, and the four bit-flag constants. These are the data-carrying types used by every other module.

**Files:**
- Create: `python/relin/types.py`
- Create: `tests/test_types.py`

### - [ ] Step 1: Write the failing tests (`tests/test_types.py`)

```python
"""Tests for data types and bad-pixel flag constants."""

from __future__ import annotations

import numpy as np
import pytest

from relin.types import (
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
    flags = [MASKED_BY_INPUT, INSUFFICIENT_POINTS, FIT_FAILED, NON_MONOTONIC]
    assert flags == [0x01, 0x02, 0x04, 0x08]
    # Pairwise AND is zero — independent bits
    for i, a in enumerate(flags):
        for b in flags[i + 1:]:
            assert a & b == 0


def test_rampConstructsWithoutMask():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    ramp = Ramp(deltas=deltas)
    assert ramp.deltas.shape == (3, 4, 5)
    assert ramp.validMask is None


def test_rampConstructsWithMask():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=np.uint8)
    ramp = Ramp(deltas=deltas, validMask=mask)
    assert ramp.validMask is not None
    assert ramp.validMask.shape == (4, 5)


def test_rampIsFrozen():
    deltas = np.zeros((3, 4, 5), dtype=np.float32)
    ramp = Ramp(deltas=deltas)
    with pytest.raises(Exception):  # FrozenInstanceError
        ramp.deltas = np.zeros((3, 4, 5), dtype=np.float32)


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
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'relin.types'` or similar — all tests ERROR at import.

### - [ ] Step 3: Implement `python/relin/types.py`

```python
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
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_types.py -v
```

Expected: all 7 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/types.py tests/test_types.py
git commit -m "Add types module with dataclasses and bad-pixel flag constants"
```

---

## Task 3: `models/base.py` — Model protocol and BlockFitResult

Define the `Model` protocol all concrete fit forms will implement, and the `BlockFitResult` dataclass that `fitBlock` returns.

**Files:**
- Create: `python/relin/models/base.py`
- Create: `tests/test_models_base.py`

### - [ ] Step 1: Write the failing tests

```python
"""Tests for the Model protocol and BlockFitResult dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from relin.models.base import BlockFitResult, Model


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
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_models_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'relin.models.base'`.

### - [ ] Step 3: Implement `python/relin/models/base.py`

```python
"""The Model protocol and BlockFitResult — shared across all concrete model forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BlockFitResult:
    """Everything a model's ``fitBlock`` must return for a single tile."""

    coefficients: np.ndarray       # model-specific shape; polynomial: (order+1, hTile, wTile) float32
    fitMin: np.ndarray             # (hTile, wTile) float32 — min m used per pixel
    fitMax: np.ndarray             # (hTile, wTile) float32 — max m used per pixel
    residualRms: np.ndarray        # (hTile, wTile) float32
    maxAbsResidual: np.ndarray     # (hTile, wTile) float32
    nPointsUsed: np.ndarray        # (hTile, wTile) int32
    conditionNumber: np.ndarray    # (hTile, wTile) float32
    monotonic: np.ndarray          # (hTile, wTile) bool
    badPixelMask: np.ndarray       # (hTile, wTile) uint8 — fit-time flags only
                                   # (INSUFFICIENT_POINTS, FIT_FAILED, NON_MONOTONIC)


@runtime_checkable
class Model(Protocol):
    """Protocol implemented by every concrete fit form (polynomial, spline, ...)."""

    modelName: str  # e.g. "POLYNOMIAL"; written into the FITS PRIMARY header

    def fitBlock(
        self,
        m: np.ndarray,                  # (nPoints, hTile, wTile) float32
        t: np.ndarray,                  # (nPoints,) float32
        valid: np.ndarray,              # (nPoints, hTile, wTile) bool
        conditionNumberLimit: float,
    ) -> BlockFitResult: ...

    def evaluate(
        self, coefficients: np.ndarray, m: np.ndarray
    ) -> np.ndarray: ...

    def isMonotonic(
        self, coefficients: np.ndarray, mMin: np.ndarray, mMax: np.ndarray
    ) -> np.ndarray: ...

    def toFitsHdus(self, correction: Any) -> list: ...  # list[astropy.io.fits.HDU]

    @classmethod
    def fromFitsHdus(cls, hdus: list) -> tuple["Model", np.ndarray]: ...
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_models_base.py -v
```

Expected: all 3 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/models/base.py tests/test_models_base.py
git commit -m "Add Model protocol and BlockFitResult"
```

---

## Task 4: `PolynomialModel` — constructor, `evaluate`, `isMonotonic`

Build the `PolynomialModel` class with the parts that don't require fitting yet: the constructor, Horner evaluation, and the post-fit monotonicity check. `fitBlock` and the FITS methods come in later tasks.

**Files:**
- Create: `python/relin/models/polynomial.py`
- Create: `tests/test_polynomial_model.py`

### - [ ] Step 1: Write the failing tests

```python
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
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: all tests ERROR — `ModuleNotFoundError: No module named 'relin.models.polynomial'`.

### - [ ] Step 3: Implement `python/relin/models/polynomial.py` (partial — constructor, evaluate, isMonotonic)

```python
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
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: all 9 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Add PolynomialModel constructor, evaluate, and isMonotonic"
```

---

## Task 5: `conftest.py` — synthetic ramp fixtures

Add shared pytest fixtures that build synthetic ramps whose true per-pixel polynomial response is known. These feed the fit tests in Task 6 and later.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_conftest_fixtures.py` (quick sanity test of the fixtures themselves)

### - [ ] Step 1: Write `tests/conftest.py`

```python
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
```

### - [ ] Step 2: Write the sanity test (`tests/test_conftest_fixtures.py`)

```python
"""Sanity checks that the synthetic-ramp fixtures behave as documented."""

from __future__ import annotations

import numpy as np


def test_tinyLinearRampBuildsCorrectly(tinyLinearRamp):
    ramp, truth = tinyLinearRamp
    N, H, W = ramp.deltas.shape
    assert (N, H, W) == (6, 2, 3)
    # Every read's delta equals rate * pixelScale
    expected = truth["targetRate"] * truth["pixelScale"]
    for n in range(N):
        np.testing.assert_allclose(ramp.deltas[n], expected, rtol=1e-6)


def test_smallSyntheticRampBuildsCorrectly(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    N, H, W = ramp.deltas.shape
    assert (N, H, W) == (29, 4, 5)
    # Verify the reconstructed polynomial: for each pixel and each read,
    # evaluate the true polynomial at the implied m and check it equals target.
    m = np.cumsum(ramp.deltas.astype(np.float64), axis=0)  # (N, H, W)
    polyVal = (
        truth["c0"] + truth["c1"] * m + truth["c2"] * m ** 2
        + truth["c3"] * m ** 3 + truth["c4"] * m ** 4
    )
    target = truth["target"]
    for n in range(N):
        np.testing.assert_allclose(polyVal[n], target[n], rtol=1e-3, atol=1e-2)
```

### - [ ] Step 3: Run tests, verify they pass

```bash
uv run pytest tests/test_conftest_fixtures.py -v
```

Expected: both tests PASS.

### - [ ] Step 4: Commit

```bash
git add tests/conftest.py tests/test_conftest_fixtures.py
git commit -m "Add synthetic ramp fixtures for fit testing"
```

---

## Task 6: `PolynomialModel.fitBlock`

Implement the batched-normal-equations solver with per-pixel scaling (to keep the normal-equation matrix well-conditioned regardless of raw DN magnitude).

**Files:**
- Modify: `python/relin/models/polynomial.py`
- Modify: `tests/test_polynomial_model.py`

### - [ ] Step 1: Append failing tests for `fitBlock` to `tests/test_polynomial_model.py`

```python
# Append to tests/test_polynomial_model.py

from relin.types import (
    MASKED_BY_INPUT,  # noqa: F401  (used transitively; referenced below)
    INSUFFICIENT_POINTS,
    FIT_FAILED,
)


def test_fitBlockRecoversLinearCoefficients(tinyLinearRamp):
    """With perfectly linear data, recovered coefficients should have c0 ≈ 0
    and c1 exactly capturing the scaling needed to hit the target rate."""
    ramp, truth = tinyLinearRamp
    N, H, W = ramp.deltas.shape
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    t = truth["target"].astype(np.float32)
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (3, H, W)
    # For pixel (h, w): m[n] = rate * pixelScale[h,w] * (n+1), t[n] = rate * (n+1)
    # so t = m / pixelScale[h,w]. Expect c0 ≈ 0, c1 ≈ 1/pixelScale, c2 ≈ 0.
    np.testing.assert_allclose(result.coefficients[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(
        result.coefficients[1], 1.0 / truth["pixelScale"], rtol=1e-4
    )
    np.testing.assert_allclose(result.coefficients[2], 0.0, atol=1e-6)
    assert (result.badPixelMask == 0).all()
    assert (result.nPointsUsed == N).all()


def test_fitBlockRecoversPolynomialCoefficients(smallSyntheticRamp):
    """Fit the known 4th-order synthetic ramp and verify coefficients are recovered."""
    ramp, truth = smallSyntheticRamp
    N, H, W = ramp.deltas.shape
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    t = truth["target"].astype(np.float32)
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=4)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (5, H, W)
    # Coefficients should match within a loose tolerance — synthetic data is
    # constructed exactly but Newton-iteration residuals and float32 arithmetic
    # limit the recovery precision.
    np.testing.assert_allclose(result.coefficients[0], truth["c0"], atol=1.0)
    np.testing.assert_allclose(
        result.coefficients[1], truth["c1"], rtol=1e-3, atol=1e-3
    )
    assert (result.badPixelMask == 0).all()
    # The residuals should be small: each pixel's target-minus-prediction is
    # close to zero.
    assert result.residualRms.max() < 1.0


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


def test_fitBlockRespectsForceThroughOrigin(tinyLinearRamp):
    """With forceThroughOrigin, c0 must be exactly zero."""
    ramp, truth = tinyLinearRamp
    N, H, W = ramp.deltas.shape
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    t = truth["target"].astype(np.float32)
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2, forceThroughOrigin=True)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    np.testing.assert_array_equal(result.coefficients[0], 0.0)
    np.testing.assert_allclose(
        result.coefficients[1], 1.0 / truth["pixelScale"], rtol=1e-4
    )
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: the five new tests fail with `AttributeError: 'PolynomialModel' object has no attribute 'fitBlock'`.

### - [ ] Step 3: Implement `fitBlock` in `python/relin/models/polynomial.py`

Add the `fitBlock` method to `PolynomialModel`. Keep the imports at the top of the file up to date (`BlockFitResult` is now used).

```python
# At the top of python/relin/models/polynomial.py, replace the existing import:
from relin.models.base import BlockFitResult
from relin.types import FIT_FAILED, INSUFFICIENT_POINTS, NON_MONOTONIC
```

Then add this method to the `PolynomialModel` class:

```python
    def fitBlock(
        self,
        m: np.ndarray,
        t: np.ndarray,
        valid: np.ndarray,
        conditionNumberLimit: float,
    ) -> BlockFitResult:
        """Fit a polynomial at every pixel in the block.

        Uses per-pixel rescaling of ``m`` to the range [0, 1] before forming
        the normal equations, which keeps the conditioning bounded independent
        of raw DN magnitude. Coefficients are unscaled at the end.
        """
        nPoints, H, W = m.shape
        p = self.order
        fto = self.forceThroughOrigin
        startExp = 1 if fto else 0
        nCoefs = p + 1 - startExp  # free coefficients

        mD = m.astype(np.float64)
        v64 = valid.astype(np.float64)
        t64 = t.astype(np.float64)

        # Count valid points per pixel
        nPointsUsed = valid.sum(axis=0).astype(np.int32)  # (H, W)
        badMask = np.zeros((H, W), dtype=np.uint8)

        # fitMin / fitMax: min/max of m over valid reads per pixel
        mMasked = np.where(valid, mD, np.nan)
        with np.errstate(invalid="ignore"):
            fitMin = np.nanmin(mMasked, axis=0)
            fitMax = np.nanmax(mMasked, axis=0)
        fitMin = np.where(np.isnan(fitMin), 0.0, fitMin)
        fitMax = np.where(np.isnan(fitMax), 0.0, fitMax)

        # Per-pixel scaling factor: divide m by max(|m|) so scaled m is in [-1, 1].
        # This keeps the normal-equation matrix well-conditioned.
        scale = np.maximum(np.abs(fitMin), np.abs(fitMax))
        scale = np.where(scale > 0, scale, 1.0)  # avoid /0 for degenerate pixels
        mScaled = mD / scale[None]  # (N, H, W)

        # Flag insufficient-points pixels now.
        insufficientPixels = nPointsUsed < (nCoefs + 1)
        badMask[insufficientPixels] |= INSUFFICIENT_POINTS

        # Accumulate AtA (upper triangular + symmetrize) and Atb in scaled space.
        AtA = np.zeros((H, W, nCoefs, nCoefs), dtype=np.float64)
        Atb = np.zeros((H, W, nCoefs), dtype=np.float64)

        # Exponents needed in the fit: startExp .. startExp + nCoefs - 1
        exps = np.arange(startExp, startExp + nCoefs, dtype=np.int32)

        # For AtA: need mScaled ** (expI + expJ), expI, expJ in exps.
        # For Atb: need mScaled ** expI with t weighting.
        # Iterate over exponent sums from 2*startExp to 2*(startExp + nCoefs - 1).
        for i in range(nCoefs):
            expI = int(exps[i])
            # Precompute v * mScaled^expI once per i
            miPow = mScaled ** expI  # (N, H, W)
            vMiPow = v64 * miPow
            # Atb[h, w, i] = Σ_n vMiPow[n, h, w] * t[n]
            Atb[..., i] = (vMiPow * t64[:, None, None]).sum(axis=0)
            for j in range(i, nCoefs):
                expJ = int(exps[j])
                mSumPow = mScaled ** (expI + expJ)  # (N, H, W)
                val = (v64 * mSumPow).sum(axis=0)  # (H, W)
                AtA[..., i, j] = val
                if i != j:
                    AtA[..., j, i] = val

        # Condition number BEFORE any modification — captures singular cases.
        with np.errstate(divide="ignore", invalid="ignore"):
            conditionNumber = np.linalg.cond(AtA)
        conditionNumber = np.nan_to_num(conditionNumber, nan=np.inf, posinf=np.inf)

        # Identify ill-conditioned or insufficient pixels; flag FIT_FAILED for
        # the ill-conditioned set (only among those that have enough points).
        fitFailed = (~insufficientPixels) & (conditionNumber > conditionNumberLimit)
        badMask[fitFailed] |= FIT_FAILED

        # For pixels we won't solve, replace AtA with identity so np.linalg.solve
        # doesn't raise for the whole batch.
        skip = insufficientPixels | fitFailed  # (H, W)
        identityBlock = np.eye(nCoefs, dtype=np.float64)
        AtA[skip] = identityBlock
        Atb[skip] = 0.0

        # Batched solve.
        try:
            solScaled = np.linalg.solve(AtA, Atb[..., None])[..., 0]  # (H, W, nCoefs)
        except np.linalg.LinAlgError:
            # Extremely defensive: solve pixel-by-pixel and flag failures.
            solScaled = np.zeros((H, W, nCoefs), dtype=np.float64)
            for hi in range(H):
                for wi in range(W):
                    if skip[hi, wi]:
                        continue
                    try:
                        solScaled[hi, wi] = np.linalg.solve(
                            AtA[hi, wi], Atb[hi, wi]
                        )
                    except np.linalg.LinAlgError:
                        badMask[hi, wi] |= FIT_FAILED
                        solScaled[hi, wi] = 0.0
            skip = skip | (badMask & FIT_FAILED != 0)

        # Unscale coefficients: in scaled space t = Σ c_scaled[k] (m/scale)^expK.
        # In original space t = Σ (c_scaled[k] / scale^expK) m^expK.
        unscaleFactors = scale[..., None] ** exps  # (H, W, nCoefs)
        solUnscaled = solScaled / unscaleFactors

        # Stitch into (order+1, H, W), filling 0 for the missing c0 when fto.
        coefficients = np.zeros((p + 1, H, W), dtype=np.float32)
        for k, e in enumerate(exps):
            coefficients[int(e)] = solUnscaled[..., k].astype(np.float32)
        coefficients[:, skip] = 0.0  # Zero out failed/insufficient pixels explicitly

        # Residuals: evaluate fit at each read and compare to t.
        tPred = self.evaluate(coefficients, m.astype(np.float32))  # (N, H, W)
        residuals = (t[:, None, None].astype(np.float32) - tPred) * valid
        nForDiv = np.where(nPointsUsed > 0, nPointsUsed, 1).astype(np.float32)
        residualRms = np.sqrt((residuals ** 2).sum(axis=0) / nForDiv).astype(np.float32)
        maxAbsResidual = np.abs(residuals).max(axis=0).astype(np.float32)

        # Monotonicity check (only meaningful for non-skipped pixels, but we
        # compute it everywhere and overwrite skipped ones below).
        monotonic = self.isMonotonic(
            coefficients, fitMin.astype(np.float32), fitMax.astype(np.float32)
        )
        # For skipped pixels, monotonicity is undefined — set False without flagging.
        monotonic[skip] = False
        # For non-skipped pixels that are non-monotonic, flag NON_MONOTONIC.
        nonMono = (~skip) & (~monotonic)
        badMask[nonMono] |= NON_MONOTONIC

        return BlockFitResult(
            coefficients=coefficients,
            fitMin=fitMin.astype(np.float32),
            fitMax=fitMax.astype(np.float32),
            residualRms=residualRms,
            maxAbsResidual=maxAbsResidual,
            nPointsUsed=nPointsUsed,
            conditionNumber=conditionNumber.astype(np.float32),
            monotonic=monotonic,
            badPixelMask=badMask,
        )
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: all tests (original 9 + 5 new) PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Add PolynomialModel.fitBlock with per-pixel scaling"
```

---

## Task 7: `PolynomialModel` FITS serialization

Add `toFitsHdus` and `fromFitsHdus` to `PolynomialModel`. The model emits the `COEFFS` ImageHDU only; the surrounding `LinearityCorrection` arrays (fitMin, fitMax, etc.) are serialized by `io.py` in a later task.

**Files:**
- Modify: `python/relin/models/polynomial.py`
- Modify: `tests/test_polynomial_model.py`

### - [ ] Step 1: Append failing tests

```python
# Append to tests/test_polynomial_model.py

from astropy.io import fits


def test_polynomialModelFitsRoundTrip():
    H, W = 2, 3
    coeffs = np.arange(5 * H * W, dtype=np.float32).reshape(5, H, W)
    model = PolynomialModel(order=4, forceThroughOrigin=False)

    # Build a minimal correction-like object with just what to/fromFitsHdus read.
    class Stub:
        def __init__(self, c):
            self.coefficients = c
    hdus = model.toFitsHdus(Stub(coeffs))

    # Expect exactly one ImageHDU named "COEFFS".
    assert len(hdus) == 1
    assert hdus[0].name == "COEFFS"
    np.testing.assert_array_equal(hdus[0].data, coeffs)

    # Header must record ORDER and FTHROUGH0.
    assert hdus[0].header["ORDER"] == 4
    assert hdus[0].header["FTHROUGH0"] is False

    # Round-trip through from_fits_hdus.
    loadedModel, loadedCoeffs = PolynomialModel.fromFitsHdus(hdus)
    assert loadedModel.order == 4
    assert loadedModel.forceThroughOrigin is False
    np.testing.assert_array_equal(loadedCoeffs, coeffs)


def test_polynomialModelFitsForceThroughOrigin():
    H, W = 2, 3
    coeffs = np.zeros((4, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    model = PolynomialModel(order=3, forceThroughOrigin=True)

    class Stub:
        def __init__(self, c):
            self.coefficients = c
    hdus = model.toFitsHdus(Stub(coeffs))

    assert hdus[0].header["FTHROUGH0"] is True
    loadedModel, loadedCoeffs = PolynomialModel.fromFitsHdus(hdus)
    assert loadedModel.order == 3
    assert loadedModel.forceThroughOrigin is True
    np.testing.assert_array_equal(loadedCoeffs, coeffs)
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: the two new tests fail with `AttributeError` on `toFitsHdus`.

### - [ ] Step 3: Add FITS methods to `python/relin/models/polynomial.py`

```python
# At the top of the file, add:
from astropy.io import fits

# Add these methods to the PolynomialModel class:

    def toFitsHdus(self, correction) -> list[fits.ImageHDU]:
        """Serialize model coefficients to a single ImageHDU named COEFFS."""
        hdu = fits.ImageHDU(data=correction.coefficients, name="COEFFS")
        hdu.header["ORDER"] = (self.order, "polynomial order")
        hdu.header["FTHROUGH0"] = (
            self.forceThroughOrigin,
            "polynomial forced through origin (c0 == 0)",
        )
        hdu.header["COMMENT"] = "COEFFS axis 0 is the coefficient index; C0 first."
        return [hdu]

    @classmethod
    def fromFitsHdus(cls, hdus) -> tuple["PolynomialModel", np.ndarray]:
        """Reconstruct a PolynomialModel + coefficients from HDUs written by toFitsHdus."""
        coeffsHdu = None
        for hdu in hdus:
            if getattr(hdu, "name", "") == "COEFFS":
                coeffsHdu = hdu
                break
        if coeffsHdu is None:
            raise ValueError("No COEFFS HDU found in provided hdus")
        order = int(coeffsHdu.header["ORDER"])
        fto = bool(coeffsHdu.header["FTHROUGH0"])
        coefficients = np.asarray(coeffsHdu.data, dtype=np.float32)
        return cls(order=order, forceThroughOrigin=fto), coefficients
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_polynomial_model.py -v
```

Expected: all tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Add PolynomialModel FITS serialization"
```

---

## Task 8: Model registry — `models/__init__.py`

Expose `PolynomialModel` and a module-level registry so `io.loadFits` can look models up by the `MODEL` header string.

**Files:**
- Modify: `python/relin/models/__init__.py`
- Create: `tests/test_models_init.py`

### - [ ] Step 1: Write the failing tests

```python
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
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_models_init.py -v
```

Expected: `ImportError: cannot import name 'MODEL_REGISTRY' from 'relin.models'`.

### - [ ] Step 3: Implement `python/relin/models/__init__.py`

```python
"""Model registry — lookup table from MODEL string (FITS header) to model class."""

from __future__ import annotations

from relin.models.base import BlockFitResult, Model
from relin.models.polynomial import PolynomialModel

MODEL_REGISTRY: dict[str, type[Model]] = {}


def registerModel(modelClass: type[Model], *, overwrite: bool = False) -> None:
    """Register a model class under its ``modelName`` attribute."""
    name = modelClass.modelName  # type: ignore[attr-defined]
    if name in MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"model name {name!r} already registered; "
            f"pass overwrite=True to replace"
        )
    MODEL_REGISTRY[name] = modelClass


# Register the built-in model.
registerModel(PolynomialModel)

__all__ = [
    "MODEL_REGISTRY",
    "Model",
    "BlockFitResult",
    "PolynomialModel",
    "registerModel",
]
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_models_init.py -v
uv run pytest -v  # full suite sanity check
```

Expected: all tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/models/__init__.py tests/test_models_init.py
git commit -m "Add model registry with PolynomialModel self-registration"
```

---

## Task 9: Top-level `fit()` function

Implement the tiled driver that accepts one or more `Ramp`s, computes per-ramp targets, iterates over `(H, W)` tiles, delegates to `model.fitBlock`, stitches the per-tile `BlockFitResult`s into a full-frame `LinearityCorrection`, and populates `Diagnostics.summary`.

**Files:**
- Create: `python/relin/fit.py`
- Create: `tests/test_fit.py`

### - [ ] Step 1: Write the failing tests

```python
"""Tests for the top-level fit() function."""

from __future__ import annotations

import numpy as np

from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import MASKED_BY_INPUT, Ramp


def test_fitSingleRampRecoversCoefficients(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    assert correction.coefficients.shape == (5, 4, 5)
    # Leading coefficients should roughly match truth.
    np.testing.assert_allclose(
        correction.coefficients[1], truth["c1"], rtol=1e-3, atol=1e-3
    )
    assert (correction.badPixelMask == 0).all()
    # Summary should carry percentiles.
    assert "residualRmsP50" in correction.diagnostics.summary
    assert "residualRmsP95" in correction.diagnostics.summary
    assert "residualRmsP99" in correction.diagnostics.summary


def test_fitTilingIsDeterministic(smallSyntheticRamp):
    """Fitting with different block sizes must yield identical coefficients
    (within float32 precision)."""
    ramp, _ = smallSyntheticRamp
    c1 = fit([ramp], blockSize=(4, 5)).coefficients
    c2 = fit([ramp], blockSize=(2, 3)).coefficients
    np.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


def test_fitPropagatesInputMask(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    mask = np.zeros(ramp.deltas.shape[1:], dtype=np.uint8)
    mask[0, 0] = 1  # Mark pixel (0, 0) as invalid
    maskedRamp = Ramp(deltas=ramp.deltas, validMask=mask)
    correction = fit([maskedRamp], model=PolynomialModel(order=1))
    assert correction.badPixelMask[0, 0] & MASKED_BY_INPUT
    assert correction.badPixelMask[0, 1] == 0


def test_fitMultipleRampsConcatenates():
    """Two ramps of different lengths combine per-pixel."""
    rng = np.random.default_rng(0)
    H, W = 3, 4
    # Pixel-linear: t = m for every pixel.
    # Ramp 1: 8 reads, rate 100.
    # Ramp 2: 12 reads, rate 200.
    rate1 = 100.0
    rate2 = 200.0
    deltas1 = np.full((8, H, W), rate1, dtype=np.float32)
    deltas2 = np.full((12, H, W), rate2, dtype=np.float32)
    correction = fit(
        [Ramp(deltas=deltas1), Ramp(deltas=deltas2)],
        model=PolynomialModel(order=2),
    )
    # Expected: t = m identically, so c0 ≈ 0, c1 ≈ 1, c2 ≈ 0.
    np.testing.assert_allclose(correction.coefficients[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(correction.coefficients[1], 1.0, rtol=1e-4)
    np.testing.assert_allclose(correction.coefficients[2], 0.0, atol=1e-6)
    # nPointsUsed should be 8 + 12 = 20 everywhere.
    assert (correction.diagnostics.nPointsUsed == 20).all()


def test_fitEmptyRampListRaises():
    import pytest
    with pytest.raises(ValueError):
        fit([])
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_fit.py -v
```

Expected: `ModuleNotFoundError: No module named 'relin.fit'`.

### - [ ] Step 3: Implement `python/relin/fit.py`

```python
"""Top-level fit(): tile-iterate over (H, W) and delegate to model.fitBlock."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from relin.models import Model, PolynomialModel
from relin.types import (
    FIT_FAILED,
    INSUFFICIENT_POINTS,
    MASKED_BY_INPUT,
    NON_MONOTONIC,
    Diagnostics,
    LinearityCorrection,
    Ramp,
)


def fit(
    ramps: Sequence[Ramp],
    model: Model | None = None,
    blockSize: tuple[int, int] = (512, 512),
    conditionNumberLimit: float = 1e12,
) -> LinearityCorrection:
    """Fit a per-pixel nonlinearity correction from one or more ramps.

    See ``docs/superpowers/specs/2026-04-16-relin-package-design.md`` for the
    full algorithm description.
    """
    if model is None:
        model = PolynomialModel(order=4)

    if len(ramps) == 0:
        raise ValueError("fit() requires at least one ramp")

    # Validate shapes.
    H, W = ramps[0].deltas.shape[1:]
    for k, ramp in enumerate(ramps):
        if ramp.deltas.ndim != 3:
            raise ValueError(
                f"ramps[{k}].deltas must be 3-D (N, H, W); got {ramp.deltas.shape}"
            )
        if ramp.deltas.shape[1:] != (H, W):
            raise ValueError(
                f"ramps[{k}].deltas H,W = {ramp.deltas.shape[1:]} "
                f"does not match ramps[0] H,W = {(H, W)}"
            )
        if ramp.validMask is not None and ramp.validMask.shape != (H, W):
            raise ValueError(
                f"ramps[{k}].validMask shape {ramp.validMask.shape} != {(H, W)}"
            )

    # Per-ramp precomputation.
    cumulatives: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ramp in ramps:
        m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
        cumulatives.append(m)
        # Rate R_k: median of first-read deltas over caller-allowed pixels.
        firstDeltas = ramp.deltas[0].astype(np.float32)
        if ramp.validMask is not None:
            allowed = ramp.validMask == 0
            if allowed.any():
                rate = float(np.median(firstDeltas[allowed]))
            else:
                rate = float(np.median(firstDeltas))
        else:
            rate = float(np.median(firstDeltas))
        Nk = ramp.deltas.shape[0]
        targets.append(rate * np.arange(1, Nk + 1, dtype=np.float32))

    # Concatenated targets across ramps — used per tile.
    tConcat = np.concatenate(targets)

    # Preallocate full-frame outputs.
    # The shape of coefficients is determined by the model:
    # - PolynomialModel: (order+1, H, W)
    # We discover the coefficient shape by running a 1x1 dummy block first.
    coefShape = _peekCoefShape(model)
    coefficients = np.zeros((coefShape, H, W), dtype=np.float32)
    fitMin = np.zeros((H, W), dtype=np.float32)
    fitMax = np.zeros((H, W), dtype=np.float32)
    residualRms = np.zeros((H, W), dtype=np.float32)
    maxAbsResidual = np.zeros((H, W), dtype=np.float32)
    nPointsUsed = np.zeros((H, W), dtype=np.int32)
    conditionNumber = np.zeros((H, W), dtype=np.float32)
    monotonic = np.zeros((H, W), dtype=bool)
    badPixelMask = np.zeros((H, W), dtype=np.uint8)

    # Iterate over tiles.
    bH, bW = blockSize
    for rowStart in range(0, H, bH):
        rowEnd = min(rowStart + bH, H)
        for colStart in range(0, W, bW):
            colEnd = min(colStart + bW, W)
            tileH = rowEnd - rowStart
            tileW = colEnd - colStart

            # Assemble per-tile m and valid by concatenating ramps.
            mSegments = []
            validSegments = []
            for k, ramp in enumerate(ramps):
                mSegments.append(
                    cumulatives[k][:, rowStart:rowEnd, colStart:colEnd]
                )
                if ramp.validMask is not None:
                    vTile = (ramp.validMask[rowStart:rowEnd, colStart:colEnd] == 0)
                    validSegments.append(
                        np.broadcast_to(
                            vTile[None], (ramp.deltas.shape[0], tileH, tileW)
                        ).copy()
                    )
                else:
                    validSegments.append(
                        np.ones(
                            (ramp.deltas.shape[0], tileH, tileW), dtype=bool
                        )
                    )
            mTile = np.concatenate(mSegments, axis=0)
            validTile = np.concatenate(validSegments, axis=0)

            result = model.fitBlock(
                m=mTile, t=tConcat, valid=validTile,
                conditionNumberLimit=conditionNumberLimit,
            )

            coefficients[:, rowStart:rowEnd, colStart:colEnd] = result.coefficients
            fitMin[rowStart:rowEnd, colStart:colEnd] = result.fitMin
            fitMax[rowStart:rowEnd, colStart:colEnd] = result.fitMax
            residualRms[rowStart:rowEnd, colStart:colEnd] = result.residualRms
            maxAbsResidual[rowStart:rowEnd, colStart:colEnd] = result.maxAbsResidual
            nPointsUsed[rowStart:rowEnd, colStart:colEnd] = result.nPointsUsed
            conditionNumber[rowStart:rowEnd, colStart:colEnd] = result.conditionNumber
            monotonic[rowStart:rowEnd, colStart:colEnd] = result.monotonic
            badPixelMask[rowStart:rowEnd, colStart:colEnd] = result.badPixelMask

    # Propagate input masks: any nonzero validMask entry in any ramp sets MASKED_BY_INPUT.
    for ramp in ramps:
        if ramp.validMask is not None:
            inputBad = (ramp.validMask != 0)
            badPixelMask[inputBad] |= MASKED_BY_INPUT

    # Dataset-wide summary.
    goodPixels = (badPixelMask == 0)
    totalPixels = int(H * W)
    summary: dict = {
        "totalPixels": totalPixels,
        "goodPixelFraction": float(goodPixels.sum()) / totalPixels,
        "badPixelFraction_maskedByInput": float((badPixelMask & MASKED_BY_INPUT > 0).sum()) / totalPixels,
        "badPixelFraction_insufficientPoints": float((badPixelMask & INSUFFICIENT_POINTS > 0).sum()) / totalPixels,
        "badPixelFraction_fitFailed": float((badPixelMask & FIT_FAILED > 0).sum()) / totalPixels,
        "badPixelFraction_nonMonotonic": float((badPixelMask & NON_MONOTONIC > 0).sum()) / totalPixels,
        "modelName": model.modelName,
        "nRamps": len(ramps),
    }
    if goodPixels.any():
        goodRms = residualRms[goodPixels]
        summary["residualRmsP50"] = float(np.percentile(goodRms, 50))
        summary["residualRmsP95"] = float(np.percentile(goodRms, 95))
        summary["residualRmsP99"] = float(np.percentile(goodRms, 99))
    else:
        summary["residualRmsP50"] = float("nan")
        summary["residualRmsP95"] = float("nan")
        summary["residualRmsP99"] = float("nan")

    diagnostics = Diagnostics(
        residualRms=residualRms,
        maxAbsResidual=maxAbsResidual,
        nPointsUsed=nPointsUsed,
        monotonic=monotonic,
        conditionNumber=conditionNumber,
        summary=summary,
    )

    return LinearityCorrection(
        model=model,
        coefficients=coefficients,
        fitMin=fitMin,
        fitMax=fitMax,
        badPixelMask=badPixelMask,
        diagnostics=diagnostics,
    )


def _peekCoefShape(model: Model) -> int:
    """Return the first-axis size of coefficients the model will produce.

    For ``PolynomialModel``, this is ``order + 1`` regardless of
    ``forceThroughOrigin`` (the stored coefficients include a forced c0 = 0).
    Models that don't expose ``order`` must run a throwaway 1x1 fit.
    """
    if isinstance(model, PolynomialModel):
        return model.order + 1
    # Fallback: run a minimal 1x1 block fit with 2*(order+1) dummy points.
    nPoints = 8
    m = np.linspace(0.0, 1.0, nPoints, dtype=np.float32)[:, None, None]
    t = m[:, 0, 0].copy()
    valid = np.ones((nPoints, 1, 1), dtype=bool)
    result = model.fitBlock(
        m=m, t=t, valid=valid, conditionNumberLimit=1e12
    )
    return int(result.coefficients.shape[0])
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_fit.py -v
```

Expected: all 5 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/fit.py tests/test_fit.py
git commit -m "Add top-level fit() with tiled driver and summary diagnostics"
```

---

## Task 10: `apply.py` — `apply()` and `applyFrame()`

Consume a `LinearityCorrection` and produce linearized outputs. Bad pixels pass through untouched.

**Files:**
- Create: `python/relin/apply.py`
- Create: `tests/test_apply.py`

### - [ ] Step 1: Write the failing tests

```python
"""Tests for apply() and applyFrame()."""

from __future__ import annotations

import numpy as np
import pytest

from relin.apply import apply, applyFrame
from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import Ramp


def test_applyOnFittedRampYieldsTarget(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    result = apply(correction, ramp)
    # cumulativeLinear should closely match the true target at every read.
    expected = np.broadcast_to(
        truth["target"][:, None, None], ramp.deltas.shape
    )
    np.testing.assert_allclose(
        result.cumulativeLinear, expected, rtol=1e-3, atol=1e-1
    )
    # No pixel should be out-of-range on the same data it was fit on.
    assert not result.outOfRangeMask.any()


def test_applyFlagsOutOfRangeForExtrapolation():
    rng = np.random.default_rng(0)
    H, W = 2, 3
    # Linear ramp: t = m.
    deltas = np.ones((5, H, W), dtype=np.float32)
    correction = fit(
        [Ramp(deltas=deltas)], model=PolynomialModel(order=1)
    )
    # Build a ramp whose cumulative values exceed fit_max.
    extrapRamp = Ramp(
        deltas=np.ones((20, H, W), dtype=np.float32)
    )
    result = apply(correction, extrapRamp)
    # Reads beyond index 4 are extrapolated (m > fit_max of original fit).
    assert result.outOfRangeMask[10:].all()


def test_applyLeavesBadPixelsUntouched(tinyLinearRamp):
    ramp, truth = tinyLinearRamp
    mask = np.zeros(ramp.deltas.shape[1:], dtype=np.uint8)
    mask[0, 0] = 1
    correction = fit(
        [Ramp(deltas=ramp.deltas, validMask=mask)],
        model=PolynomialModel(order=1),
    )
    result = apply(correction, Ramp(deltas=ramp.deltas))
    # Pixel (0, 0) is bad: output equals cumulative of input (unchanged).
    inputCum = np.cumsum(ramp.deltas.astype(np.float32), axis=0)
    np.testing.assert_allclose(
        result.cumulativeLinear[:, 0, 0], inputCum[:, 0, 0]
    )
    # Pixel (0, 1) is good: output is linearized (differs from raw cumulative).
    assert not np.allclose(
        result.cumulativeLinear[:, 0, 1], inputCum[:, 0, 1]
    )


def test_applyFrameMatchesApplyOnSingleRead(smallSyntheticRamp):
    ramp, _ = smallSyntheticRamp
    correction = fit([ramp])
    fullResult = apply(correction, ramp)

    mSingle = np.cumsum(ramp.deltas, axis=0)[-1]  # last read's cumulative
    linFrame, oorFrame = applyFrame(correction, mSingle)

    np.testing.assert_allclose(
        linFrame, fullResult.cumulativeLinear[-1], rtol=1e-6
    )
    np.testing.assert_array_equal(oorFrame, fullResult.outOfRangeMask[-1])


def test_applyRejectsShapeMismatch(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    correction = fit([ramp], model=PolynomialModel(order=1))
    wrongRamp = Ramp(deltas=np.zeros((3, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        apply(correction, wrongRamp)


def test_applyEmptyRampRaises(tinyLinearRamp):
    ramp, _ = tinyLinearRamp
    correction = fit([ramp], model=PolynomialModel(order=1))
    empty = Ramp(
        deltas=np.zeros((0, *ramp.deltas.shape[1:]), dtype=np.float32)
    )
    with pytest.raises(ValueError):
        apply(correction, empty)
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_apply.py -v
```

Expected: `ModuleNotFoundError: No module named 'relin.apply'`.

### - [ ] Step 3: Implement `python/relin/apply.py`

```python
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
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_apply.py -v
```

Expected: all 6 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/apply.py tests/test_apply.py
git commit -m "Add apply() and applyFrame() with bad-pixel pass-through"
```

---

## Task 11: `io.py` — `saveFits` / `loadFits` with checksums

Persist a `LinearityCorrection` to FITS (header-only PRIMARY + image HDUs with `CHECKSUM`/`DATASUM`) and load it back. Model-specific HDUs flow through `model.toFitsHdus` / `Model.fromFitsHdus` via the registry.

**Files:**
- Create: `python/relin/io.py`
- Create: `tests/test_io_fits.py`

### - [ ] Step 1: Write the failing tests

```python
"""Tests for FITS save/load of LinearityCorrection."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from relin.fit import fit
from relin.io import loadFits, saveFits
from relin.models import PolynomialModel
from relin.types import Ramp


def _makeCorrection():
    rng = np.random.default_rng(0)
    H, W = 4, 5
    deltas = np.full((10, H, W), 100.0, dtype=np.float32)
    deltas += rng.normal(0.0, 0.1, size=deltas.shape).astype(np.float32)
    return fit([Ramp(deltas=deltas)], model=PolynomialModel(order=2))


def test_saveLoadRoundTrip(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)

    loaded = loadFits(path)
    np.testing.assert_array_equal(loaded.coefficients, correction.coefficients)
    np.testing.assert_array_equal(loaded.fitMin, correction.fitMin)
    np.testing.assert_array_equal(loaded.fitMax, correction.fitMax)
    np.testing.assert_array_equal(loaded.badPixelMask, correction.badPixelMask)
    np.testing.assert_array_equal(
        loaded.diagnostics.residualRms, correction.diagnostics.residualRms
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.maxAbsResidual, correction.diagnostics.maxAbsResidual
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.nPointsUsed, correction.diagnostics.nPointsUsed
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.monotonic, correction.diagnostics.monotonic
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.conditionNumber, correction.diagnostics.conditionNumber
    )
    assert loaded.model.modelName == "POLYNOMIAL"
    assert loaded.model.order == 2


def test_saveFitsPrimaryIsHeaderOnly(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    with fits.open(path) as hdul:
        assert hdul[0].data is None
        assert hdul[0].header["MODEL"] == "POLYNOMIAL"
        assert "RELINVER" in hdul[0].header


def test_saveFitsImageHdusHaveChecksums(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    with fits.open(path) as hdul:
        for hdu in hdul[1:]:
            assert "CHECKSUM" in hdu.header
            assert "DATASUM" in hdu.header


def test_loadFitsUnknownModelRaises(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    # Corrupt the MODEL keyword.
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["MODEL"] = "NOSUCHMODEL"
        hdul.flush()
    with pytest.raises(ValueError):
        loadFits(path)
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_io_fits.py -v
```

Expected: `ModuleNotFoundError: No module named 'relin.io'`.

### - [ ] Step 3: Implement `python/relin/io.py`

```python
"""FITS persistence for LinearityCorrection objects."""

from __future__ import annotations

import datetime as _dt
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from astropy.io import fits

from relin.models import MODEL_REGISTRY
from relin.types import Diagnostics, LinearityCorrection


def _relinVersion() -> str:
    try:
        return version("relin")
    except PackageNotFoundError:
        return "unknown"


def saveFits(path: str | Path, correction: LinearityCorrection) -> None:
    """Write a ``LinearityCorrection`` to a FITS file."""
    path = Path(path)

    # Build PRIMARY header.
    primaryHeader = fits.Header()
    primaryHeader["MODEL"] = (
        correction.model.modelName, "model form identifier"
    )
    primaryHeader["FITDATE"] = (
        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ISO-8601 fit timestamp",
    )
    primaryHeader["RELINVER"] = (_relinVersion(), "relin package version")
    # Scalar summary fields (numbers only; skip non-scalar entries).
    for key, value in correction.diagnostics.summary.items():
        fitsKey = _toFitsKey(key)
        if isinstance(value, (int, float, bool, str)):
            primaryHeader[fitsKey] = (value, key)
    primary = fits.PrimaryHDU(header=primaryHeader)

    # Model-specific HDUs.
    modelHdus = list(correction.model.toFitsHdus(correction))

    # Standard HDUs for the non-model-specific arrays.
    fitMinHdu = fits.ImageHDU(data=correction.fitMin, name="FITMIN")
    fitMaxHdu = fits.ImageHDU(data=correction.fitMax, name="FITMAX")
    bpHdu = fits.ImageHDU(data=correction.badPixelMask, name="BPMASK")
    bpHdu.header["COMMENT"] = "Bit flags: MASKED_BY_INPUT=0x01 INSUFFICIENT_POINTS=0x02"
    bpHdu.header["COMMENT"] = "          FIT_FAILED=0x04 NON_MONOTONIC=0x08"
    resRmsHdu = fits.ImageHDU(
        data=correction.diagnostics.residualRms, name="RESRMS"
    )
    resMaxHdu = fits.ImageHDU(
        data=correction.diagnostics.maxAbsResidual, name="RESMAX"
    )
    nPtsHdu = fits.ImageHDU(
        data=correction.diagnostics.nPointsUsed, name="NPOINTS"
    )
    monoHdu = fits.ImageHDU(
        data=correction.diagnostics.monotonic.astype(np.uint8), name="MONOTON"
    )
    condHdu = fits.ImageHDU(
        data=correction.diagnostics.conditionNumber, name="CONDNUM"
    )

    hdul = fits.HDUList(
        [primary, *modelHdus, fitMinHdu, fitMaxHdu, bpHdu,
         resRmsHdu, resMaxHdu, nPtsHdu, monoHdu, condHdu]
    )

    # Add CHECKSUM/DATASUM to every image HDU.
    for hdu in hdul[1:]:
        hdu.add_checksum()

    hdul.writeto(path, overwrite=True)


def loadFits(path: str | Path) -> LinearityCorrection:
    """Read a FITS file written by :func:`saveFits`."""
    path = Path(path)
    with fits.open(path) as hdul:
        primary = hdul[0]
        modelName = primary.header["MODEL"]
        if modelName not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {modelName!r}; known: {sorted(MODEL_REGISTRY)}"
            )
        modelClass = MODEL_REGISTRY[modelName]

        # Collect model HDUs (anything the model classmethod consumes) and the
        # fixed non-model HDUs by name.
        allHdus = [hdu for hdu in hdul]
        model, coefficients = modelClass.fromFitsHdus(allHdus)

        fitMin = _arrayByName(hdul, "FITMIN")
        fitMax = _arrayByName(hdul, "FITMAX")
        badPixelMask = _arrayByName(hdul, "BPMASK").astype(np.uint8)
        residualRms = _arrayByName(hdul, "RESRMS")
        maxAbsResidual = _arrayByName(hdul, "RESMAX")
        nPointsUsed = _arrayByName(hdul, "NPOINTS").astype(np.int32)
        monotonic = _arrayByName(hdul, "MONOTON").astype(bool)
        conditionNumber = _arrayByName(hdul, "CONDNUM")

        # Rebuild summary from primary header (best-effort; drops non-scalar keys).
        summary: dict = {}
        for card in primary.header.cards:
            key = card.keyword
            if key in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "MODEL",
                      "FITDATE", "RELINVER") or key.startswith("NAXIS"):
                continue
            if card.comment:  # the "key" field stores the original dict key
                # We store the original key in the comment when saving.
                originalKey = card.comment
                summary[originalKey] = card.value

    diagnostics = Diagnostics(
        residualRms=residualRms,
        maxAbsResidual=maxAbsResidual,
        nPointsUsed=nPointsUsed,
        monotonic=monotonic,
        conditionNumber=conditionNumber,
        summary=summary,
    )
    return LinearityCorrection(
        model=model,
        coefficients=coefficients,
        fitMin=fitMin,
        fitMax=fitMax,
        badPixelMask=badPixelMask,
        diagnostics=diagnostics,
    )


def _arrayByName(hdul: fits.HDUList, name: str) -> np.ndarray:
    for hdu in hdul:
        if getattr(hdu, "name", "") == name:
            return np.asarray(hdu.data)
    raise ValueError(f"HDU {name!r} not found in FITS file")


def _toFitsKey(pythonKey: str) -> str:
    """FITS header keys are <= 8 chars, uppercase, no underscore-leading digits.
    Truncate and uppercase defensively; collisions are avoided by using the
    original key as the comment so ``loadFits`` can reconstruct the dict."""
    return pythonKey.upper()[:8]
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_io_fits.py -v
```

Expected: all 4 tests PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/io.py tests/test_io_fits.py
git commit -m "Add FITS save/load with checksums and model registry dispatch"
```

---

## Task 12: `loaders.py` and `saturation.py` stub

Add the dev convenience `loadNpz` loader for standalone testing, and the empty `saturation.py` module that establishes the future home of saturation detection.

**Files:**
- Create: `python/relin/loaders.py`
- Create: `python/relin/saturation.py`
- Create: `tests/test_loaders.py`

### - [ ] Step 1: Write the failing tests

```python
"""Tests for loaders.loadNpz."""

from __future__ import annotations

import numpy as np

from relin.loaders import loadNpz
from relin.types import Ramp


def test_loadNpzReturnsRampAndPhotodiode(tmp_path):
    N, H, W = 5, 3, 4
    deltas = np.arange(N * H * W, dtype=np.float32).reshape(N, H, W)
    photodiode = np.array([1.0, 1.1, 1.05, 1.0, 0.95], dtype=np.float64)
    path = tmp_path / "ramp.npz"
    np.savez(path, deltas=deltas, photodiode=photodiode)

    ramp, pdio = loadNpz(path)
    assert isinstance(ramp, Ramp)
    np.testing.assert_array_equal(ramp.deltas, deltas)
    assert ramp.validMask is None
    np.testing.assert_array_equal(pdio, photodiode)
```

Also add a stub test for saturation:

```python
# Add to tests/test_loaders.py

def test_saturationModuleIsImportable():
    import relin.saturation  # noqa: F401
```

### - [ ] Step 2: Run tests, verify they fail

```bash
uv run pytest tests/test_loaders.py -v
```

Expected: `ModuleNotFoundError` for both `relin.loaders` and `relin.saturation`.

### - [ ] Step 3: Implement `python/relin/loaders.py`

```python
"""Development convenience loaders. Production callers supply their own loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from relin.types import Ramp


def loadNpz(path: str | Path) -> tuple[Ramp, np.ndarray]:
    """Load a ``.npz`` with ``deltas`` and ``photodiode`` arrays.

    Returns a ``(Ramp, photodiode)`` pair. The caller is expected to apply
    the photodiode correction (e.g. ``deltas * photodiode[:, None, None]``)
    before passing the ramp into :func:`relin.fit.fit`.
    """
    path = Path(path)
    with np.load(path) as data:
        deltas = np.asarray(data["deltas"], dtype=np.float32)
        photodiode = np.asarray(data["photodiode"])
    return Ramp(deltas=deltas), photodiode
```

### - [ ] Step 4: Implement the stub `python/relin/saturation.py`

```python
"""Saturation-detection utility — planned post-MVP.

The core ``fit`` path accepts a pixel-level ``validMask`` on each ``Ramp``.
This module will eventually provide a default saturation detector that
produces such a mask from a ramp's raw deltas / cumulative signal. For the
MVP, it intentionally contains no implementation.

See ``docs/superpowers/specs/2026-04-16-relin-package-design.md``
section 1 ("Planned (post-MVP) extensions") and section 11.
"""

from __future__ import annotations

# Intentionally empty.
```

### - [ ] Step 5: Run tests, verify they pass

```bash
uv run pytest tests/test_loaders.py -v
```

Expected: both tests PASS.

### - [ ] Step 6: Commit

```bash
git add python/relin/loaders.py python/relin/saturation.py tests/test_loaders.py
git commit -m "Add loadNpz dev loader and saturation.py stub"
```

---

## Task 13: Public `__init__.py` re-exports

Expose the whole public API from `relin`.

**Files:**
- Modify: `python/relin/__init__.py`
- Modify: `tests/test_smoke.py`

### - [ ] Step 1: Append failing test to `tests/test_smoke.py`

```python
# Append to tests/test_smoke.py

def test_publicApiExports():
    import relin

    for attr in [
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
    ]:
        assert hasattr(relin, attr), f"relin.{attr} missing"
```

### - [ ] Step 2: Run test, verify it fails

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: the new test fails on the first missing attribute.

### - [ ] Step 3: Implement `python/relin/__init__.py`

```python
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
```

### - [ ] Step 4: Run tests, verify they pass

```bash
uv run pytest tests/test_smoke.py -v
uv run pytest -v  # full suite
```

Expected: all tests across the suite PASS.

### - [ ] Step 5: Commit

```bash
git add python/relin/__init__.py tests/test_smoke.py
git commit -m "Add public API re-exports from relin/__init__.py"
```

---

## Task 14: End-to-end integration test

A single test that exercises fit → save → load → apply on a synthetic dataset with a tiled block size (so the tiling path is explicitly exercised end-to-end), using the public API only.

**Files:**
- Create: `tests/test_integration.py`

### - [ ] Step 1: Write the test

```python
"""End-to-end integration test: fit -> save -> load -> apply."""

from __future__ import annotations

import numpy as np

import relin


def test_integrationEndToEnd(smallSyntheticRamp, tmp_path):
    ramp, truth = smallSyntheticRamp

    # Fit with a small block size to exercise the tiling path.
    correction = relin.fit([ramp], blockSize=(2, 3))

    # Save + load round trip.
    path = tmp_path / "correction.fits"
    relin.saveFits(path, correction)
    loaded = relin.loadFits(path)

    # Apply loaded correction to the original ramp.
    result = relin.apply(loaded, ramp)

    # cumulativeLinear should match the true target at every read, up to the
    # precision of the fit.
    expected = np.broadcast_to(
        truth["target"][:, None, None], ramp.deltas.shape
    )
    residual = result.cumulativeLinear - expected
    # Residuals should be small for every pixel.
    rms = np.sqrt(np.mean(residual ** 2, axis=0))
    assert rms.max() < 1.0, f"max per-pixel RMS {rms.max()} too large"

    # No bad pixels, no out-of-range flags on the fitting data itself.
    assert (loaded.badPixelMask == 0).all()
    assert not result.outOfRangeMask.any()

    # Summary is populated and sane.
    summary = loaded.diagnostics.summary
    assert summary["MODELNAME"] == "POLYNOMIAL" or \
           summary.get("modelName") == "POLYNOMIAL"
    # Good-pixel fraction is 1.0 for this synthetic dataset.
    # (Key name is uppercased in the FITS header round-trip.)
    goodKeys = [k for k in summary if "GOOD" in k.upper() or "good" in k]
    assert goodKeys
    for k in goodKeys:
        assert summary[k] == 1.0 or summary[k] == "1.0"
```

### - [ ] Step 2: Run the test

```bash
uv run pytest tests/test_integration.py -v
```

Expected: PASS.

### - [ ] Step 3: Run the full suite as a final sanity check

```bash
uv run pytest -v
```

Expected: every test PASSes. If anything fails, fix it before committing.

### - [ ] Step 4: Commit

```bash
git add tests/test_integration.py
git commit -m "Add end-to-end integration test"
```

---

## Post-plan checklist

After completing all 14 tasks:

- [ ] `uv run pytest -v` reports all tests passing.
- [ ] `git log --oneline` shows one commit per task (at least 14 commits beyond the spec commit).
- [ ] Every spec section has a corresponding implementation (cross-check against `docs/superpowers/specs/2026-04-16-relin-package-design.md`).
- [ ] No placeholder files, no `TODO` comments in shipped code except the intentional stub in `saturation.py`.
