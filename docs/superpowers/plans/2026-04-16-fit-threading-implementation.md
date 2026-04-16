# `fit()` Tile Threading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `workers` parameter to `relin.fit()` that runs the tile loop on a `concurrent.futures.ThreadPoolExecutor`. Output must be byte-identical regardless of worker count.

**Architecture:** Keep the existing sequential tile loop unchanged for `workers == 1` (fast path). For `workers > 1`, submit each tile's `model.fitBlock` call as a future; consume completed futures on the main thread and stitch results into disjoint output slices. Tile assembly happens on the submitting thread; workers receive fully-formed numpy arrays and do pure compute. A module-level `_executorFactory` pointer allows tests to observe executor construction without globally patching `concurrent.futures`.

**Tech Stack:** Python 3.12, `concurrent.futures.ThreadPoolExecutor`, `os.cpu_count()`, numpy. No new runtime dependencies.

**Design spec:** `docs/superpowers/specs/2026-04-16-fit-threading-design.md`

---

## File layout

| File | Role |
|------|------|
| `python/relin/fit.py` | Modified: add `_resolveWorkerCount`, `_SMALL_FRAME_PIXEL_LIMIT`, `_executorFactory`, `workers` param, threaded branch, tile-coord error wrapping. |
| `tests/test_fit_threading.py` | New: all tests for the heuristic, the threaded path, byte-identical output, and error propagation. |
| `examples/benchmark_fit_threading.py` | New: standalone benchmark script. Not a pytest test. |

The sequential loop in `fit()` stays byte-identical for `workers == 1`. No other production file is modified.

---

## Task 1: Add `_resolveWorkerCount` helper

Extract the worker-count resolution heuristic as a pure function so it can be unit-tested without running a real fit. This task adds the helper and its tests **only** — no integration into `fit()` yet.

**Files:**
- Modify: `python/relin/fit.py` (add module-level `_SMALL_FRAME_PIXEL_LIMIT`, `_DEFAULT_WORKER_CAP`, and `_resolveWorkerCount`).
- Create: `tests/test_fit_threading.py`.

### - [ ] Step 1: Write the failing tests

Create `tests/test_fit_threading.py` with this exact content:

```python
"""Tests for fit() threading: heuristic, parallel path, errors, determinism."""

from __future__ import annotations

import pytest

from relin.fit import _resolveWorkerCount


def test_resolveWorkerCountExplicitIntIsReturnedAsIs():
    # Explicit wins over heuristic — no clamping, no size check.
    assert _resolveWorkerCount(1, 10, 10) == 1
    assert _resolveWorkerCount(4, 10, 10) == 4
    assert _resolveWorkerCount(16, 10, 10) == 16
    # Even on a "large" frame, explicit 1 is honored.
    assert _resolveWorkerCount(1, 5000, 5000) == 1


def test_resolveWorkerCountSmallFrameDefaultsToOne():
    # With H*W < _SMALL_FRAME_PIXEL_LIMIT (1_000_000), None → 1 worker
    # regardless of os.cpu_count().
    assert _resolveWorkerCount(None, 100, 100) == 1
    assert _resolveWorkerCount(None, 1000, 999) == 1  # 999_000 < 1_000_000


def test_resolveWorkerCountLargeFrameCapsAtEight(monkeypatch):
    # When H*W >= 1_000_000 and os.cpu_count() > 8, cap at 8.
    import relin.fit as fitModule
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: 16)
    assert _resolveWorkerCount(None, 2000, 500) == 8  # 1_000_000 exactly
    assert _resolveWorkerCount(None, 4096, 4096) == 8


def test_resolveWorkerCountLargeFrameUncappedBelowEight(monkeypatch):
    # When H*W is large but os.cpu_count() < 8, use cpu_count.
    import relin.fit as fitModule
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: 4)
    assert _resolveWorkerCount(None, 4096, 4096) == 4


def test_resolveWorkerCountHandlesNoneCpuCount(monkeypatch):
    # os.cpu_count() can return None on some platforms; fall back to 1.
    import relin.fit as fitModule
    monkeypatch.setattr(fitModule.os, "cpu_count", lambda: None)
    assert _resolveWorkerCount(None, 4096, 4096) == 1


def test_resolveWorkerCountInvalidRaises():
    with pytest.raises(ValueError, match="workers"):
        _resolveWorkerCount(0, 10, 10)
    with pytest.raises(ValueError, match="workers"):
        _resolveWorkerCount(-3, 10, 10)
```

### - [ ] Step 2: Run tests, verify they fail

```
uv run pytest tests/test_fit_threading.py -v
```

Expected: `ImportError` / `AttributeError` — `_resolveWorkerCount` is not defined yet.

### - [ ] Step 3: Add the helper and module constants

In `python/relin/fit.py`, add the `os` import alongside existing imports, and add a new block of module-level definitions directly after the imports (before `def fit(...)`):

```python
import os
```

Then, after the existing imports and before `def fit(`:

```python
# Worker-count resolution constants. Tunable at module level; the tests
# monkeypatch `os.cpu_count` rather than these, so changing them does not
# break tests but will change the default behavior for small/large frames.
_SMALL_FRAME_PIXEL_LIMIT = 1_000_000   # H*W below this → sequential default
_DEFAULT_WORKER_CAP = 8                # auto-detected cpu_count is capped here

# Override point for tests. Default is the real ThreadPoolExecutor; a test
# can `monkeypatch.setattr("relin.fit._executorFactory", ...)` to observe
# construction or to inject a recording executor.
from concurrent.futures import ThreadPoolExecutor
_executorFactory = ThreadPoolExecutor


def _resolveWorkerCount(workers: int | None, H: int, W: int) -> int:
    """Resolve the effective worker count for a `fit()` call.

    - If ``workers`` is an ``int``: returned as-is; must be >= 1.
    - If ``workers`` is ``None``:
        - H*W < ``_SMALL_FRAME_PIXEL_LIMIT`` → 1 (sequential default).
        - Otherwise → ``min(os.cpu_count() or 1, _DEFAULT_WORKER_CAP)``.

    Raises:
        ValueError: if ``workers`` is an int less than 1.
    """
    if workers is None:
        if H * W < _SMALL_FRAME_PIXEL_LIMIT:
            return 1
        return min(os.cpu_count() or 1, _DEFAULT_WORKER_CAP)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return workers
```

### - [ ] Step 4: Run tests, verify they pass

```
uv run pytest tests/test_fit_threading.py -v
```

Expected: 6 tests PASS.

Then run the full suite to verify nothing else broke:

```
uv run pytest -q
```

Expected: 60 passed (was 54; added 6 tests this task).

### - [ ] Step 5: Commit

```
git add python/relin/fit.py tests/test_fit_threading.py
git commit -m "Add _resolveWorkerCount helper and worker-count constants"
```

---

## Task 2: Thread the tile loop

Add the `workers` parameter to `fit()`, route between the existing sequential tile loop (unchanged) and a new threaded branch. Verify byte-identical output.

**Files:**
- Modify: `python/relin/fit.py` (add `workers` param, threaded branch, imports).
- Modify: `tests/test_fit_threading.py` (append integration tests).

### - [ ] Step 1: Append the failing integration tests

Append these to the end of `tests/test_fit_threading.py`:

```python
import numpy as np

from relin.fit import fit
from relin.models import PolynomialModel
from relin.types import Ramp


def _arraysEqual(a, b, label):
    """np.array_equal with a helpful assertion message."""
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{label}: dtype {a.dtype} != {b.dtype}"
    assert np.array_equal(a, b), f"{label}: arrays differ"


def test_fitWorkers4ProducesByteIdenticalOutputToWorkers1(smallSyntheticRamp):
    """The threaded path must produce output byte-identical to the sequential
    path, because each tile writes to disjoint output slices on the main
    thread and the tile-assembly and fit arithmetic are deterministic."""
    ramp, _ = smallSyntheticRamp
    # blockSize=(2, 3) on a 4x5 frame produces 4 tiles — enough to exercise
    # parallelism when workers=4.
    serial = fit([ramp], blockSize=(2, 3), workers=1)
    threaded = fit([ramp], blockSize=(2, 3), workers=4)

    _arraysEqual(threaded.coefficients, serial.coefficients, "coefficients")
    _arraysEqual(threaded.fitMin, serial.fitMin, "fitMin")
    _arraysEqual(threaded.fitMax, serial.fitMax, "fitMax")
    _arraysEqual(threaded.badPixelMask, serial.badPixelMask, "badPixelMask")
    _arraysEqual(
        threaded.diagnostics.residualRms,
        serial.diagnostics.residualRms,
        "residualRms",
    )
    _arraysEqual(
        threaded.diagnostics.maxAbsResidual,
        serial.diagnostics.maxAbsResidual,
        "maxAbsResidual",
    )
    _arraysEqual(
        threaded.diagnostics.nPointsUsed,
        serial.diagnostics.nPointsUsed,
        "nPointsUsed",
    )
    _arraysEqual(
        threaded.diagnostics.monotonic,
        serial.diagnostics.monotonic,
        "monotonic",
    )
    _arraysEqual(
        threaded.diagnostics.conditionNumber,
        serial.diagnostics.conditionNumber,
        "conditionNumber",
    )
    # Summary dicts must be equal (same float values, same keys).
    assert threaded.diagnostics.summary == serial.diagnostics.summary


def test_fitWorkers1DoesNotConstructExecutor(monkeypatch, smallSyntheticRamp):
    """The sequential fast path must not touch the executor factory at all."""
    import relin.fit as fitModule

    def _fail(*args, **kwargs):
        raise AssertionError(
            f"_executorFactory was called for workers=1 path: "
            f"args={args} kwargs={kwargs}"
        )

    monkeypatch.setattr(fitModule, "_executorFactory", _fail)
    ramp, _ = smallSyntheticRamp
    fit([ramp], blockSize=(2, 3), workers=1)  # must not raise


def test_fitWorkers4ConstructsExecutorWithMaxWorkers4(
    monkeypatch, smallSyntheticRamp
):
    """When workers=4, the factory must be called with max_workers=4."""
    import relin.fit as fitModule
    from concurrent.futures import ThreadPoolExecutor

    recorded = {}

    def _recordingFactory(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = dict(kwargs)
        # Return the real executor so the fit still runs to completion.
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr(fitModule, "_executorFactory", _recordingFactory)
    ramp, _ = smallSyntheticRamp
    fit([ramp], blockSize=(2, 3), workers=4)
    assert recorded["kwargs"].get("max_workers") == 4


def test_fitAutoWorkersUsesResolvedCount(monkeypatch, smallSyntheticRamp):
    """With workers=None and a small frame, resolved count is 1 (no executor
    call). This mirrors test_fitWorkers1DoesNotConstructExecutor but via the
    default code path rather than explicit workers=1."""
    import relin.fit as fitModule

    called = []
    monkeypatch.setattr(
        fitModule, "_executorFactory", lambda *a, **k: called.append(1)
    )
    ramp, _ = smallSyntheticRamp  # H=4 W=5, well below 1_000_000.
    fit([ramp], blockSize=(2, 3))  # workers=None
    assert called == []


def test_fitInvalidWorkersRaises(smallSyntheticRamp):
    ramp, _ = smallSyntheticRamp
    with pytest.raises(ValueError, match="workers"):
        fit([ramp], workers=0)
    with pytest.raises(ValueError, match="workers"):
        fit([ramp], workers=-1)
```

### - [ ] Step 2: Run the tests, verify they fail

```
uv run pytest tests/test_fit_threading.py -v
```

Expected: the new tests fail. Typical failures:

- `TypeError: fit() got an unexpected keyword argument 'workers'` for the explicit-workers tests.
- The "does not construct executor" test passes incidentally (because nothing constructs the executor yet) — that's fine; it pins down the sequential path's behavior now and will still hold after the threaded branch is added.
- The "constructs executor" test fails because the branch doesn't exist.

### - [ ] Step 3: Implement the threaded branch

In `python/relin/fit.py`, modify `def fit(...)` to accept `workers` and dispatch to either the existing sequential loop or a new threaded loop. Add the necessary imports.

At the top of `python/relin/fit.py`, immediately after the existing `from concurrent.futures import ThreadPoolExecutor` line (added in Task 1), also add:

```python
from concurrent.futures import as_completed
```

(Alternatively you can change the existing import to
`from concurrent.futures import ThreadPoolExecutor, as_completed`. Either is fine.)

Change the `fit(...)` signature by inserting `workers` between `blockSize` and `conditionNumberLimit`:

```python
def fit(
    ramps: Sequence[Ramp],
    model: Model | None = None,
    blockSize: tuple[int, int] = (512, 512),
    workers: int | None = None,
    conditionNumberLimit: float = 1e12,
) -> LinearityCorrection:
```

Right after the existing validation loop that checks ramp shapes (the one that ends with the `ramp.validMask shape ... != (H, W)` check), but **before** the "Per-ramp precomputation" section, insert:

```python
    effectiveWorkers = _resolveWorkerCount(workers, H, W)
```

Now the tile loop needs to branch. Find the existing code starting with:

```python
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
            ...
```

**Do not delete the sequential loop.** Instead, refactor the per-tile assembly into a local helper, then branch between the sequential stitch loop (unchanged semantics) and a threaded variant.

Replace the entire tile-loop section (from `# Iterate over tiles.` down to the `badPixelMask[rowStart:rowEnd, colStart:colEnd] = result.badPixelMask` inclusive) with the following block:

```python
    # Iterate over tiles. Tile-assembly (mTile, validTile) is identical
    # for sequential and threaded paths; factor it into a closure so both
    # paths call model.fitBlock with exactly the same inputs.
    bH, bW = blockSize

    def _assembleTile(
        rowStart: int, rowEnd: int, colStart: int, colEnd: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tileH = rowEnd - rowStart
        tileW = colEnd - colStart
        mSegments: list[np.ndarray] = []
        validSegments: list[np.ndarray] = []
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
        return mTile, validTile

    def _storeResult(
        rowStart: int, rowEnd: int, colStart: int, colEnd: int, result
    ) -> None:
        coefficients[:, rowStart:rowEnd, colStart:colEnd] = result.coefficients
        fitMin[rowStart:rowEnd, colStart:colEnd] = result.fitMin
        fitMax[rowStart:rowEnd, colStart:colEnd] = result.fitMax
        residualRms[rowStart:rowEnd, colStart:colEnd] = result.residualRms
        maxAbsResidual[rowStart:rowEnd, colStart:colEnd] = result.maxAbsResidual
        nPointsUsed[rowStart:rowEnd, colStart:colEnd] = result.nPointsUsed
        conditionNumber[rowStart:rowEnd, colStart:colEnd] = result.conditionNumber
        monotonic[rowStart:rowEnd, colStart:colEnd] = result.monotonic
        badPixelMask[rowStart:rowEnd, colStart:colEnd] = result.badPixelMask

    if effectiveWorkers == 1:
        # Sequential fast path — no executor involvement.
        for rowStart in range(0, H, bH):
            rowEnd = min(rowStart + bH, H)
            for colStart in range(0, W, bW):
                colEnd = min(colStart + bW, W)
                mTile, validTile = _assembleTile(
                    rowStart, rowEnd, colStart, colEnd
                )
                result = model.fitBlock(
                    m=mTile, t=tConcat, valid=validTile,
                    conditionNumberLimit=conditionNumberLimit,
                )
                _storeResult(rowStart, rowEnd, colStart, colEnd, result)
    else:
        # Threaded path. Submit each tile as a future; consume completed
        # futures on the main thread and stitch into disjoint output slices.
        # Tile-assembly runs on the submitting thread so workers do pure
        # compute and memory is bounded to `effectiveWorkers` in-flight
        # tiles worth of m/valid arrays.
        with _executorFactory(max_workers=effectiveWorkers) as executor:
            futures: dict = {}
            for rowStart in range(0, H, bH):
                rowEnd = min(rowStart + bH, H)
                for colStart in range(0, W, bW):
                    colEnd = min(colStart + bW, W)
                    mTile, validTile = _assembleTile(
                        rowStart, rowEnd, colStart, colEnd
                    )
                    fut = executor.submit(
                        model.fitBlock,
                        m=mTile, t=tConcat, valid=validTile,
                        conditionNumberLimit=conditionNumberLimit,
                    )
                    futures[fut] = (rowStart, rowEnd, colStart, colEnd)

            for fut in as_completed(futures):
                rs, re, cs, ce = futures[fut]
                result = fut.result()
                _storeResult(rs, re, cs, ce, result)
```

Note: this step does **not** yet wrap exceptions with tile coords — that is Task 3. The bare `fut.result()` call will propagate the original exception unchanged if a tile raises, which is acceptable as an intermediate state.

### - [ ] Step 4: Run tests, verify they pass

```
uv run pytest tests/test_fit_threading.py -v
```

Expected: all 11 tests PASS (6 from Task 1 + 5 added here).

Then run the full suite:

```
uv run pytest -q
```

Expected: 65 passed.

### - [ ] Step 5: Commit

```
git add python/relin/fit.py tests/test_fit_threading.py
git commit -m "Thread fit() tile loop via ThreadPoolExecutor"
```

---

## Task 3: Wrap worker exceptions with tile coordinates

If a worker thread raises, replace the generic propagation with a `RuntimeError` whose message identifies the offending tile's row/col range. Preserve the original exception via `__cause__` (chained exception).

**Files:**
- Modify: `python/relin/fit.py` (wrap `fut.result()` with `try/except`).
- Modify: `tests/test_fit_threading.py` (append error-propagation test).

### - [ ] Step 1: Append the failing test

Append to the end of `tests/test_fit_threading.py`:

```python
import threading


def test_fitWorkerExceptionIncludesTileCoords(smallSyntheticRamp):
    """If any fitBlock call raises on a worker thread, the exception must
    be re-raised as a RuntimeError whose message identifies the offending
    tile's row/col slice and whose __cause__ is the original exception."""
    ramp, _ = smallSyntheticRamp
    pm = PolynomialModel(order=2)
    originalFitBlock = pm.fitBlock
    failedOnce = threading.Event()

    def failingFitBlock(m, t, valid, conditionNumberLimit):
        # Let the first real (non-peek) tile raise; all subsequent calls
        # succeed. The peek in _peekCoefShape does not go through here for
        # PolynomialModel (it uses the isinstance fast path), so we don't
        # need a shape guard.
        if not failedOnce.is_set():
            failedOnce.set()
            raise RuntimeError("injected failure")
        return originalFitBlock(
            m=m, t=t, valid=valid, conditionNumberLimit=conditionNumberLimit
        )

    # PolynomialModel is a frozen dataclass; bypass the frozen __setattr__
    # to shadow the bound method with an instance attribute.
    object.__setattr__(pm, "fitBlock", failingFitBlock)

    with pytest.raises(
        RuntimeError,
        match=r"fitBlock failed at tile \[rows \d+:\d+, cols \d+:\d+\]",
    ) as excInfo:
        fit([ramp], model=pm, blockSize=(2, 3), workers=2)
    # __cause__ carries the original exception.
    cause = excInfo.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "injected failure"
```

### - [ ] Step 2: Run the test, verify it fails

```
uv run pytest tests/test_fit_threading.py::test_fitWorkerExceptionIncludesTileCoords -v
```

Expected: FAIL. The raw `RuntimeError("injected failure")` propagates without being wrapped, so the `match=` on the message fails and/or `__cause__` is `None`.

### - [ ] Step 3: Wrap future results with a tile-coord-aware RuntimeError

In `python/relin/fit.py`, find the consumer loop added in Task 2:

```python
            for fut in as_completed(futures):
                rs, re, cs, ce = futures[fut]
                result = fut.result()
                _storeResult(rs, re, cs, ce, result)
```

Replace it with:

```python
            for fut in as_completed(futures):
                rs, re, cs, ce = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    # Cancel any futures that haven't started; in-flight
                    # tasks still run to completion but their results are
                    # discarded when the `with` block shuts down.
                    for other in futures:
                        if other is not fut:
                            other.cancel()
                    raise RuntimeError(
                        f"fitBlock failed at tile "
                        f"[rows {rs}:{re}, cols {cs}:{ce}]"
                    ) from e
                _storeResult(rs, re, cs, ce, result)
```

### - [ ] Step 4: Run tests, verify they pass

```
uv run pytest tests/test_fit_threading.py -v
```

Expected: all 12 tests PASS (11 from earlier + this one).

```
uv run pytest -q
```

Expected: 66 passed.

### - [ ] Step 5: Commit

```
git add python/relin/fit.py tests/test_fit_threading.py
git commit -m "Wrap worker exceptions in fit() with tile-coord RuntimeError"
```

---

## Task 4: Update `fit()` docstring

Document the new `workers` parameter, the BLAS-oversubscription note, and the determinism guarantee. No behavior change, no new tests.

**Files:**
- Modify: `python/relin/fit.py` (docstring only).

### - [ ] Step 1: Replace the `fit()` docstring

Find the current one-line docstring in `python/relin/fit.py`:

```python
    """Fit a per-pixel nonlinearity correction from one or more ramps.

    See ``docs/superpowers/specs/2026-04-16-relin-package-design.md`` for the
    full algorithm description.
    """
```

Replace it with:

```python
    """Fit a per-pixel nonlinearity correction from one or more ramps.

    See ``docs/superpowers/specs/2026-04-16-relin-package-design.md`` for
    the full algorithm description.

    Parameters
    ----------
    ramps : sequence of Ramp
        One or more ramps to fit jointly. All ramps must share the same
        ``(H, W)`` frame shape.
    model : Model, optional
        Model to fit. Defaults to ``PolynomialModel(order=4)``.
    blockSize : (int, int), optional
        Tile size in pixels for the per-tile normal-equations fit.
        Default is ``(512, 512)``. Smaller tiles reduce peak memory;
        larger tiles reduce per-tile overhead.
    workers : int or None, optional
        Number of worker threads for the tile loop.

        - ``1`` (explicit): sequential — no thread pool is constructed.
        - ``N > 1`` (explicit): run the tile loop on a
          ``ThreadPoolExecutor`` with ``max_workers=N``. No upper cap is
          applied to explicit values.
        - ``None`` (default): heuristic. If ``H * W < 1_000_000``, use
          ``1`` worker (sequential). Otherwise use
          ``min(os.cpu_count() or 1, 8)``. The cap at 8 applies only
          to the auto-detected default.

        Output is deterministic and worker-count-independent: every tile
        writes to a disjoint slice of the preallocated output arrays
        on the main thread, so the fit result is byte-identical
        regardless of ``workers``.

        Note on BLAS/LAPACK: numpy's linear-algebra routines may spawn
        additional internal threads. On multi-core machines, combining
        ``workers > 1`` with an uncontrolled BLAS thread count can lead
        to oversubscription and diminishing returns. If threaded
        speedup plateaus below expectations, try setting
        ``OMP_NUM_THREADS=1`` and/or ``MKL_NUM_THREADS=1`` in the
        process environment.
    conditionNumberLimit : float, optional
        Pixels whose normal-equations matrix has a condition number
        above this threshold are flagged as ``FIT_FAILED`` and left
        with zeroed coefficients. Default ``1e12``.

    Returns
    -------
    LinearityCorrection
        Fitted coefficients, range bounds, bad-pixel mask, and
        diagnostics.
    """
```

### - [ ] Step 2: Verify the tests still pass

```
uv run pytest -q
```

Expected: 66 passed (no change from Task 3).

### - [ ] Step 3: Commit

```
git add python/relin/fit.py
git commit -m "Document workers parameter on fit()"
```

---

## Task 5: Benchmark script

Standalone script (not a test). Runs the real 4096² lab ramp through `fit()` with a range of worker counts and prints a small timing table. Informs the later decision on whether `threadpoolctl` is worth adding.

**Files:**
- Create: `examples/benchmark_fit_threading.py`.

### - [ ] Step 1: Create the script

Create `examples/benchmark_fit_threading.py` with exactly this content:

```python
"""Benchmark: fit() wall-clock across worker counts.

Reads the example lab ramp, applies the standard illumination-drift
photodiode correction, then runs `relin.fit` for several worker counts
and prints a timing table. Not a test — no pass/fail criteria.

Usage:
    uv run python examples/benchmark_fit_threading.py

The dataset is not in the repo (see .gitignore); if it's missing, the
script prints a pointer and exits cleanly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import relin
from relin.loaders import loadNpz
from relin.types import Ramp


def main() -> int:
    dataPath = Path("examples/linearity/18734/18734_164220.npz")
    if not dataPath.exists():
        print(
            f"Data file missing: {dataPath}\n"
            "Place the lab NPZ at that path (see .gitignore)."
        )
        return 1

    print(f"Loading {dataPath} ...", flush=True)
    ramp, photodiode = loadNpz(dataPath)
    scale = (photodiode[0] / photodiode).astype(np.float32)
    correctedDeltas = ramp.deltas * scale[:, None, None]
    correctedRamp = Ramp(deltas=correctedDeltas)
    print(
        f"  shape={correctedDeltas.shape} dtype={correctedDeltas.dtype}",
        flush=True,
    )

    workerCounts = [1, 2, 4, 8]
    results = []

    # Warm-up call so the first measurement isn't paying first-touch costs
    # on memory allocations / imported BLAS libraries.
    print("Warm-up run (workers=1) ...", flush=True)
    _ = relin.fit([correctedRamp], blockSize=(512, 512), workers=1)

    for w in workerCounts:
        t0 = time.perf_counter()
        correction = relin.fit(
            [correctedRamp], blockSize=(512, 512), workers=w
        )
        elapsed = time.perf_counter() - t0
        goodFrac = correction.diagnostics.summary["goodPixelFraction"]
        results.append((w, elapsed, goodFrac))
        print(
            f"  workers={w:>2d}  elapsed={elapsed:7.2f}s  "
            f"goodPixelFraction={goodFrac:.4f}",
            flush=True,
        )

    baseline = results[0][1]
    print("\nSummary (speedup vs workers=1):")
    print(f"  {'workers':>8s}  {'elapsed (s)':>12s}  {'speedup':>8s}")
    for w, elapsed, _ in results:
        speedup = baseline / elapsed if elapsed > 0 else float("nan")
        print(f"  {w:>8d}  {elapsed:>12.2f}  {speedup:>7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### - [ ] Step 2: Verify the script imports cleanly

```
uv run python -c "import ast; ast.parse(open('examples/benchmark_fit_threading.py').read())"
```

Expected: no output (syntax is valid).

Also verify it fails gracefully when data is absent (or, if the real data is present in your checkout, runs end-to-end):

```
uv run python examples/benchmark_fit_threading.py
```

Expected: either a full benchmark table, or — if the `.npz` file is missing on this machine — the "Data file missing" message and exit code 1. Both outcomes are acceptable; the point is that the script is valid Python, doesn't crash on import, and shows where to get the data.

### - [ ] Step 3: Commit

```
git add examples/benchmark_fit_threading.py
git commit -m "Add benchmark script for fit() threading"
```

---

## Post-plan checklist

After all 5 tasks are complete:

- [ ] `uv run pytest -q` reports 66 passed.
- [ ] `uv run pytest -q -W error::astropy.io.fits.verify.VerifyWarning` reports 66 passed (unrelated to threading, but confirms no regressions in io.py).
- [ ] `git log --oneline` shows at least 5 new commits on `feat/fit-threading` beyond the spec commit (`368f6b8`).
- [ ] Every section of the design spec maps to at least one task (self-check against `docs/superpowers/specs/2026-04-16-fit-threading-design.md`).
- [ ] No new runtime dependencies in `pyproject.toml`.
- [ ] Output of `relin.fit(ramp, workers=None)` is unchanged on small fixtures (sequential path preserved byte-for-byte).
