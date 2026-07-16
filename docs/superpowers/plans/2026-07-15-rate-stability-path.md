# Rate-stability Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `sanityCheck` driver to `fitLinearity`, then add an opt-in `--rate-stability` path that runs the PIPE2D-1844 split-half `detectRateInstability` gate on the linearized ramp, folds `RATE_UNSTABLE` into the correction's bad-pixel mask, and re-saves the FITS.

**Architecture:** A pure helper `_foldRateStability` runs the gate on the linearized cumulative ramp and returns the folded mask — unit-testable on tiny in-memory arrays. An I/O wrapper `_applyRateStability` calls it, reports, writes provenance, and re-saves; `runFit` calls the wrapper only when the flag is set. Three new `SanityCheckConfig` fields (defaulted) and three `bin/` args drive it.

**Tech Stack:** Python 3.12, numpy. Upstream `lsst.obs.pfs.h4Linearity` (`fit`, `apply`, `saveFits`, `types.RATE_UNSTABLE`, `rateStability.detectRateInstability`) via EUPS.

## Global Constraints

- **camelCase** for function/method/variable names, tests included. Classes PascalCase, constants UPPER_SNAKE_CASE. Do not "correct" existing camelCase to snake_case.
- Comments and docstrings describe what the code does **now**, never its history. No "was", "renamed from", "used to be sanityCheck", "moved from".
- Package source under `python/`, executables in `bin/`, tests at `tests/`.
- Line length 110 (ruff). Rules `["E","F","W","I"]`; pep8-naming (`N`) stays off. Ruff isort may not know `fitLinearity` is first-party (I001); if so, `ruff check --fix` — import grouping only.
- Every shell that runs tests or `bin/` scripts must first run, in the SAME Bash call:
  ```bash
  source /work/stack/loadLSST.bash && setup pfs_pipe2d && setup -j -r /work/cloomis/claude/PIPE2D-1844/obs_pfs && setup -j -r /work/cloomis/claude/PIPE2D-1844/drp_stella
  ```
  Use the LSST-env `python`/`pytest` directly. **Never `uv run`** — it can't see the LSST conda site-packages.
- **Known-red baseline:** the suite is **9 failed, 63 passed** before this work, and has been since before this branch (an upstream `(N,H,W)`→`(H,W,N)` transpose; failures in `tests/test_apply.py`, `tests/test_integration.py`, Chebyshev tests in `tests/test_polynomial_model.py`). DO NOT try to fix them. The gate everywhere in this plan is: **no new failures beyond those 9, and every test this plan adds passes.**
- `RATE_UNSTABLE = 0x0800`, imported from `lsst.obs.pfs.h4Linearity.types`.
- Function/class names `runFit`, `runPlot`, `cliTag`, `SanityCheckConfig` are retained unchanged — only the *file* is renamed. Do not rename the symbols.

---

### Task 1: Rename sanityCheck → fitLinearity

Pure rename: module, `bin/` wrapper, test file, and doc references. No behavior change.

**Files:**
- Rename: `python/fitLinearity/sanityCheck.py` → `python/fitLinearity/fitLinearity.py`
- Rename: `bin/sanityCheck.py` → `bin/fitLinearity.py`
- Rename: `tests/test_sanityCheck.py` → `tests/test_fitLinearity.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: importable `fitLinearity.fitLinearity` exposing `SanityCheckConfig`, `cliTag`, `runFit`, `runPlot`; executable `bin/fitLinearity.py`.

- [ ] **Step 1: git mv the three files**

```bash
git mv python/fitLinearity/sanityCheck.py python/fitLinearity/fitLinearity.py
git mv bin/sanityCheck.py bin/fitLinearity.py
git mv tests/test_sanityCheck.py tests/test_fitLinearity.py
```

- [ ] **Step 2: Update the bin wrapper's import and docstring**

In `bin/fitLinearity.py`:
- Change the import line from
  `from fitLinearity.sanityCheck import SanityCheckConfig, cliTag, runFit, runPlot  # noqa: E402`
  to
  `from fitLinearity.fitLinearity import SanityCheckConfig, cliTag, runFit, runPlot  # noqa: E402`
- Change the module docstring `"""Run the linearity sanity check on one detector's lab ramp."""`
  to `"""Fit a per-pixel linearity correction for one detector's lab ramp."""`

- [ ] **Step 3: Update the test file's import**

In `tests/test_fitLinearity.py`, change
`from fitLinearity.sanityCheck import SanityCheckConfig, cliTag`
to
`from fitLinearity.fitLinearity import SanityCheckConfig, cliTag`

- [ ] **Step 4: Update the doc references**

Run this to find them:
```bash
grep -rn "sanityCheck\|sanity check\|sanity_check" CLAUDE.md README.md
```
In both `CLAUDE.md` and `README.md`, replace every `bin/sanityCheck.py` invocation with `bin/fitLinearity.py` (e.g. the real-data check `bin/sanityCheck.py --det 18734 --fit --plot` → `bin/fitLinearity.py --det 18734 --fit --plot`). If the surrounding prose calls it "the sanity check", reword to "the fit driver" / "the linearity fit" — present-tense, no "renamed" language. Leave any reference to upstream `lsst.obs.pfs.h4Linearity` untouched.

- [ ] **Step 5: Verify import path and suite**

```bash
python -c "import sys; sys.path.insert(0,'python'); from fitLinearity.fitLinearity import runFit, runPlot, cliTag, SanityCheckConfig; print('ok')"
pytest -q 2>&1 | tail -2
```
Expected: prints `ok`; suite is **9 failed, 63 passed** (unchanged — the 7 cliTag tests now live in `tests/test_fitLinearity.py`).

- [ ] **Step 6: Verify the CLI wrapper still wires up**

```bash
bin/fitLinearity.py --help 2>&1 | head -5
```
Expected: usage block prints listing `--det`, no import error.

- [ ] **Step 7: Confirm no stale references remain**

```bash
grep -rn "sanityCheck\|sanity_check" --include=*.py --include=*.md . | grep -v docs/superpowers
```
Expected: no output. (`docs/superpowers/` specs/plans are historical, excluded.)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Rename the fit driver module from sanityCheck to fitLinearity

Move the driver, its bin/ wrapper, and its test to the fitLinearity name;
import path is now fitLinearity.fitLinearity."
```

---

### Task 2: Config fields and cliTag

Add the three rate-stability config fields (defaulted, so nothing else breaks) and extend `cliTag`, with tests.

**Files:**
- Modify: `python/fitLinearity/fitLinearity.py` (the `SanityCheckConfig` dataclass and `cliTag`)
- Test: `tests/test_fitLinearity.py`

**Interfaces:**
- Consumes: `SanityCheckConfig`, `cliTag` from Task 1.
- Produces: `SanityCheckConfig` with `rateStability: bool = False`, `rateStabilityThreshold: float = 0.20`, `rateStabilityFloor: float = 5.0`; `cliTag` appending `_rs{threshold}` / `_rsf{floor}`.

- [ ] **Step 1: Write the failing cliTag tests**

Append to `tests/test_fitLinearity.py`:

```python
def testRateStabilityOffAddsNoTag():
    assert cliTag(_config(rateStability=False)) == "o4"


def testRateStabilityEnabledDefaultTag():
    assert cliTag(_config(rateStability=True)) == "o4_rs0.2"


def testRateStabilityNonDefaultThresholdTag():
    assert cliTag(_config(rateStability=True, rateStabilityThreshold=0.3)) == "o4_rs0.3"


def testRateStabilityNonDefaultFloorTag():
    assert cliTag(_config(rateStability=True, rateStabilityFloor=8.0)) == "o4_rs0.2_rsf8.0"


def testRateStabilityFloorIgnoredWhenDisabled():
    # Floor only appears when the gate is on.
    assert cliTag(_config(rateStability=False, rateStabilityFloor=8.0)) == "o4"
```

The `_config` helper builds `SanityCheckConfig(**defaults)` from the 14 original keys; the three new fields have dataclass defaults, so `_config()` and `_config(rateStability=True, ...)` both construct cleanly without touching the helper's `defaults` dict.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fitLinearity.py -k RateStability -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rateStability'`.

- [ ] **Step 3: Add the config fields**

In `python/fitLinearity/fitLinearity.py`, append these three fields to the end of `SanityCheckConfig` (after `fitrangeMax`):

```python
    rateStability: bool = False
    rateStabilityThreshold: float = 0.20
    rateStabilityFloor: float = 5.0
```

They must be last, because they carry defaults and the existing fields do not.

- [ ] **Step 4: Extend cliTag**

In `cliTag`, immediately before `return tag`, add:

```python
    if config.rateStability:
        tag += f"_rs{config.rateStabilityThreshold}"
        if config.rateStabilityFloor != 5.0:
            tag += f"_rsf{config.rateStabilityFloor}"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_fitLinearity.py -v`
Expected: all cliTag tests pass (the original 7 plus the 5 new).

- [ ] **Step 6: Lint and full suite**

Run: `ruff check python/fitLinearity/fitLinearity.py tests/test_fitLinearity.py && pytest -q 2>&1 | tail -2`
Expected: ruff clean; suite **9 failed, 68 passed** (63 + 5 new).

- [ ] **Step 7: Commit**

```bash
git add python/fitLinearity/fitLinearity.py tests/test_fitLinearity.py
git commit -m "Add rate-stability config fields and cliTag components

SanityCheckConfig gains rateStability/threshold/floor (defaulted); cliTag
appends _rs<threshold> when enabled and _rsf<floor> when the floor is non-default."
```

---

### Task 3: The `_foldRateStability` pure helper

The gate + fold, factored as a pure function so it is testable on tiny in-memory arrays.

**Files:**
- Modify: `python/fitLinearity/fitLinearity.py` (imports + new helper)
- Test: `tests/test_fitLinearity.py`

**Interfaces:**
- Consumes: nothing from Task 2 (takes plain `threshold`/`rateFloorADU` floats, not a config).
- Produces: `_foldRateStability(cumulativeLinear, badPixelMask, threshold, rateFloorADU) -> tuple[np.ndarray, RateStabilityResult | None]`. Returns `(foldedMask, result)`; `foldedMask` is a copy of `badPixelMask` with `RATE_UNSTABLE` OR-ed in where the gate rejects. On a too-short ramp it prints a skip message and returns `(badPixelMask.copy(), None)`.

- [ ] **Step 1: Write the failing fold-logic test**

Append to `tests/test_fitLinearity.py`. Add `import numpy as np` at the top of the file if it is not already imported.

```python
from fitLinearity.fitLinearity import _foldRateStability
from lsst.obs.pfs.h4Linearity.types import RATE_UNSTABLE


def _cumulativeFromDeltas(deltaRows):
    """Build a (N, H, W) linearized cumulative cube from per-pixel delta lists.

    deltaRows[i] is the length-(N-1) delta sequence for pixel (0, i); read 0 is 0.
    """
    deltas = np.array(deltaRows, dtype=np.float32).T          # (N-1, npix)
    nDeltas, npix = deltas.shape
    cube = np.zeros((nDeltas + 1, 1, npix), dtype=np.float32)
    np.cumsum(deltas, axis=0, out=cube[1:, 0, :])
    return cube


def testFoldRateStabilityFlagsOnlyTheUnstablePixel():
    # 3 pixels, 7 deltas each. Halves split [0:3] / [3:7].
    #   pixel 0: constant rate 10 -> stable
    #   pixel 1: 10 in first half, 100 in second -> unstable
    #   pixel 2: constant, but pre-masked (excluded from the gate)
    cube = _cumulativeFromDeltas([
        [10, 10, 10, 10, 10, 10, 10],
        [10, 10, 10, 100, 100, 100, 100],
        [10, 10, 10, 10, 10, 10, 10],
    ])
    badPixelMask = np.array([[0, 0, 0x0001]], dtype=np.int32)
    folded, result = _foldRateStability(cube, badPixelMask, threshold=0.20, rateFloorADU=5.0)

    assert folded[0, 0] == 0                       # stable good pixel: untouched
    assert folded[0, 1] == RATE_UNSTABLE           # unstable good pixel: flagged
    assert folded[0, 2] == 0x0001                  # pre-masked pixel: preserved, not gated
    assert result is not None and result.nRejected == 1
    assert badPixelMask[0, 1] == 0                 # input mask not mutated in place


def testFoldRateStabilityShortRampSkips():
    # 4 deltas < 2*minDeltasPerSegment (6): gate cannot form two halves.
    cube = _cumulativeFromDeltas([[10, 10, 10, 10]])
    badPixelMask = np.array([[0]], dtype=np.int32)
    folded, result = _foldRateStability(cube, badPixelMask, threshold=0.20, rateFloorADU=5.0)
    assert result is None
    assert folded[0, 0] == 0                       # nothing flagged
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fitLinearity.py -k FoldRateStability -v`
Expected: FAIL — `ImportError: cannot import name '_foldRateStability'`.

- [ ] **Step 3: Add the imports**

In `python/fitLinearity/fitLinearity.py`, add `RATE_UNSTABLE` to the existing `from lsst.obs.pfs.h4Linearity.types import (...)` block (keep the block alphabetically consistent with its current ordering), and add a new import line after it:

```python
from lsst.obs.pfs.h4Linearity.rateStability import detectRateInstability
```

- [ ] **Step 4: Implement `_foldRateStability`**

Add this function to `python/fitLinearity/fitLinearity.py` (place it above `runFit`):

```python
def _foldRateStability(cumulativeLinear, badPixelMask, threshold, rateFloorADU):
    """Run the split-half rate-stability gate on the linearized ramp and fold it in.

    ``cumulativeLinear`` is the linearized cumulative ramp in ``(N, H, W)``; its
    per-read deltas are the rate the gate tests for constancy. Only pixels good
    in ``badPixelMask`` are tested. Returns ``(foldedMask, result)`` where
    ``foldedMask`` is a copy of ``badPixelMask`` with ``RATE_UNSTABLE`` OR-ed in
    at rejected pixels. A ramp too short to form two testable halves is reported
    and skipped, returning ``(badPixelMask.copy(), None)``.
    """
    linDeltas = np.moveaxis(np.diff(cumulativeLinear, axis=0), 0, -1)  # (H, W, N-1)
    good = badPixelMask == 0
    flagMask = np.zeros(linDeltas.shape, dtype=bool)
    try:
        result = detectRateInstability(
            linDeltas, flagMask, goodPixelMask=good,
            threshold=threshold, rateFloorADU=rateFloorADU,
        )
    except ValueError as exc:
        print(f"  rate-stability skipped: {exc}", flush=True)
        return badPixelMask.copy(), None
    folded = badPixelMask.copy()
    folded[result.rejectMask] |= RATE_UNSTABLE
    return folded, result
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_fitLinearity.py -k FoldRateStability -v`
Expected: both tests pass.

- [ ] **Step 6: Lint and full suite**

Run: `ruff check python/fitLinearity/fitLinearity.py tests/test_fitLinearity.py && pytest -q 2>&1 | tail -2`
Expected: ruff clean (run `ruff check --fix` on those two files if I001 import-order fires); suite **9 failed, 70 passed** (68 + 2 new).

- [ ] **Step 7: Commit**

```bash
git add python/fitLinearity/fitLinearity.py tests/test_fitLinearity.py
git commit -m "Add _foldRateStability: run the split-half gate and fold RATE_UNSTABLE

Pure helper over the linearized cumulative ramp; folds the gate's reject mask
into a copy of the bad-pixel mask, skipping ramps too short to test."
```

---

### Task 4: Wire into runFit and the CLI

The I/O wrapper `_applyRateStability`, the `runFit` call site, and the three `bin/` arguments. End-to-end verification.

**Files:**
- Modify: `python/fitLinearity/fitLinearity.py` (`_applyRateStability` + `runFit` call site)
- Modify: `bin/fitLinearity.py` (three args + config wiring)

**Interfaces:**
- Consumes: `_foldRateStability` (Task 3); `SanityCheckConfig.rateStability/rateStabilityThreshold/rateStabilityFloor` (Task 2).
- Produces: `_applyRateStability(loaded, cumulativeLinear, config, fitsPath) -> RateStabilityResult | None`; `bin/fitLinearity.py` accepting `--rate-stability`, `--rate-stability-threshold`, `--rate-stability-floor`.

- [ ] **Step 1: Implement `_applyRateStability`**

Add to `python/fitLinearity/fitLinearity.py`, directly below `_foldRateStability`:

```python
def _applyRateStability(loaded, cumulativeLinear, config, fitsPath):
    """Fold the rate-stability gate into ``loaded`` and re-save the FITS.

    Runs :func:`_foldRateStability` on the linearized ramp, updates
    ``loaded.badPixelMask`` in place, reports the outcome, records the run's
    threshold/floor and the unstable count in the summary header, and re-saves.
    Returns the ``RateStabilityResult``, or ``None`` if the ramp was too short.
    """
    print(
        f"Rate-stability gate (threshold={config.rateStabilityThreshold}, "
        f"floor={config.rateStabilityFloor}) ...",
        flush=True,
    )
    nGood = int((loaded.badPixelMask == 0).sum())  # good pixels the gate tested, pre-fold
    folded, result = _foldRateStability(
        cumulativeLinear, loaded.badPixelMask,
        threshold=config.rateStabilityThreshold,
        rateFloorADU=config.rateStabilityFloor,
    )
    if result is None:
        return None

    loaded.badPixelMask = folded
    finiteFraction = result.fraction[np.isfinite(result.fraction)]
    p50 = float(np.percentile(finiteFraction, 50)) if finiteFraction.size else float("nan")
    p95 = float(np.percentile(finiteFraction, 95)) if finiteFraction.size else float("nan")
    print(
        f"  RATE_UNSTABLE: {result.nRejected} rejected "
        f"({100 * result.nRejected / max(nGood, 1):.3f}% of good), "
        f"{result.nUntestable} untestable",
        flush=True,
    )
    print(f"  fraction p50={p50:.4f} p95={p95:.4f}", flush=True)

    loaded.diagnostics.summary["rateStabilityThreshold"] = config.rateStabilityThreshold
    loaded.diagnostics.summary["rateStabilityFloor"] = config.rateStabilityFloor
    loaded.diagnostics.summary["rateUnstableCount"] = result.nRejected

    print(f"Re-saving {fitsPath} with RATE_UNSTABLE flags ...", flush=True)
    nirLinearity.saveFits(fitsPath, loaded)
    return result
```

- [ ] **Step 2: Call it from runFit**

In `python/fitLinearity/fitLinearity.py`, in `runFit`, find the block that ends the fit reporting — the two lines:

```python
    print(f"  above-range pixel fraction: {aboveCount / nPix:.4f}", flush=True)
    _t("done (fit)", t0)
```

Insert, between those two lines:

```python
    if config.rateStability:
        _applyRateStability(loaded, result.cumulativeLinear, config, fitsPath)
```

`loaded`, `result`, `config`, and `fitsPath` are all already in scope there (`result = _applyBridge(loaded, correctedRamp)` ran earlier in `runFit`).

- [ ] **Step 3: Add the CLI arguments**

In `bin/fitLinearity.py`, add these three arguments to the parser (place them after the existing `--bad-linearity-multiplier` argument, before `--fitrange-min`):

```python
    parser.add_argument("--rate-stability", action="store_true",
                        help="Run the split-half rate-stability gate after the fit and "
                             "fold RATE_UNSTABLE into the saved correction")
    parser.add_argument("--rate-stability-threshold", type=float, default=0.20,
                        help="Fractional half-vs-half rate-disagreement threshold (default: 0.20)")
    parser.add_argument("--rate-stability-floor", type=float, default=5.0,
                        help="Rate-floor DN in the disagreement denominator (default: 5.0)")
```

- [ ] **Step 4: Thread them into the config**

In `bin/fitLinearity.py`, in the `SanityCheckConfig(...)` construction, add these three keyword arguments (after `fitrangeMax=args.fitrange_max,`):

```python
        rateStability=args.rate_stability,
        rateStabilityThreshold=args.rate_stability_threshold,
        rateStabilityFloor=args.rate_stability_floor,
```

- [ ] **Step 5: Verify wiring without a full fit**

```bash
bin/fitLinearity.py --help 2>&1 | grep -A1 "rate-stability"
ruff check python/fitLinearity/fitLinearity.py bin/fitLinearity.py
pytest -q 2>&1 | tail -2
```
Expected: `--help` lists `--rate-stability`, `--rate-stability-threshold`, `--rate-stability-floor`; ruff clean; suite **9 failed, 70 passed** (unchanged — no new tests here, wiring is covered by Task 3's unit test plus the end-to-end run below).

- [ ] **Step 6: End-to-end verification (the real proof — needs lab data + EUPS)**

```bash
bin/fitLinearity.py --det 18734 --fit --rate-stability
```
Expected, in order:
- prints `out: /work/cloomis/outputs/fitLinearity/18734/o4_rs0.2` (the `_rs0.2` cliTag).
- completes the fit → save → load → apply chain (bitwise-equal checks `True`).
- prints the `Rate-stability gate ...` line, then `RATE_UNSTABLE: <n> rejected ...`, `<n> untestable`, and `fraction p50=... p95=...`, then `Re-saving ... with RATE_UNSTABLE flags ...`.

Then confirm the flag is in the saved FITS and the input is untouched:
```bash
python -c "
import sys; sys.path.insert(0,'python')
import lsst.obs.pfs.h4Linearity as nir
from lsst.obs.pfs.h4Linearity.types import RATE_UNSTABLE
c = nir.loadFits('/work/cloomis/outputs/fitLinearity/18734/o4_rs0.2/18734_linearity.fits')
print('RATE_UNSTABLE pixels:', int((c.badPixelMask & RATE_UNSTABLE > 0).sum()))
print('summary threshold:', c.diagnostics.summary.get('rateStabilityThreshold'))
"
find ../jhu-data/18734 -newer pyproject.toml
```
Expected: a non-negative `RATE_UNSTABLE pixels` count matching the reported `nRejected`, `summary threshold: 0.2`, and the `find` returns nothing (input untouched). The FITS is written under `/work/cloomis/outputs/`, outside the repo, and is not committed.

- [ ] **Step 7: Commit**

```bash
git add python/fitLinearity/fitLinearity.py bin/fitLinearity.py
git commit -m "Wire --rate-stability through runFit and the CLI

runFit folds the gate into the saved correction when --rate-stability is set;
the wrapper reports counts and fraction percentiles and records provenance in
the FITS summary header."
```

---

## Self-Review

**Spec coverage.** Rename (module + wrapper + test + docs) → Task 1. Config fields + `_rs`/`_rsf` cliTag → Task 2. Gate on linearized deltas, `flagMask` zeros, `goodPixelMask` from the fit, fold `RATE_UNSTABLE`, `ValueError` skip → Task 3. `_applyRateStability` reporting + provenance + re-save, `runFit` wiring, three `bin/` args, `--fit`-only scope → Task 4. cliTag tests + fold-logic test + end-to-end → Tasks 2, 3, 4. Retained symbol names → Global Constraints + Task 1 note.

**Placeholder scan.** No TBD/TODO; every code step carries the literal code; every command has an expected result.

**Type consistency.** `SanityCheckConfig` fields `rateStability`/`rateStabilityThreshold`/`rateStabilityFloor` are defined in Task 2 and consumed by that spelling in Task 4's config wiring and by `config.rateStability*` in the `runFit` call and `_applyRateStability`. `_foldRateStability(cumulativeLinear, badPixelMask, threshold, rateFloorADU)` is defined in Task 3 and called with those positional/keyword names in Task 4's `_applyRateStability`. `_applyRateStability(loaded, cumulativeLinear, config, fitsPath)` is defined and called in Task 4 with `result.cumulativeLinear` — the same `result` from `_applyBridge`. `RATE_UNSTABLE` and `detectRateInstability` imports are added once, in Task 3.

**Reporting note.** `nGood` in `_applyRateStability` is counted from the pre-fold mask (`loaded.badPixelMask == 0`) and used only as the denominator of the printed "% of good" line — it does not affect the mask or the saved FITS.
