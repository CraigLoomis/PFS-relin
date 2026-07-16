# Rate-stability path for the linearity fit driver

**Date:** 2026-07-15
**Status:** approved

## Purpose

Add an opt-in `--rate-stability` path to the fit driver that runs the PIPE2D-1844
split-half rate-stability gate (`detectRateInstability`) on the linearized ramp,
folds its `RATE_UNSTABLE` verdict into the correction's bad-pixel mask, and
re-saves the FITS. As a prerequisite structural step, rename the driver module
from `sanityCheck` to `fitLinearity`.

## Scope

1. **Rename** `sanityCheck` → `fitLinearity` across the module, its `bin/`
   wrapper, and its test, updating imports and docs.
2. **Feature**: the opt-in `--rate-stability` gate, `--fit`-only, that flags and
   re-saves.

Out of scope: the public function and class names (`runFit`, `runPlot`,
`cliTag`, `SanityCheckConfig`) are retained — only the file was requested for
rename, and renaming the symbols is unrequested churn. Rate-stability on an
existing FITS without refitting (`--plot`-time re-fold) is not built; to flag a
correction, re-run `--fit`.

## Part 1 — Rename

| from | to |
|---|---|
| `python/fitLinearity/sanityCheck.py` | `python/fitLinearity/fitLinearity.py` |
| `bin/sanityCheck.py` | `bin/fitLinearity.py` |
| `tests/test_sanityCheck.py` | `tests/test_fitLinearity.py` |

The import path becomes `from fitLinearity.fitLinearity import runFit, runPlot,
cliTag, SanityCheckConfig` — a `datetime.datetime`-style module/package stutter,
accepted as the cost of the eponymous name. Update:

- `bin/fitLinearity.py`'s import line.
- `tests/test_fitLinearity.py`'s import line.
- `CLAUDE.md` real-data check command → `bin/fitLinearity.py --det 18734 --fit --plot`.
- `README.md` any `bin/sanityCheck.py` invocation.

`git mv` all three so history follows. No behavior changes in this part.

## Part 2 — Rate-stability gate

### Data flow (in `runFit`, only when `config.rateStability`)

The gate needs *linearized* deltas, so it runs after the existing
fit → save → load → apply chain, reusing the `result = _applyBridge(loaded,
correctedRamp)` output. `result.cumulativeLinear` is the linearized cumulative
ramp in `(N, H, W)`.

```python
linDeltas = np.moveaxis(np.diff(result.cumulativeLinear, axis=0), 0, -1)  # (H, W, N-1)
flagMask  = np.zeros(linDeltas.shape, dtype=bool)   # no CR repair here — matches the diagnostic's no-CR fallback
good      = loaded.badPixelMask == 0                # test only pixels that fit
rs = detectRateInstability(
    linDeltas, flagMask, goodPixelMask=good,
    threshold=config.rateStabilityThreshold,
    rateFloorADU=config.rateStabilityFloor,
)   # minDeltasPerSegment stays at the library default (3)
loaded.badPixelMask[rs.rejectMask] |= RATE_UNSTABLE   # 0x0800
```

`detectRateInstability` raises `ValueError` when `nDeltas < 2 *
minDeltasPerSegment` (i.e. < 6). Real ramps clear this easily (18734 gives 29
linearized deltas), but the call is wrapped so that a too-short ramp reports a
clear message and skips the fold rather than crashing the run.

`RATE_UNSTABLE` (`0x0800`) is imported from `lsst.obs.pfs.h4Linearity.types`.

### Structure

A helper in `fitLinearity.py`:

```python
def _applyRateStability(loaded, cumulativeLinear, config, fitsPath):
    """Run the split-half rate-stability gate on the linearized ramp,
    fold RATE_UNSTABLE into loaded.badPixelMask, record provenance, and
    re-save the FITS. Returns the RateStabilityResult, or None if the ramp
    was too short to test."""
```

owns gate + fold + provenance + re-save + reporting, keeping `runFit` readable.
`runFit` calls it only when `config.rateStability`, immediately after the apply
step (after `good`/`nGood` are computed).

### Config and tag

`SanityCheckConfig` gains three fields:

```python
rateStability: bool
rateStabilityThreshold: float   # default 0.20
rateStabilityFloor: float       # default 5.0
```

`cliTag` appends, after the existing components:

- `_rs{rateStabilityThreshold}` whenever `rateStability` is true (so a default
  run tags `o4`, an enabled run `o4_rs0.2`, keeping swept runs in distinct
  output dirs).
- `_rsf{rateStabilityFloor}` only when `rateStability` is true and the floor is
  not 5.0.

### Reporting and provenance

`_applyRateStability` prints:

- `nRejected` (both absolute and as a fraction of good pixels), `nUntestable`.
- `fraction` p50/p95 over testable pixels (finite `fraction` entries).

Before the re-save it writes provenance into `loaded.diagnostics.summary`
(the same dict that already carries `visit`/`detector`, serialized to the FITS
header): `rateStabilityThreshold`, `rateStabilityFloor`, `rateUnstableCount`.
It then re-saves with `nirLinearity.saveFits(fitsPath, loaded)`, logging the
re-save so the earlier "badPixelMask bitwise-equal" line (which compared the
pre-fold masks) is not misread — the on-disk FITS now intentionally carries the
added flag.

### bin/fitLinearity.py

Three new arguments, threaded into the config (plain floats — no `-1`-means-None
mapping):

```
--rate-stability                    store_true (default off)
--rate-stability-threshold FLOAT    default 0.20
--rate-stability-floor FLOAT        default 5.0
```

## Testing

- **cliTag** (`tests/test_fitLinearity.py`): extend the existing cases —
  rate-stability off → no `rs` component; enabled at default → `_rs0.2`;
  non-default threshold → `_rs0.3`; non-default floor → `_rsf<v>`; combined with
  other non-default params in the documented order.
- **Fold logic** (new, fast, no lab data): build a tiny linearized cumulative
  cube where one pixel's rate deliberately jumps between the two halves and the
  rest are constant-rate; run the real `detectRateInstability` + the fold; assert
  the `RATE_UNSTABLE` bit lands on exactly the unstable pixel, good pixels stay
  clear, and a pre-existing mask bit on another pixel is preserved (OR, not
  overwrite). This exercises our integration against the real library on
  in-memory arrays.
- **End-to-end** (manual, needs lab data + EUPS): `bin/fitLinearity.py --det
  18734 --fit --rate-stability` writes to `.../fitLinearity/18734/o4_rs0.2/`,
  prints the counts, and the saved FITS carries `RATE_UNSTABLE` on the rejected
  pixels.

## Verification gate

The suite's known-red baseline is 9 failed (upstream transpose). After this
work: no new failures beyond those 9, every new test passes, and the end-to-end
run produces a flagged FITS in the `o4_rs0.2` directory.
