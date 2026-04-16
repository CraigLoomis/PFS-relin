# relin — per-pixel nonlinearity correction package

**Status:** design / pre-implementation
**Date:** 2026-04-16
**Scope:** full package design for the MVP; follow-up plan will enumerate implementation steps.

## 1. Purpose and scope

`relin` is a Python library for fitting per-pixel nonlinearity corrections from up-the-ramp IR-detector linearity calibration data and applying those corrections to new ramps. Typical detector size is 4096×4096; ramps have variable read counts (the reference example has 29 reads).

### What the package does

1. **Fit** per-pixel nonlinearity corrections from one or more already-photodiode-corrected ramps, using a pluggable `Model` (polynomial, 4th-order default).
2. **Apply** a fitted correction to a full ramp (primary path) or a single already-cumulated frame (secondary path).
3. **Report diagnostics** per pixel (residual RMS, monotonicity, condition number, bit-flag reason codes) plus a dataset-wide summary.

### What the package does not do

- **No photodiode handling.** The photodiode time series is a lab-specific artifact. Callers multiply `deltas *= photodiode[:, None, None]` (or whatever their lab convention is) upstream. The package only sees already-corrected `deltas`.
- **No data loading beyond a thin `loadNpz` development convenience.** Production callers provide already-loaded `Ramp` objects from their own loaders.
- **No saturation detection or per-read masking in the core fit path.** A pixel-level mask (`validMask: (H, W)`, `0 == valid`) is supported. A saturation-detection utility is a planned but separate module.
- **No CLI.**
- **No cosmic-ray handling, reference-pixel subtraction, dark subtraction, flat fielding, or cross-talk correction.** Upstream concerns.
- **No iterative outlier rejection during the fit.** Diagnostics surface outliers; downstream tools decide whether to re-mask and re-fit.

### Planned (post-MVP) extensions

- `relin.saturation`: consumes a `Ramp` and returns a `(H, W)` mask suitable for feeding back into `fit(...)` via `validMask`.
- Additional `Model` implementations (spline, LUT). The MVP ships the plumbing; no non-polynomial implementations.
- Threaded/parallel tile processing.
- CLI (`relin fit`, `relin apply`) driving the npz loader.
- Public docs site.

## 2. Concept and math

Given `K` ramps (each possibly a different read count `N_k`):

For each ramp `k`:
- `R_k = median over pixels of deltas_k[0]` — scalar rate reference, computed over pixels allowed by the caller's mask (so reference pixels, etc., don't skew the PRNU median). If `validMask is None`, the median is taken over all pixels.
- `t_k[n] = R_k · (n + 1)` — linearization target for read `n`, same for every pixel in ramp `k`. Assumes uniform read cadence.
- `m_k[pixel, n] = Σᵢ ≤ n deltas_k[pixel, i]` — per-pixel cumulative measured signal.

A `validMask` value of `None` on a `Ramp` means "all pixels valid" — no pixel is flagged `MASKED_BY_INPUT` by that ramp, and the median is computed over the full frame.

For each pixel, the fit concatenates `(m_k[pixel, :], t_k[:])` pairs across all ramps where the pixel is valid, and fits the chosen model.

For `PolynomialModel(order=p)` with default `p = 4` and general form (not forced through origin):

```
t = c₀ + c₁·m + c₂·m² + … + c_p·m^p
```

A flag `forceThroughOrigin=True` drops `c₀` (`c₀ ≡ 0`) for callers who want zero signal ⇒ zero flux.

## 3. Public API

### Dataclasses

All data types are frozen dataclasses in `python/relin/types.py`.

```python
@dataclass(frozen=True)
class Ramp:
    deltas: np.ndarray          # (N, H, W), float32; photodiode correction already applied
    validMask: np.ndarray | None = None  # (H, W), uint8 or bool; 0 == valid

@dataclass(frozen=True)
class LinearizedRamp:
    cumulativeLinear: np.ndarray   # (N, H, W), float32
    outOfRangeMask: np.ndarray     # (N, H, W), bool
    badPixelMask: np.ndarray       # (H, W), uint8 — carried from the fit

@dataclass(frozen=True)
class Diagnostics:
    residualRms: np.ndarray        # (H, W), float32
    maxAbsResidual: np.ndarray     # (H, W), float32
    nPointsUsed: np.ndarray        # (H, W), int32
    monotonic: np.ndarray          # (H, W), bool
    conditionNumber: np.ndarray    # (H, W), float32
    summary: dict                  # scalar aggregates: fractions by bad-pixel reason,
                                   # residualRms percentiles, totals, model description

@dataclass(frozen=True)
class LinearityCorrection:
    model: Model                   # instance; carries hyperparameters
    coefficients: np.ndarray       # shape depends on model; polynomial: (order+1, H, W), float32
    fitMin: np.ndarray             # (H, W), float32 — min m used per pixel
    fitMax: np.ndarray             # (H, W), float32 — max m used per pixel
    badPixelMask: np.ndarray       # (H, W), uint8 — bit flags (see below)
    diagnostics: Diagnostics
```

### Bad-pixel bit flags (uint8)

| Flag | Bit | Meaning |
|------|-----|---------|
| `MASKED_BY_INPUT` | `0x01` | Caller's `validMask` had nonzero at this pixel in at least one ramp |
| `INSUFFICIENT_POINTS` | `0x02` | Fewer valid samples than `order + 2` after combining ramps |
| `FIT_FAILED` | `0x04` | Singular normal equations / numerical failure |
| `NON_MONOTONIC` | `0x08` | Post-fit derivative sign check failed on `[fitMin, fitMax]` |

Multiple flags may combine per pixel. `badPixelMask == 0` means usable.

### Top-level functions

```python
# python/relin/fit.py
def fit(
    ramps: Sequence[Ramp],
    model: Model = PolynomialModel(order=4),
    blockSize: tuple[int, int] = (512, 512),
    conditionNumberLimit: float = 1e12,
) -> LinearityCorrection: ...

# python/relin/apply.py
def apply(correction: LinearityCorrection, ramp: Ramp) -> LinearizedRamp: ...

def applyFrame(
    correction: LinearityCorrection, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...   # (linearized (H,W), oorMask (H,W))

# python/relin/io.py
def saveFits(path: Path, correction: LinearityCorrection) -> None: ...
def loadFits(path: Path) -> LinearityCorrection: ...

# python/relin/loaders.py  (dev/test convenience only)
def loadNpz(path: Path) -> tuple[Ramp, np.ndarray]: ...   # (Ramp, photodiode)
```

### Model protocol

```python
# python/relin/models/base.py
class Model(Protocol):
    def fitBlock(
        self,
        m: np.ndarray,      # (nPoints, blockH, blockW), float32
        t: np.ndarray,      # (nPoints,), float32 — same target for every pixel (formulation B)
        valid: np.ndarray,  # (nPoints, blockH, blockW), bool
    ) -> BlockFitResult: ...

    def evaluate(
        self, coefficients: np.ndarray, m: np.ndarray
    ) -> np.ndarray: ...

    def isMonotonic(
        self, coefficients: np.ndarray, mMin: np.ndarray, mMax: np.ndarray
    ) -> np.ndarray: ...

    def toFitsHdus(self, correction: LinearityCorrection) -> list[fits.HDU]: ...

    @classmethod
    def fromFitsHdus(cls, hdus: list[fits.HDU]) -> tuple["Model", np.ndarray]: ...
```

`BlockFitResult` carries per-block coefficients, condition numbers, residual stats, and per-pixel fit-failure flags.

## 4. Fit algorithm

Executed by `fit()` in `python/relin/fit.py`, delegating the per-block solve to `model.fitBlock`.

### Step 1 — per-ramp precomputation

For each ramp `k`:
- `m_k = np.cumsum(deltas_k, axis=0)` → `(N_k, H, W)`, float32. Computed once per ramp and held in memory across tiles (≈ 1.8 GB per 29-read, 4k² ramp).
- `R_k = np.median(deltas_k[0, validPixels])` restricted to caller-mask-allowed pixels.
- `t_k = R_k * np.arange(1, N_k + 1, dtype=np.float64)` → `(N_k,)`.

### Step 2 — tile iteration

Iterate `(rowSlice, colSlice)` tiles sized `blockSize` (default 512×512) across `(H, W)`. Per tile, gather from all ramps:
- `mTile`: `m_k[:, rowSlice, colSlice]` per ramp, concatenated along axis 0 → `(ΣN_k, hTile, wTile)`.
- `tTile`: `t_k` per ramp, concatenated → `(ΣN_k,)`.
- `validTile`: per-ramp `validMask[rowSlice, colSlice]` broadcast to `N_k` along axis 0 and concatenated → `(ΣN_k, hTile, wTile)`.

### Step 3 — block fit (polynomial model)

Inside `PolynomialModel.fitBlock`:

- Accumulate normal equations without materializing the full Vandermonde. For each exponent pair `(i, j)` with `0 ≤ i ≤ j ≤ p` (skip `0` if `forceThroughOrigin`), compute one `(ΣN_k, hTile, wTile)` intermediate `mTile ** (i+j)` (~30 MB at 29 reads × 512² × float32), multiply by `validTile.astype(float32)`, sum along axis 0, and write into `AtA[..., i, j]` and `AtA[..., j, i]`. Similarly for `Atb[..., i] = (validTile · mTile^i · tTile[:, None, None]).sum(axis=0)`. Peak intermediate allocation stays around one `(ΣN_k, hTile, wTile)` tile — no `(..., p+1)` tensor is ever built.
- Count valid points per pixel: `nPoints = validTile.sum(axis=0)`. Pixels with `nPoints < p + 2` get flagged `INSUFFICIENT_POINTS` and skipped in the solve.
- `conditionNumberTile = np.linalg.cond(AtA)` → `(hTile, wTile)`, computed first and retained for diagnostics.
- Batched solve: `coefficientsTile = np.linalg.solve(AtA, Atb[..., None])[..., 0]` for non-flagged pixels. Pixels with `conditionNumberTile > conditionNumberLimit` (default `1e12`, parameter of `fit`) are flagged `FIT_FAILED` and their coefficient row is set to zero. Any remaining solver failure also yields `FIT_FAILED`.

### Step 4 — per-tile residual and range stats

- `tPred[h, w, n] = Σ_i coefficientsTile[i, h, w] · mTile[n, h, w]^i`, masked by `validTile`.
- `residuals = (tTile[:, None, None] - tPred) * validTile`
- `residualRmsTile = sqrt((residuals**2).sum(axis=0) / nPoints)`
- `maxAbsResidualTile = abs(residuals).max(axis=0)` (masked)
- `fitMinTile`, `fitMaxTile` from min/max of `mTile` over valid reads per pixel.

### Step 5 — post-fit monotonicity check

Per tile, evaluate the derivative polynomial at 32 evenly-spaced points on `[fitMinTile, fitMaxTile]` per pixel. If all values positive, `monotonicTile = True`; else flag `NON_MONOTONIC`. Vectorized over the tile.

### Step 6 — stitch tiles

Each tile writes into preallocated full-frame outputs: `coefficients`, `fitMin`, `fitMax`, `residualRms`, `maxAbsResidual`, `nPointsUsed`, `monotonic`, `conditionNumber`, `badPixelMask`.

### Step 7 — propagate input mask

For any pixel where `validMask` was nonzero in *any* ramp, OR `MASKED_BY_INPUT` into `badPixelMask`.

### Step 8 — dataset-wide summary

`Diagnostics.summary` is populated with: percentiles (p50, p95, p99) of `residualRms` over unflagged pixels, fractions by bad-pixel reason, total pixel count, and a model description string.

### Performance and memory

- Peak concurrent allocations during a 1-ramp 29-read 4k² fit: cumulative `(N, H, W)` float32 (≈ 1.8 GB), one tile-sized intermediate `mTile**(i+j)` (≈ 30 MB at 29 reads × 512² × float32), output `(order+1, H, W)` and per-pixel diagnostic arrays (≈ 500 MB). Fits comfortably in 4 GB.
- Expected single-threaded runtime (order-of-magnitude): cumsum + median ≈ 1 s; 64 tiles × ~0.3 s each for fit+residuals ≈ 20 s; save to FITS ≈ a few s. Profiling and tuning will confirm or contradict.
- No threading in the MVP. The per-tile loop is structured so `ThreadPoolExecutor` can be dropped in later (numpy releases the GIL during the heavy ops).

## 5. Apply path

### `apply(correction, ramp) -> LinearizedRamp`

1. Shape check: `ramp.deltas.shape[1:] == correction.coefficients.shape[1:]`. Raise `ValueError` on mismatch.
2. Compute cumulative: `m = np.cumsum(ramp.deltas, axis=0)` → `(N, H, W)`. Tile across `(H, W)` mirroring the fit; each tile writes into a preallocated `cumulativeLinear: (N, H, W) float32` output.
3. Per tile, evaluate: `t = correction.model.evaluate(coefficientsTile, mTile)` via Horner's method on the coefficient axis. Output shape `(N, hTile, wTile)`.
4. Out-of-range per tile: `oor = (mTile < fitMinTile[None]) | (mTile > fitMaxTile[None])` → `(N, hTile, wTile) bool`. Extrapolated values are still returned; the flag lets downstream code decide whether to trust them.
5. **Bad-pixel pass-through:** for any pixel where `correction.badPixelMask != 0`, leave the output unchanged — copy `m` into `cumulativeLinear` for that pixel (no polyval, no NaN). The mask carried on `LinearizedRamp` is the authoritative signal for downstream consumers.
6. Return `LinearizedRamp(cumulativeLinear, outOfRangeMask, badPixelMask=correction.badPixelMask.copy())`.

### `applyFrame(correction, m) -> (t, oor)`

Identical per-pixel logic; input is already-cumulated `(H, W)` float32. Returns `(linearized: (H, W), oor: (H, W) bool)`. Bad pixels leave their input values unchanged in `linearized`. Used internally by `apply` and exposed publicly for downstream composition (e.g., slope-fitting code that builds its own `m`).

### Edge cases

- Ramp with `N == 0`: raise `ValueError`.
- `m` containing `NaN` from upstream: `NaN` propagates through polyval; `oor` is `False` at `NaN` positions (comparison with `NaN` is `False`), which is acceptable — callers see `NaN` in the output and know to ignore.
- `fitMin == fitMax` for a pixel (pathological): any `m` is flagged OOR; evaluation still returns a value.

### Thread safety

`LinearityCorrection` is frozen and its arrays are not mutated post-construction. Multiple `apply`/`applyFrame` calls against the same correction in different threads are safe.

## 6. FITS persistence

FITS is the primary on-disk format. Round-trip invariant: `loadFits(saveFits(x)) == x` up to float32 precision — verified by test.

### HDU layout

| HDU | Name | Type | Shape / contents |
|-----|------|------|------------------|
| 0 | `PRIMARY` | header-only | `MODEL` (str, e.g. `POLYNOMIAL`), `ORDER`, `FTHROUGH0` (bool), `NRAMPS`, `FITDATE` (ISO UTC), `RELINVER` (package version), plus scalar summaries from `Diagnostics.summary` |
| 1 | `COEFFS` | ImageHDU float32 | `(order+1, H, W)` polynomial coefficients, constant term first |
| 2 | `FITMIN` | ImageHDU float32 | `(H, W)` |
| 3 | `FITMAX` | ImageHDU float32 | `(H, W)` |
| 4 | `BPMASK` | ImageHDU uint8 | `(H, W)` bit-flag bad-pixel mask |
| 5 | `RESRMS` | ImageHDU float32 | `(H, W)` per-pixel residual RMS |
| 6 | `RESMAX` | ImageHDU float32 | `(H, W)` per-pixel max absolute residual |
| 7 | `NPOINTS` | ImageHDU int32 | `(H, W)` points used in fit |
| 8 | `MONOTON` | ImageHDU uint8 | `(H, W)` 0/1 |
| 9 | `CONDNUM` | ImageHDU float32 | `(H, W)` |

### Conventions

- `CHECKSUM` and `DATASUM` added to every image HDU before write (`hdu.add_checksum()`). Verified on load; a mismatch emits a warning (not an error).
- Bit-flag meanings written into the `BPMASK` header as `COMMENT` lines (`MASKED_BY_INPUT = 0x01`, …) so the file is self-describing in DS9 or other FITS viewers.
- `COEFFS` header notes coefficient ordering (`C0` = constant term first).
- `RELINVER` stored in PRIMARY. On load, a missing or newer version emits a warning; version-specific migration logic can be added if/when it becomes necessary.

### Model-specific persistence

`saveFits` calls `correction.model.toFitsHdus(correction)` to produce the model-specific HDUs (today: `COEFFS` only; future spline/LUT models contribute different extensions). `loadFits` reads `MODEL` from the primary header and dispatches via a module-level registry in `python/relin/models/__init__.py`:

```python
_MODEL_REGISTRY: dict[str, type[Model]] = {"POLYNOMIAL": PolynomialModel}
```

Future models self-register via the same mechanism; `io.py` does not change.

## 7. Package layout

```
relin/
├── pyproject.toml           # project metadata, deps, hatchling build backend
├── README.md                # install, one usage example, pointer to this spec
├── python/
│   └── relin/
│       ├── __init__.py      # re-exports the public API
│       ├── types.py
│       ├── fit.py
│       ├── apply.py
│       ├── io.py
│       ├── loaders.py
│       ├── saturation.py    # stub with a TODO docstring; no implementation
│       └── models/
│           ├── __init__.py  # registry + PolynomialModel registration
│           ├── base.py      # Model protocol, BlockFitResult
│           └── polynomial.py
├── tests/
│   ├── conftest.py          # synthetic ramp fixtures; optional real-data slice
│   ├── test_types.py
│   ├── test_polynomial_model.py
│   ├── test_fit.py
│   ├── test_apply.py
│   ├── test_io_fits.py
│   ├── test_integration.py  # end-to-end on a small synthetic dataset
│   └── data/
│       └── tinyRamp.npz     # ~64×64 synthetic ramp for fast tests
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-16-relin-package-design.md  # this document
├── examples/                # gitignored; holds large real-data .npz files
└── .gitignore
```

`python/` (not `src/`) is the chosen source root.

### Public API re-exports in `__init__.py`

`Ramp`, `LinearizedRamp`, `Diagnostics`, `LinearityCorrection`, `Model`, `PolynomialModel`, `fit`, `apply`, `applyFrame`, `saveFits`, `loadFits`, and bad-pixel flag constants.

## 8. Style conventions

- **Functions, methods, variables, parameters: camelCase** (e.g., `fitBlock`, `validMask`, `outOfRangeMask`). This overrides PEP 8's snake_case default and is project-wide.
- Classes: PascalCase.
- Module-level constants: `UPPER_SNAKE_CASE` (e.g., `MASKED_BY_INPUT`).
- Module filenames: short / single-word where possible; camelCase only for compound names.
- Type hints required on all public functions; `from __future__ import annotations` at the top of each module.

## 9. Testing

Framework: `pytest`. Coverage is not a numeric target; every public function gets at least a happy-path test and one failure-mode test.

### Test categories

- **Unit: types.** Dataclass construction, frozen semantics.
- **Unit: PolynomialModel.** Known-answer tests where synthetic data follows an exact polynomial; fitted coefficients must match within float32 tolerance. Edge cases: `INSUFFICIENT_POINTS` pixel, singular-matrix (`FIT_FAILED`) pixel, crafted non-monotonic coefficient set → `NON_MONOTONIC` flag.
- **Unit: fit.** Single-ramp and multi-ramp paths; variable `N_k` across ramps; mask propagation (`MASKED_BY_INPUT`); batched normal equations match per-pixel `np.polyfit` on a small grid.
- **Unit: apply.** `apply` and `applyFrame` agreement; out-of-range flag behavior; bad-pixel pass-through (input values preserved, no NaN overwrite); NaN-in-input propagation.
- **Unit: io.** FITS round-trip preserves all arrays within float32 precision; checksum fields present and verified on load; model registry dispatch picks the right subclass.
- **Determinism:** identical inputs → bitwise-identical coefficients across runs.
- **Block-size invariance:** same data fit with `blockSize=(512,512)` vs. `(256,256)` → identical outputs within float32 precision.
- **Integration:** end-to-end fit → save → load → apply on a 64×64 synthetic ramp, checking residual percentiles fall below a configured threshold. Optional test using a 64×64 slice extracted from a real example file; skipped if the file is absent.

### Fixtures

- Synthetic ramp builder: given a known polynomial `f`, constructs `deltas` such that `cumsum(deltas)` exactly matches the inverse `f⁻¹` (or a forward construction), plus a per-pixel PRNU field. Used throughout unit tests.
- `tinyRamp.npz`: small committed fixture for load/integration tests; generated by a script in `tests/conftest.py` if missing.

## 10. Dependencies and tooling

- **Python:** `>= 3.12`.
- **Package manager / environment:** `uv`. `pyproject.toml` is authoritative; `uv sync` provisions the dev environment.
- **Runtime dependencies:** `numpy >= 1.26`, `scipy >= 1.11`, `astropy >= 6.0`.
- **Dev dependencies:** `pytest`, `pytest-cov`, `ruff` (lint + format; configured to allow camelCase identifier names).
- **Build backend:** `uv_build` (uv's native backend; `[tool.uv.build-backend]` points `module-root = "python"` so the non-standard source location is discovered).

## 11. Out of scope for MVP

Reiterated for clarity, since several items below naturally tempt "just one more thing":

- `saturation.py` ships as a stub module with a TODO docstring; no implementation.
- Spline/LUT models: plumbing (`Model` protocol, registry, FITS dispatch) is present; no additional concrete models.
- Parallel/threaded tile processing.
- CLI entry point.
- Public documentation site (docstrings + this spec + README only).
- Dark/flat/cosmic-ray handling (upstream).
- Photodiode or other lamp-correction utilities (caller's responsibility).
