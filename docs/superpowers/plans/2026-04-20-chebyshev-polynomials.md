# Chebyshev Polynomials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the power-basis polynomial model with Chebyshev polynomials, storing coefficients in the Chebyshev basis and requiring `[fitMin, fitMax]` for evaluation.

**Architecture:** Modify `PolynomialModel` in-place: Chebyshev basis in `fitBlock`, Clenshaw's algorithm in `evaluate`, Chebyshev derivative recurrence in `isMonotonic`. Update `apply.py` to map `m → x ∈ [-1,1]` via `fitMin`/`fitMax`. Change `MODEL` header to `"CHEBYSHEV"`, remove `forceThroughOrigin`. Breaking change — old FITS files will not load.

**Tech Stack:** Python 3.12, numpy, astropy.io.fits, pytest, uv

---

### Task 1: Remove `forceThroughOrigin` from `PolynomialModel` and update FITS serialization

**Files:**
- Modify: `python/relin/models/polynomial.py:14-27` (dataclass fields)
- Modify: `python/relin/models/polynomial.py:95-112` (fitBlock fto plumbing)
- Modify: `python/relin/models/polynomial.py:242-266` (FITS HDU methods)
- Modify: `python/relin/models/__init__.py:23` (registration uses new modelName)
- Test: `tests/test_polynomial_model.py`

- [ ] **Step 1: Update the `PolynomialModel` dataclass**

Remove the `forceThroughOrigin` field and change `modelName` to `"CHEBYSHEV"`:

```python
@dataclass(frozen=True)
class PolynomialModel:
    """Pluggable Chebyshev polynomial-fit model. Default 4th order."""

    order: int = 4

    modelName: str = "CHEBYSHEV"
```

Keep `__post_init__` as-is (order validation unchanged).

- [ ] **Step 2: Remove `forceThroughOrigin` from `fitBlock`**

In `fitBlock` (line 108-112), remove the three lines that reference `fto`:

```python
# REMOVE these lines:
fto = self.forceThroughOrigin
startExp = 1 if fto else 0
nCoefs = p + 1 - startExp  # free coefficients
```

Replace with:

```python
nCoefs = p + 1
```

Also remove the `startExp` variable usage at lines 145 and 149 — change `exps` to:

```python
exps = np.arange(0, nCoefs, dtype=np.int32)
```

And in the coefficient-stitch loop (line 208), since `exps` is now always `[0, 1, ..., p]`, simplify to:

```python
coefficients = np.zeros((p + 1, H, W), dtype=np.float32)
for k in range(nCoefs):
    coefficients[k] = solUnscaled[..., k].astype(np.float32)
coefficients[:, skip] = 0.0
```

- [ ] **Step 3: Update `toFitsHdus` — remove FTHRU0 header**

Replace `toFitsHdus` (lines 242-251) with:

```python
def toFitsHdus(self, correction) -> list[fits.ImageHDU]:
    """Serialize model coefficients to a single ImageHDU named COEFFS."""
    hdu = fits.ImageHDU(data=correction.coefficients, name="COEFFS")
    hdu.header["ORDER"] = (self.order, "polynomial order")
    hdu.header["COMMENT"] = "COEFFS axis 0 is the Chebyshev coefficient index; C0 (T_0) first."
    return [hdu]
```

- [ ] **Step 4: Update `fromFitsHdus` — remove FTHRU0 read**

Replace `fromFitsHdus` (lines 253-266) with:

```python
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
    coefficients = np.asarray(coeffsHdu.data, dtype=np.float32)
    return cls(order=order), coefficients
```

- [ ] **Step 5: Update model registry**

In `python/relin/models/__init__.py`, the line `registerModel(PolynomialModel)` will now register under `"CHEBYSHEV"` (since `modelName` changed). No code change needed — just verify this is correct.

- [ ] **Step 6: Update module docstring**

Change line 1 of `python/relin/models/polynomial.py` from:
```python
"""Polynomial nonlinearity model: t = c0 + c1*m + ... + cp*m^p (per pixel)."""
```
to:
```python
"""Chebyshev polynomial nonlinearity model: t = Σ c_k T_k(x) (per pixel)."""
```

- [ ] **Step 7: Update tests for removed `forceThroughOrigin`**

In `tests/test_polynomial_model.py`:

Remove `test_fitBlockRespectsForceThroughOrigin` (lines 191-205) and `test_polynomialModelFitsForceThroughOrigin` (lines 263-278) entirely.

Update `test_defaultConstructor` (lines 12-15):

```python
def test_defaultConstructor():
    m = PolynomialModel()
    assert m.order == 4
    assert m.modelName == "CHEBYSHEV"
```

Update `test_polynomialModelFitsRoundTrip` (lines 236-260) — remove `forceThroughOrigin` from construction and assertion:

```python
def test_polynomialModelFitsRoundTrip():
    H, W = 2, 3
    coeffs = np.arange(5 * H * W, dtype=np.float32).reshape(5, H, W)
    model = PolynomialModel(order=4)

    class Stub:
        def __init__(self, c):
            self.coefficients = c
    hdus = model.toFitsHdus(Stub(coeffs))

    assert len(hdus) == 1
    assert hdus[0].name == "COEFFS"
    np.testing.assert_array_equal(hdus[0].data, coeffs)

    assert hdus[0].header["ORDER"] == 4

    loadedModel, loadedCoeffs = PolynomialModel.fromFitsHdus(hdus)
    assert loadedModel.order == 4
    np.testing.assert_array_equal(loadedCoeffs, coeffs)
```

- [ ] **Step 8: Run tests to verify removal is clean**

Run: `uv run pytest tests/test_polynomial_model.py -v`

Expected: All remaining tests pass. The two forceThroughOrigin tests are gone. Some evaluate/fit tests may fail because we haven't switched to Chebyshev yet — that's expected and will be fixed in subsequent tasks.

- [ ] **Step 9: Commit**

```bash
git add python/relin/models/polynomial.py python/relin/models/__init__.py tests/test_polynomial_model.py
git commit -m "Remove forceThroughOrigin and change modelName to CHEBYSHEV"
```

---

### Task 2: Implement Chebyshev evaluation via Clenshaw's algorithm

The `evaluate` method currently uses Horner's method on power-basis coefficients. Replace it with Clenshaw's algorithm for Chebyshev series.

**Files:**
- Modify: `python/relin/models/polynomial.py:29-52` (evaluate method)
- Test: `tests/test_polynomial_model.py`

- [ ] **Step 1: Write failing tests for Chebyshev evaluation**

Add these tests to `tests/test_polynomial_model.py`:

```python
def test_evaluateChebyshevIdentity():
    """t = T_1(x) = x. With coeffs c0=0, c1=1, rest=0, evaluate(x) = x."""
    model = PolynomialModel(order=4)
    H, W, N = 4, 5, 6
    coeffs = np.zeros((5, H, W), dtype=np.float32)
    coeffs[1] = 1.0  # c1 = 1 → T_1(x) = x
    # x values in [-1, 1]
    x = np.linspace(-1, 1, N * H * W, dtype=np.float32).reshape(N, H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, x, rtol=1e-6)


def test_evaluateChebyshevConstant():
    """t = 3*T_0(x) = 3. Constant everywhere."""
    model = PolynomialModel(order=2)
    H, W, N = 2, 3, 5
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[0] = 3.0
    x = np.linspace(-1, 1, N * H * W, dtype=np.float32).reshape(N, H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, 3.0, atol=1e-6)


def test_evaluateChebyshevKnownSeries():
    """t = 2*T_0(x) + 3*T_1(x) + 0.5*T_2(x).
    T_0=1, T_1=x, T_2=2x^2-1. So t = 2 + 3x + 0.5(2x^2-1) = 1.5 + 3x + x^2."""
    model = PolynomialModel(order=2)
    H, W, N = 2, 3, 10
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[0] = 2.0
    coeffs[1] = 3.0
    coeffs[2] = 0.5
    x = np.tile(np.linspace(-1, 1, N, dtype=np.float32)[:, None, None], (1, H, W))
    t = model.evaluate(coeffs, x)
    expected = 1.5 + 3.0 * x + x ** 2
    np.testing.assert_allclose(t, expected, rtol=1e-5)


def test_evaluateChebyshevSingleFrameShape():
    """evaluate must accept (H, W) input."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0  # T_1(x) = x
    x = np.linspace(-1, 1, H * W, dtype=np.float32).reshape(H, W)
    t = model.evaluate(coeffs, x)
    np.testing.assert_allclose(t, x, rtol=1e-6)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `uv run pytest tests/test_polynomial_model.py::test_evaluateChebyshevKnownSeries -v`

Expected: FAIL (Horner gives wrong result for Chebyshev coefficients).

- [ ] **Step 3: Implement Clenshaw's algorithm in `evaluate`**

Replace the `evaluate` method (lines 29-52) with:

```python
def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the per-pixel Chebyshev series via Clenshaw's algorithm.

    Parameters
    ----------
    coefficients
        Shape ``(order+1, H, W)``, float32. ``coefficients[k]`` is the
        coefficient of T_k(x).
    x
        Shape ``(..., H, W)``, float32. Mapped input in [-1, 1].

    Returns
    -------
    t
        Same shape as ``x``.
    """
    coefficients = np.asarray(coefficients)
    x = np.asarray(x)
    order = coefficients.shape[0] - 1

    if order == 0:
        return np.broadcast_to(coefficients[0], x.shape).astype(x.dtype).copy()

    # Clenshaw recurrence for Chebyshev series:
    # b_{p+1} = 0, b_p = c_p
    # b_k = 2*x*b_{k+1} - b_{k+2} + c_k   for k = p-1, ..., 1
    # result = x*b_1 - b_2 + c_0
    bNext = np.zeros_like(x, dtype=x.dtype)           # b_{k+2}
    bCurr = np.full_like(x, coefficients[order], dtype=x.dtype)  # b_{k+1}
    for k in range(order - 1, 0, -1):
        bPrev = 2.0 * x * bCurr - bNext + coefficients[k]
        bNext = bCurr
        bCurr = bPrev
    # Final: result = x * b_1 - b_2 + c_0
    return x * bCurr - bNext + coefficients[0]
```

- [ ] **Step 4: Run all evaluate tests**

Run: `uv run pytest tests/test_polynomial_model.py -k evaluate -v`

Expected: All 4 new Chebyshev tests pass. The old `test_evaluateIdentity` and `test_evaluateKnownPolynomial` will now fail because they assume power-basis coefficients.

- [ ] **Step 5: Remove old power-basis evaluate tests**

Remove `test_evaluateIdentity`, `test_evaluateKnownPolynomial`, and `test_evaluateSingleFrameShape` from `tests/test_polynomial_model.py` (they're superseded by the new Chebyshev evaluate tests).

- [ ] **Step 6: Run tests to confirm**

Run: `uv run pytest tests/test_polynomial_model.py -k evaluate -v`

Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Replace Horner evaluation with Clenshaw algorithm for Chebyshev series"
```

---

### Task 3: Implement Chebyshev `isMonotonic` via derivative recurrence

**Files:**
- Modify: `python/relin/models/polynomial.py:54-93` (isMonotonic method)
- Test: `tests/test_polynomial_model.py`

- [ ] **Step 1: Write failing test for Chebyshev monotonicity check**

Add to `tests/test_polynomial_model.py`:

```python
def test_isMonotonicChebyshevLinear():
    """t = T_1(x) = x is monotonically increasing on [-1, 1]."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[1] = 1.0
    mMin = np.full((H, W), -1.0, dtype=np.float32)
    mMax = np.full((H, W), 1.0, dtype=np.float32)
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert result.shape == (H, W)
    assert result.all()


def test_isMonotonicChebyshevDetectsNonMonotonic():
    """t = T_2(x) = 2x^2 - 1 has derivative 4x, which is negative for x < 0.
    On [-1, 1] this is non-monotonic."""
    model = PolynomialModel(order=2)
    H, W = 2, 3
    coeffs = np.zeros((3, H, W), dtype=np.float32)
    coeffs[2] = 1.0  # T_2(x) = 2x^2 - 1
    mMin = np.full((H, W), -1.0, dtype=np.float32)
    mMax = np.full((H, W), 1.0, dtype=np.float32)
    result = model.isMonotonic(coeffs, mMin, mMax)
    assert not result.any()
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run pytest tests/test_polynomial_model.py::test_isMonotonicChebyshevDetectsNonMonotonic -v`

Expected: FAIL (old power-basis derivative gives wrong result).

- [ ] **Step 3: Implement Chebyshev derivative + monotonicity check**

Replace `isMonotonic` (lines 54-93) with:

```python
def isMonotonic(
    self,
    coefficients: np.ndarray,
    mMin: np.ndarray,
    mMax: np.ndarray,
    nSamples: int = 32,
) -> np.ndarray:
    """Return an ``(H, W)`` boolean map: ``True`` if the fit is monotonically
    increasing on ``[mMin, mMax]`` per pixel.

    Computes the derivative of the Chebyshev series, evaluates it at
    ``nSamples`` evenly-spaced points on the mapped interval ``[-1, 1]``,
    and checks that all sampled derivatives (in original m-space) are
    non-negative.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    order = coefficients.shape[0] - 1
    if order < 1:
        return np.ones(coefficients.shape[1:], dtype=bool)

    # Derivative of Chebyshev series: d/dx [Σ c_k T_k(x)] = Σ d_k T_k(x)
    # where the derivative coefficients satisfy the recurrence:
    #   d_{p-1} = 2 * p * c_p
    #   d_k     = d_{k+2} + 2 * (k+1) * c_{k+1}   for k = p-2, ..., 1
    #   d_0     = d_2 / 2 + c_1
    derivCoefs = np.zeros((order, *coefficients.shape[1:]), dtype=np.float64)
    derivCoefs[order - 1] = 2.0 * order * coefficients[order]
    for k in range(order - 2, 0, -1):
        derivCoefs[k] = derivCoefs[k + 2] + 2.0 * (k + 1) * coefficients[k + 1]
    # d_0: handle the k+2 index carefully (it's 0 if order < 3)
    d2 = derivCoefs[2] if order >= 3 else np.zeros_like(derivCoefs[0])
    derivCoefs[0] = d2 / 2.0 + coefficients[1]

    H, W = mMin.shape
    # Sample x in [-1, 1]
    xSamples = np.linspace(-1.0, 1.0, nSamples, dtype=np.float64)[:, None, None]
    xSamples = np.broadcast_to(xSamples, (nSamples, H, W)).copy()

    # Evaluate derivative Chebyshev series at sample points via Clenshaw.
    derivOrder = order - 1
    if derivOrder == 0:
        d = np.broadcast_to(derivCoefs[0], (nSamples, H, W)).copy()
    else:
        bNext = np.zeros((nSamples, H, W), dtype=np.float64)
        bCurr = np.broadcast_to(
            derivCoefs[derivOrder], (nSamples, H, W)
        ).copy().astype(np.float64)
        for k in range(derivOrder - 1, 0, -1):
            bPrev = 2.0 * xSamples * bCurr - bNext + derivCoefs[k]
            bNext = bCurr
            bCurr = bPrev
        d = xSamples * bCurr - bNext + derivCoefs[0]

    # Chain rule: dt/dm = (dt/dx) * (dx/dm) = (dt/dx) * 2/(fitMax - fitMin)
    # For monotonicity we only care about sign, and 2/(fitMax - fitMin) > 0,
    # so we can just check dt/dx >= 0.
    allNonNegative = (d >= 0).all(axis=0)
    degenerate = mMax <= mMin
    return allNonNegative | degenerate
```

- [ ] **Step 4: Remove old power-basis monotonicity tests**

Remove `test_isMonotonicOnLinearCoefficients` and `test_isMonotonicDetectsNonMonotonicQuadratic` from `tests/test_polynomial_model.py`.

- [ ] **Step 5: Run monotonicity tests**

Run: `uv run pytest tests/test_polynomial_model.py -k Monotonic -v`

Expected: 2 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Implement Chebyshev derivative recurrence for isMonotonic"
```

---

### Task 4: Implement Chebyshev `fitBlock`

Replace the power-basis normal equations with Chebyshev basis functions.

**Files:**
- Modify: `python/relin/models/polynomial.py:95-240` (fitBlock method)
- Test: `tests/test_polynomial_model.py`

- [ ] **Step 1: Write failing test for Chebyshev fitBlock**

Add to `tests/test_polynomial_model.py`:

```python
def test_fitBlockChebyshevRecoversLinear():
    """Fit t = x (the identity on [-1,1]) and verify c1 = 1, others ≈ 0."""
    N, H, W = 20, 2, 3
    # m values uniformly spaced per pixel (same for all pixels here)
    mVals = np.linspace(100.0, 500.0, N, dtype=np.float32)
    m = np.tile(mVals[:, None, None], (1, H, W))
    # Target: t = m (identity relationship)
    t = mVals.copy()
    valid = np.ones((N, H, W), dtype=bool)

    model = PolynomialModel(order=2)
    result = model.fitBlock(m=m, t=t, valid=valid, conditionNumberLimit=1e12)

    assert result.coefficients.shape == (3, H, W)
    # After mapping m → x in [-1,1] and fitting Chebyshev coefficients,
    # then evaluating: model.evaluate(coeffs, x) should reproduce t.
    fitMin = result.fitMin
    fitMax = result.fitMax
    x = 2.0 * (m - fitMin[None]) / (fitMax - fitMin)[None] - 1.0
    tPred = model.evaluate(result.coefficients, x.astype(np.float32))
    np.testing.assert_allclose(tPred, t[:, None, None], rtol=1e-4, atol=1e-2)
    assert (result.badPixelMask == 0).all()
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `uv run pytest tests/test_polynomial_model.py::test_fitBlockChebyshevRecoversLinear -v`

Expected: FAIL (fitBlock still uses power basis).

- [ ] **Step 3: Rewrite `fitBlock` to use Chebyshev basis**

Replace the `fitBlock` method (lines 95-240) with:

```python
def fitBlock(
    self,
    m: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    conditionNumberLimit: float,
) -> BlockFitResult:
    """Fit a Chebyshev polynomial at every pixel in the block.

    Maps ``m`` to ``x ∈ [-1, 1]`` via ``x = 2*(m - fitMin)/(fitMax - fitMin) - 1``
    before forming normal equations in the Chebyshev basis ``T_k(x)``.
    """
    nPoints, H, W = m.shape
    p = self.order
    nCoefs = p + 1

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

    # Affine map m → x ∈ [-1, 1]: x = 2*(m - fitMin)/(fitMax - fitMin) - 1
    denom = fitMax - fitMin
    denom = np.where(denom > 0, denom, 1.0)  # avoid /0 for degenerate pixels
    x = 2.0 * (mD - fitMin[None]) / denom[None] - 1.0  # (N, H, W)

    # Flag insufficient-points pixels now.
    insufficientPixels = nPointsUsed < (nCoefs + 1)
    badMask[insufficientPixels] |= INSUFFICIENT_POINTS

    # Build Chebyshev basis values T_k(x) via three-term recurrence,
    # then accumulate AtA and Atb.
    AtA = np.zeros((H, W, nCoefs, nCoefs), dtype=np.float64)
    Atb = np.zeros((H, W, nCoefs), dtype=np.float64)

    # Compute Chebyshev basis values T_k(x) via three-term recurrence.
    # nCoefs is small (typically 5), so storing all of them is fine.
    tCheb = []
    for k in range(nCoefs):
        if k == 0:
            tk = np.ones_like(x)
        elif k == 1:
            tk = x.copy()
        else:
            tk = 2.0 * x * tCheb[k - 1] - tCheb[k - 2]
        tCheb.append(tk)

    # Accumulate normal equations
    for i in range(nCoefs):
        vTi = v64 * tCheb[i]  # (N, H, W)
        Atb[..., i] = (vTi * t64[:, None, None]).sum(axis=0)
        for j in range(i, nCoefs):
            val = (vTi * tCheb[j]).sum(axis=0)  # (H, W)
            AtA[..., i, j] = val
            if i != j:
                AtA[..., j, i] = val

    # Condition number check
    with np.errstate(divide="ignore", invalid="ignore"):
        conditionNumber = np.linalg.cond(AtA)
    conditionNumber = np.nan_to_num(conditionNumber, nan=np.inf, posinf=np.inf)

    fitFailed = (~insufficientPixels) & (conditionNumber > conditionNumberLimit)
    badMask[fitFailed] |= FIT_FAILED

    skip = insufficientPixels | fitFailed
    identityBlock = np.eye(nCoefs, dtype=np.float64)
    AtA[skip] = identityBlock
    Atb[skip] = 0.0

    # Batched solve
    try:
        sol = np.linalg.solve(AtA, Atb[..., None])[..., 0]  # (H, W, nCoefs)
    except np.linalg.LinAlgError:
        sol = np.zeros((H, W, nCoefs), dtype=np.float64)
        for hi in range(H):
            for wi in range(W):
                if skip[hi, wi]:
                    continue
                try:
                    sol[hi, wi] = np.linalg.solve(AtA[hi, wi], Atb[hi, wi])
                except np.linalg.LinAlgError:
                    badMask[hi, wi] |= FIT_FAILED
                    sol[hi, wi] = 0.0
        skip = skip | (badMask & FIT_FAILED != 0)

    # No unscaling needed — coefficients are in Chebyshev basis directly.
    coefficients = np.zeros((p + 1, H, W), dtype=np.float32)
    for k in range(nCoefs):
        coefficients[k] = sol[..., k].astype(np.float32)
    coefficients[:, skip] = 0.0

    # Residuals: evaluate fit at each read and compare to t.
    # Map m to x for evaluation.
    tPred = self.evaluate(coefficients, x.astype(np.float32))  # (N, H, W)
    residuals = (t[:, None, None].astype(np.float32) - tPred) * valid
    nForDiv = np.where(nPointsUsed > 0, nPointsUsed, 1).astype(np.float32)
    residualRms = np.sqrt((residuals ** 2).sum(axis=0) / nForDiv).astype(np.float32)
    maxAbsResidual = np.abs(residuals).max(axis=0).astype(np.float32)

    # Monotonicity check on the mapped interval.
    monotonic = self.isMonotonic(
        coefficients, fitMin.astype(np.float32), fitMax.astype(np.float32)
    )
    monotonic[skip] = False
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

- [ ] **Step 4: Remove old power-basis fitBlock tests**

In `tests/test_polynomial_model.py`, remove:
- `test_fitBlockRecoversLinearCoefficients` (uses power-basis coefficient expectations)
- `test_fitBlockRecoversPolynomialCoefficients` (uses power-basis coefficient expectations)

Keep:
- `test_fitBlockFlagsInsufficientPoints` (unchanged — still tests flag logic)
- `test_fitBlockFlagsFitFailedWhenIllConditioned` (unchanged — still tests flag logic)
- `test_fitBlockFlagsNonMonotonicFit` (but needs update — see next step)

- [ ] **Step 5: Update `test_fitBlockFlagsNonMonotonicFit`**

The existing test constructs `m` and `t` vectors directly. With Chebyshev fitting, the fitBlock now maps `m → x` internally, so the same input data should still trigger non-monotonic detection. Verify the test still works as-is (it should — the test constructs a parabola `t = -m^2 + 2m` which is non-monotonic regardless of basis). No code change expected, but verify.

- [ ] **Step 6: Run all polynomial model tests**

Run: `uv run pytest tests/test_polynomial_model.py -v`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/relin/models/polynomial.py tests/test_polynomial_model.py
git commit -m "Implement Chebyshev basis in fitBlock with affine mapping to [-1,1]"
```

---

### Task 5: Update `apply.py` to map `m → x` before evaluation

**Files:**
- Modify: `python/relin/apply.py:24-26` (apply function)
- Modify: `python/relin/apply.py:45-52` (applyFrame function)
- Test: `tests/test_apply.py`

- [ ] **Step 1: Write failing test for apply with Chebyshev correction**

Add to `tests/test_apply.py`:

```python
def test_applyChebyshevMapsToMinusOnePlusOne(smallSyntheticRamp):
    """After fitting, applying to the same ramp should reproduce targets."""
    ramp, truth = smallSyntheticRamp
    correction = fit([ramp])
    result = apply(correction, ramp)
    # cumulativeLinear should closely match the target at every read.
    expected = np.broadcast_to(
        truth["target"][:, None, None], ramp.deltas.shape
    )
    np.testing.assert_allclose(
        result.cumulativeLinear, expected, rtol=1e-3, atol=1e-1
    )
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_apply.py::test_applyChebyshevMapsToMinusOnePlusOne -v`

Expected: FAIL (apply passes raw `m` to evaluate, but coefficients are now in Chebyshev basis expecting `x ∈ [-1,1]`).

- [ ] **Step 3: Update `apply` to map `m → x`**

In `python/relin/apply.py`, replace lines 24-26:

```python
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)  # (N, H, W)

    t = correction.model.evaluate(correction.coefficients, m)
```

with:

```python
    m = np.cumsum(ramp.deltas.astype(np.float32), axis=0)  # (N, H, W)

    # Map m → x ∈ [-1, 1] for Chebyshev evaluation
    denom = correction.fitMax - correction.fitMin
    denom = np.where(denom > 0, denom, 1.0)
    x = 2.0 * (m - correction.fitMin[None]) / denom[None] - 1.0

    t = correction.model.evaluate(correction.coefficients, x)
```

- [ ] **Step 4: Update `applyFrame` similarly**

In `python/relin/apply.py`, replace lines 52-53:

```python
    t = correction.model.evaluate(correction.coefficients, m)
```

with:

```python
    # Map m → x ∈ [-1, 1] for Chebyshev evaluation
    denom = correction.fitMax - correction.fitMin
    denom = np.where(denom > 0, denom, 1.0)
    x = 2.0 * (m - correction.fitMin) / denom - 1.0

    t = correction.model.evaluate(correction.coefficients, x)
```

- [ ] **Step 5: Run apply tests**

Run: `uv run pytest tests/test_apply.py -v`

Expected: All tests pass (the existing `test_applyOnFittedRampYieldsTarget` effectively tests the same thing as the new test).

- [ ] **Step 6: Remove the redundant new test**

Since `test_applyOnFittedRampYieldsTarget` already covers this, remove `test_applyChebyshevMapsToMinusOnePlusOne`.

- [ ] **Step 7: Commit**

```bash
git add python/relin/apply.py tests/test_apply.py
git commit -m "Map m to [-1,1] in apply/applyFrame for Chebyshev evaluation"
```

---

### Task 6: Update IO — `MODEL` header and backward-incompatibility

**Files:**
- Modify: `python/relin/io.py` (no code change needed — MODEL comes from model.modelName)
- Test: `tests/test_io_fits.py`

- [ ] **Step 1: Update `test_saveLoadRoundTrip` assertions**

In `tests/test_io_fits.py`, change line 48:

```python
    assert loaded.model.modelName == "CHEBYSHEV"
```

And line 58:

```python
        assert hdul[0].header["MODEL"] == "CHEBYSHEV"
```

- [ ] **Step 2: Update `test_saveLoadRoundTripSummary`**

The summary should now contain `"modelName": "CHEBYSHEV"`. No code change needed if the test just round-trips generic keys. But `test_integration.py` line 51 checks:

```python
    assert summary.get("modelName") == "POLYNOMIAL"
```

Change to:

```python
    assert summary.get("modelName") == "CHEBYSHEV"
```

- [ ] **Step 3: Add test for old POLYNOMIAL model rejection**

Add to `tests/test_io_fits.py`:

```python
def test_loadFitsRejectsOldPolynomialModel(tmp_path):
    """FITS files with MODEL='POLYNOMIAL' from before the Chebyshev switch
    must produce a clear error."""
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    # Overwrite MODEL to simulate an old-format file
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["MODEL"] = "POLYNOMIAL"
        hdul.flush()
    with pytest.raises(ValueError, match="Unknown model"):
        loadFits(path)
```

- [ ] **Step 4: Update `test_models_init.py`**

In `tests/test_models_init.py`, change line 9:

```python
def test_chebyshevRegistered():
    assert MODEL_REGISTRY["CHEBYSHEV"] is PolynomialModel
```

Remove or update `test_registerModelRejectsDuplicateByDefault` — it tries to register `PolynomialModel` again, which now has `modelName = "CHEBYSHEV"`:

```python
def test_registerModelRejectsDuplicateByDefault():
    import pytest
    with pytest.raises(ValueError):
        registerModel(PolynomialModel)  # CHEBYSHEV already registered
```

- [ ] **Step 5: Run IO and registry tests**

Run: `uv run pytest tests/test_io_fits.py tests/test_models_init.py tests/test_integration.py -v`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_io_fits.py tests/test_models_init.py tests/test_integration.py
git commit -m "Update tests for CHEBYSHEV model header and reject old POLYNOMIAL files"
```

---

### Task 7: Run full test suite and fix any remaining failures

**Files:**
- Possibly: any test file that references `"POLYNOMIAL"` or `forceThroughOrigin`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`

Expected: All tests pass. If any fail, diagnose and fix.

- [ ] **Step 2: Check for stale references**

Search for any remaining references to `"POLYNOMIAL"` or `forceThroughOrigin` in the source:

```bash
grep -r "POLYNOMIAL\|forceThroughOrigin\|FTHRU0" python/relin/ tests/ --include="*.py"
```

Fix any stale references found.

- [ ] **Step 3: Run full test suite again if fixes were needed**

Run: `uv run pytest tests/ -v`

Expected: All pass.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "Clean up remaining POLYNOMIAL/forceThroughOrigin references"
```

Only commit if there were changes. Skip if no stale references found.
