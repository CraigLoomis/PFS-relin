# Switch to Chebyshev Polynomials

## Summary

Replace the power-basis polynomial fitting and evaluation in `PolynomialModel` with
Chebyshev polynomials. Coefficients are stored in the Chebyshev basis; evaluation
requires the per-pixel `[fitMin, fitMax]` range to map inputs to `[-1, 1]`. This is
a breaking change: old FITS files with `MODEL = "POLYNOMIAL"` will not load.

## Motivation

Chebyshev polynomials are near-optimal for polynomial approximation on a bounded
interval. They minimize the Runge phenomenon and produce a better-conditioned
Vandermonde-like system than the power basis, even with the current per-pixel
rescaling. Storing coefficients in the Chebyshev basis preserves this conditioning
for any downstream re-evaluation or analysis.

## Design

### Fitting (`fitBlock`)

1. Compute per-pixel `fitMin` and `fitMax` from the valid `m` values (unchanged).
2. Map `m` to `x in [-1, 1]` with the affine transform:
   `x = 2 * (m - fitMin) / (fitMax - fitMin) - 1`.
3. Build normal equations using Chebyshev basis functions `T_k(x)` instead of
   `x^k`. Compute `T_k` via the three-term recurrence `T_0 = 1`, `T_1 = x`,
   `T_{k+1} = 2x * T_k - T_{k-1}`. This preserves the current memory pattern:
   one tile-sized intermediate array at a time, no full Vandermonde materialized.
4. Solve the normal equations with `np.linalg.solve` as before (batched, with
   per-pixel fallback on failure).
5. Store coefficients directly in the Chebyshev basis. No unscaling step.
6. Condition-number check and bad-pixel flagging unchanged.

### Evaluation (`evaluate`)

Replace Horner's method with Clenshaw's algorithm, the standard method for
evaluating a Chebyshev series:

Given coefficients `c_0, ..., c_p` and mapped input `x`:
```
b_{p+1} = 0
b_{p}   = c_p
b_{k}   = 2 * x * b_{k+1} - b_{k+2} + c_k   (k = p-1, ..., 1)
result  = x * b_1 - b_2 + c_0
```

Same computational cost as Horner: one multiply-add per degree.

### Application (`apply`)

`apply()` computes `m = cumsum(deltas)` as before, then maps to `x` using the
stored `fitMin`/`fitMax` before calling `evaluate`. The mapping requires both
arrays, which are already carried in `LinearityCorrection` and saved in the FITS
file.

### Derivative (monotonicity check)

The derivative of a Chebyshev series `sum c_k T_k(x)` is itself a Chebyshev
series with known recurrence for the derivative coefficients:

```
d_{p-1} = 2 * p * c_p
d_{k}   = d_{k+2} + 2 * (k+1) * c_{k+1}   (k = p-2, ..., 1)
d_0     = d_2 / 2 + c_1
```

The derivative series is evaluated with Clenshaw and sampled at the same grid of
points across `[fitMin, fitMax]` to check monotonicity. Additionally, the chain
rule factor `2 / (fitMax - fitMin)` applies since the derivative is with respect
to the mapped variable `x`, not the original `m`.

### IO / FITS Format

- `MODEL` header: `"CHEBYSHEV"` (was `"POLYNOMIAL"`).
- `COEFFS` HDU: shape `(order+1, H, W)` float32, Chebyshev-basis coefficients.
  Index 0 is `c_0` (the `T_0` coefficient).
- `FITMIN` / `FITMAX` HDUs: unchanged in format. Now required for evaluation.
- All other HDUs (BPMASK, RESRMS, RESMAX, NPOINTS, MONOTON, CONDNUM): unchanged.
- `loadFits`: requires `MODEL == "CHEBYSHEV"`. Raises a clear error if it
  encounters `"POLYNOMIAL"`.

### `forceThroughOrigin` Removed

In the Chebyshev basis, forcing `t(m=0) = 0` is a linear constraint across all
coefficients rather than simply dropping the constant term. This feature is
dropped. The `forceThroughOrigin` parameter is removed from `PolynomialModel` and
all call sites.

## Scope

### Changed files

| File | Change |
|------|--------|
| `models/polynomial.py` | Chebyshev basis in `fitBlock`, Clenshaw in `evaluate`, Chebyshev derivative in `derivative` |
| `apply.py` | Map `m` to `x` via `fitMin`/`fitMax` before calling `evaluate` |
| `io.py` | `MODEL` header `"CHEBYSHEV"`, reject old `"POLYNOMIAL"` on load |
| `types.py` | Remove `forceThroughOrigin` from `PolynomialModel` and `LinearityCorrection` |
| `fit.py` | Remove `forceThroughOrigin` plumbing |
| `tests/` | Update for Chebyshev basis and removal of `forceThroughOrigin` |

### Unchanged

Threading, tiling, bad-pixel logic, diagnostics, `Ramp` type, `blockSize`,
`conditionNumberLimit`, residual computation, summary statistics.
