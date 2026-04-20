"""Chebyshev polynomial nonlinearity model: t = Σ c_k T_k(x) (per pixel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from relin.models.base import BlockFitResult
from relin.types import FIT_FAILED, INSUFFICIENT_POINTS, NON_MONOTONIC


@dataclass(frozen=True)
class PolynomialModel:
    """Pluggable Chebyshev polynomial-fit model. Default 4th order."""

    order: int = 4
    modelName: str = "CHEBYSHEV"

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise ValueError(f"order must be an int, got {type(self.order).__name__}")
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")

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
            dKplus2 = derivCoefs[k + 2] if k + 2 < order else np.zeros_like(derivCoefs[0])
            derivCoefs[k] = dKplus2 + 2.0 * (k + 1) * coefficients[k + 1]
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

    def fitBlock(
        self,
        m: np.ndarray,
        t: np.ndarray,
        valid: np.ndarray,
        conditionNumberLimit: float,
    ) -> BlockFitResult:
        """Fit a polynomial at every pixel in the block.

        Uses per-pixel rescaling of ``m`` to the range [-1, 1] before forming
        the normal equations, which keeps the conditioning bounded independent
        of raw DN magnitude. Coefficients are unscaled at the end.
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

        # Exponents needed in the fit: 0 .. nCoefs - 1
        exps = np.arange(0, nCoefs, dtype=np.int32)

        # For AtA: need mScaled ** (expI + expJ), expI, expJ in exps.
        # For Atb: need mScaled ** expI with t weighting.
        # Iterate over exponent sums from 0 to 2*(nCoefs - 1).
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

        coefficients = np.zeros((p + 1, H, W), dtype=np.float32)
        for k in range(nCoefs):
            coefficients[k] = solUnscaled[..., k].astype(np.float32)
        coefficients[:, skip] = 0.0

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

    def toFitsHdus(self, correction) -> list[fits.ImageHDU]:
        """Serialize model coefficients to a single ImageHDU named COEFFS."""
        hdu = fits.ImageHDU(data=correction.coefficients, name="COEFFS")
        hdu.header["ORDER"] = (self.order, "polynomial order")
        hdu.header["COMMENT"] = "COEFFS axis 0 is the Chebyshev coefficient index; C0 (T_0) first."
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
        coefficients = np.asarray(coeffsHdu.data, dtype=np.float32)
        return cls(order=order), coefficients
