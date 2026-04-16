"""Top-level fit(): tile-iterate over (H, W) and delegate to model.fitBlock."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

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

# Worker-count resolution constants. Tunable at module level; the tests
# monkeypatch `os.cpu_count` rather than these, so changing them does not
# break tests but will change the default behavior for small/large frames.
_SMALL_FRAME_PIXEL_LIMIT = 1_000_000   # H*W below this → sequential default
_DEFAULT_WORKER_CAP = 8                # auto-detected cpu_count is capped here

# Override point for tests. Default is the real ThreadPoolExecutor; a test
# can `monkeypatch.setattr("relin.fit._executorFactory", ...)` to observe
# construction or to inject a recording executor.
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


def fit(
    ramps: Sequence[Ramp],
    model: Model | None = None,
    blockSize: tuple[int, int] = (512, 512),
    workers: int | None = None,
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

    effectiveWorkers = _resolveWorkerCount(workers, H, W)

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
        # compute on independent numpy arrays (no shared mutable state).
        # Note: ThreadPoolExecutor.submit has no back-pressure, so all
        # tiles' assembled (mTile, validTile) arrays coexist in memory
        # until their futures complete. On 4096x4096 with blockSize=(512,
        # 512), that is 64 tiles of roughly tile-sized float32 plus a
        # small bool array each — manageable on the reference workload
        # but worth keeping in mind if tile size grows.
        with _executorFactory(max_workers=effectiveWorkers) as executor:
            futures: dict[Future, tuple[int, int, int, int]] = {}
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
