"""End-to-end integration test: fit -> save -> load -> apply."""

from __future__ import annotations

import numpy as np

import lsst.obs.pfs.h4Linearity as nirLinearity

def test_integrationEndToEnd(smallSyntheticRamp, tmp_path):
    ramp, truth = smallSyntheticRamp

    # Fit with a small block size to exercise the tiling path.
    correction = nirLinearity.fit([ramp], blockSize=(2, 3))

    # Save + load round trip.
    path = tmp_path / "correction.fits"
    nirLinearity.saveFits(path, correction)
    loaded = nirLinearity.loadFits(path)

    # Apply loaded correction to the original ramp.
    result = nirLinearity.apply(loaded, ramp)

    # The fit infers its own target rate R = median(reads[2] - reads[1]); for
    # a fixture whose pixels solve polynomial(m[n]) = rateTrue * n, the fit
    # recovers a SCALED polynomial so that polynomial_fit(m[n]) = R * n —
    # i.e. the correction's output trajectory is R * n, which generally
    # differs from the fixture's truth["target"] by a small scale factor.
    # Compare against the fit's self-inferred target grid, not the truth.
    fitRate = float(np.median(ramp.reads[2] - ramp.reads[1]))
    N = ramp.reads.shape[0]
    expectedCurve = fitRate * np.arange(N, dtype=np.float32)
    expected = np.broadcast_to(
        expectedCurve[:, None, None], ramp.reads.shape
    )
    residual = result.cumulativeLinear - expected
    # Residuals should be small for every pixel.
    rms = np.sqrt(np.mean(residual ** 2, axis=0))
    assert rms.max() < 1.0, f"max per-pixel RMS {rms.max()} too large"

    # No bad pixels, no out-of-range flags on the fitting data itself.
    assert (loaded.badPixelMask == 0).all()
    assert (result.badPixelMask == 0).all()

    # Summary is populated and sane. `io.py` preserves summary keys through
    # the FITS round-trip (HIERARCH cards for >8-char keys, and the original
    # Python key stashed in the comment for short keys), so we look up the
    # camelCase keys directly. We still use ``summary.get`` so the assertion
    # short-circuits cleanly if an upstream change ever renames keys.
    summary = loaded.diagnostics.summary
    assert summary.get("modelName") == "CHEBYSHEV"
    # Good-pixel fraction is 1.0 for this synthetic dataset.
    goodKeys = [k for k in summary if "good" in k.lower()]
    assert goodKeys
    for k in goodKeys:
        assert summary[k] == 1.0 or summary[k] == "1.0"
