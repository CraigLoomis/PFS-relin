"""Sanity checks that the synthetic-ramp fixtures behave as documented."""

from __future__ import annotations

import numpy as np


def test_tinyLinearRampBuildsCorrectly(tinyLinearRamp):
    ramp, truth = tinyLinearRamp
    N, H, W = ramp.reads.shape
    assert (N, H, W) == (6, 2, 3)
    # Cumulative reads grow linearly: reads[n] = rate * pixelScale * (n+1)
    perRead = truth["targetRate"] * truth["pixelScale"]
    for n in range(N):
        np.testing.assert_allclose(ramp.reads[n], perRead * (n + 1), rtol=1e-6)


def test_smallSyntheticRampBuildsCorrectly(smallSyntheticRamp):
    ramp, truth = smallSyntheticRamp
    N, H, W = ramp.reads.shape
    assert (N, H, W) == (29, 4, 5)
    # Verify the reconstructed polynomial: for each pixel and each read,
    # evaluate the true polynomial at the cumulative m and check it equals target.
    m = ramp.reads.astype(np.float64)  # (N, H, W)
    polyVal = (
        truth["c0"] + truth["c1"] * m + truth["c2"] * m ** 2
        + truth["c3"] * m ** 3 + truth["c4"] * m ** 4
    )
    target = truth["target"]
    for n in range(N):
        np.testing.assert_allclose(polyVal[n], target[n], rtol=1e-3, atol=1e-2)
