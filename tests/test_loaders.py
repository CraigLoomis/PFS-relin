"""Tests for loaders.loadNpz."""

from __future__ import annotations

import numpy as np

from nirLinearity.loaders import loadNpz
from nirLinearity.types import Ramp


def test_loadNpzReturnsRampAndPhotodiode(tmp_path):
    N, H, W = 5, 3, 4
    deltas = np.arange(N * H * W, dtype=np.float32).reshape(N, H, W)
    photodiode = np.array([1.0, 1.1, 1.05, 1.0, 0.95], dtype=np.float64)
    path = tmp_path / "ramp.npz"
    np.savez(path, deltas=deltas, photodiode=photodiode)

    ramp, pdio = loadNpz(path)
    assert isinstance(ramp, Ramp)
    # Loader converts on-disk deltas to cumulative reads.
    np.testing.assert_array_equal(ramp.reads, np.cumsum(deltas, axis=0))
    assert ramp.validMask is None
    np.testing.assert_array_equal(pdio, photodiode)


def test_saturationModuleIsImportable():
    import nirLinearity.saturation  # noqa: F401
