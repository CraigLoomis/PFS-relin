"""Tests for loaders.loadNpz."""

from __future__ import annotations

import numpy as np

from lsst.obs.pfs.h4Linearity.loaders import loadNpz
from lsst.obs.pfs.h4Linearity.types import Ramp


def test_loadNpzReturnsRampAndPhotodiode(tmp_path):
    N, H, W = 5, 3, 4
    deltas = np.arange(N * H * W, dtype=np.float32).reshape(N, H, W)
    photodiode = np.array([1.0, 1.1, 1.05, 1.0, 0.95], dtype=np.float64)
    path = tmp_path / "ramp.npz"
    np.savez(path, deltas=deltas, photodiode=photodiode)

    ramp, pdio = loadNpz(path)
    assert isinstance(ramp, Ramp)
    # Loader prepends an implicit read0 = 0 and accumulates the on-disk deltas.
    assert ramp.reads.shape == (N + 1, H, W)
    np.testing.assert_array_equal(ramp.reads[0], 0)
    np.testing.assert_array_equal(ramp.reads[1:], np.cumsum(deltas, axis=0))
    assert ramp.validMask is None
    np.testing.assert_array_equal(pdio, photodiode)


def test_saturationModuleIsImportable():
    import lsst.obs.pfs.h4Linearity.saturation  # noqa: F401
