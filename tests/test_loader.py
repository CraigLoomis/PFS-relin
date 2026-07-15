"""Tests for the harness's npz loader and photodiode correction."""

from __future__ import annotations

import numpy as np
import pytest
from fitLinearity.loader import loadCorrectedRamp, loadNpz


def _writeNpz(path, deltas, photodiode):
    np.savez(path, deltas=np.asarray(deltas, dtype=np.float32),
             photodiode=np.asarray(photodiode))
    return path


def testLoadNpzPrependsZeroReadAndAccumulates(tmp_path):
    deltas = np.full((3, 2, 2), 10.0)
    path = _writeNpz(tmp_path / "r.npz", deltas, np.ones(3))
    ramp, photodiode = loadNpz(path)
    assert ramp.reads.shape == (4, 2, 2)
    np.testing.assert_allclose(ramp.reads[:, 0, 0], [0.0, 10.0, 20.0, 30.0])
    assert photodiode.shape == (3,)


def testLoadNpzDropsLeadingPhotodiodeSampleWhenNPlusOne(tmp_path):
    deltas = np.full((3, 2, 2), 10.0)
    path = _writeNpz(tmp_path / "r.npz", deltas, [99.0, 1.0, 2.0, 3.0])
    _, photodiode = loadNpz(path)
    np.testing.assert_allclose(photodiode, [1.0, 2.0, 3.0])


def testLoadNpzPassesThroughEmptyPhotodiode(tmp_path):
    deltas = np.full((3, 2, 2), 10.0)
    path = _writeNpz(tmp_path / "r.npz", deltas, np.zeros(0))
    _, photodiode = loadNpz(path)
    assert photodiode.shape == (0,)


def testLoadNpzRejectsMismatchedPhotodiode(tmp_path):
    deltas = np.full((3, 2, 2), 10.0)
    path = _writeNpz(tmp_path / "r.npz", deltas, np.ones(7))
    with pytest.raises(ValueError, match="does not match nDeltas=3"):
        loadNpz(path)


def testLoadCorrectedRampNormalizesIlluminationDrift(tmp_path):
    # Flux halves at the third read; the photodiode sees the same drop, so the
    # corrected ramp must come out linear again.
    deltas = np.stack([
        np.full((2, 2), 10.0),
        np.full((2, 2), 10.0),
        np.full((2, 2), 5.0),
    ])
    path = _writeNpz(tmp_path / "r.npz", deltas, [1.0, 1.0, 0.5])
    ramp, photodiode = loadCorrectedRamp(path)
    np.testing.assert_allclose(ramp.reads[:, 0, 0], [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(photodiode, [1.0, 1.0, 0.5])


def testLoadCorrectedRampSkipsCorrectionWhenNoPhotodiodeRequested(tmp_path):
    deltas = np.stack([
        np.full((2, 2), 10.0),
        np.full((2, 2), 10.0),
        np.full((2, 2), 5.0),
    ])
    path = _writeNpz(tmp_path / "r.npz", deltas, [1.0, 1.0, 0.5])
    ramp, _ = loadCorrectedRamp(path, noPhotodiode=True)
    np.testing.assert_allclose(ramp.reads[:, 0, 0], [0.0, 10.0, 20.0, 25.0])


def testLoadCorrectedRampSkipsCorrectionWhenPhotodiodeAbsent(tmp_path):
    deltas = np.full((3, 2, 2), 10.0)
    path = _writeNpz(tmp_path / "r.npz", deltas, np.zeros(0))
    ramp, photodiode = loadCorrectedRamp(path)
    np.testing.assert_allclose(ramp.reads[:, 0, 0], [0.0, 10.0, 20.0, 30.0])
    assert photodiode.shape == (0,)
