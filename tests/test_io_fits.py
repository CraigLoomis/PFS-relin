"""Tests for FITS save/load of LinearityCorrection."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from relin.fit import fit
from relin.io import loadFits, saveFits
from relin.models import PolynomialModel
from relin.types import Ramp


def _makeCorrection():
    rng = np.random.default_rng(0)
    H, W = 4, 5
    deltas = np.full((10, H, W), 100.0, dtype=np.float32)
    deltas += rng.normal(0.0, 0.1, size=deltas.shape).astype(np.float32)
    reads = np.cumsum(deltas, axis=0)
    return fit([Ramp(reads=reads)], model=PolynomialModel(order=2))


def test_saveLoadRoundTrip(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)

    loaded = loadFits(path)
    np.testing.assert_array_equal(loaded.coefficients, correction.coefficients)
    np.testing.assert_array_equal(loaded.fitMin, correction.fitMin)
    np.testing.assert_array_equal(loaded.fitMax, correction.fitMax)
    np.testing.assert_array_equal(loaded.badPixelMask, correction.badPixelMask)
    np.testing.assert_array_equal(
        loaded.diagnostics.residualRms, correction.diagnostics.residualRms
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.maxAbsResidual, correction.diagnostics.maxAbsResidual
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.nPointsUsed, correction.diagnostics.nPointsUsed
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.monotonic, correction.diagnostics.monotonic
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.conditionNumber, correction.diagnostics.conditionNumber
    )
    assert loaded.model.modelName == "CHEBYSHEV"
    assert loaded.model.order == 2


def test_saveFitsPrimaryIsHeaderOnly(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    with fits.open(path) as hdul:
        assert hdul[0].data is None
        assert hdul[0].header["MODEL"] == "CHEBYSHEV"
        assert "RELINVER" in hdul[0].header


def test_saveFitsImageHdusHaveChecksums(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    with fits.open(path) as hdul:
        for hdu in hdul[1:]:
            assert "CHECKSUM" in hdu.header
            assert "DATASUM" in hdu.header


def test_saveLoadRoundTripSummary(tmp_path):
    """All scalar summary keys survive the round-trip, including those
    longer than 8 characters (which share 8-char prefixes)."""
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    loaded = loadFits(path)
    # Every scalar entry in the original summary must appear in the loaded
    # summary with an equal value (floats allowed to round-trip via FITS).
    for key, value in correction.diagnostics.summary.items():
        if not isinstance(value, (int, float, bool, str)):
            continue
        assert key in loaded.diagnostics.summary, f"missing summary key {key!r}"
        if isinstance(value, float):
            assert float(loaded.diagnostics.summary[key]) == pytest.approx(value)
        else:
            assert loaded.diagnostics.summary[key] == value


def test_loadFitsUnknownModelRaises(tmp_path):
    correction = _makeCorrection()
    path = tmp_path / "correction.fits"
    saveFits(path, correction)
    # Corrupt the MODEL keyword.
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["MODEL"] = "NOSUCHMODEL"
        hdul.flush()
    with pytest.raises(ValueError):
        loadFits(path)


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
