"""Tests for the sanity check's configuration tag."""

from __future__ import annotations

from fitLinearity.fitLinearity import SanityCheckConfig, cliTag


def _config(**overrides) -> SanityCheckConfig:
    defaults = dict(
        order=4, deviationLimit=None, deviationStart=0.5, saturationLevel=None,
        lowFluxFraction=0.5, saturationKnee=0.5, badLinearityMultiplier=5.0,
        noPhotodiode=False, seed=0, nplot=1000, plotFormat="png",
        fitrangeMin=None, fitrangeMax=None,
    )
    defaults.update(overrides)
    return SanityCheckConfig(**defaults)


def testDefaultsTagIsOrderOnly():
    assert cliTag(_config()) == "o4"


def testNoPhotodiodeTag():
    assert cliTag(_config(noPhotodiode=True)) == "o4_pdOff"


def testDeviationLimitTag():
    assert cliTag(_config(order=5, deviationLimit=0.45)) == "o5_dev0.45"


def testNonDefaultDeviationStartTag():
    assert cliTag(_config(deviationStart=0.3)) == "o4_ds0.3"


def testSaturationLevelTagIsInteger():
    assert cliTag(_config(saturationLevel=45000.0)) == "o4_sat45000"


def testDisabledGatesTag():
    tag = cliTag(_config(saturationKnee=None, badLinearityMultiplier=None))
    assert tag == "o4_kneeOff_blmOff"


def testEveryFieldTogether():
    tag = cliTag(_config(
        order=5, noPhotodiode=True, deviationLimit=0.45, deviationStart=0.3,
        saturationLevel=45000.0, lowFluxFraction=0.7, saturationKnee=0.6,
        badLinearityMultiplier=3.0,
    ))
    assert tag == "o5_pdOff_dev0.45_ds0.3_sat45000_lff0.7_knee0.6_blm3.0"


def testRateStabilityOffAddsNoTag():
    assert cliTag(_config(rateStability=False)) == "o4"


def testRateStabilityEnabledDefaultTag():
    assert cliTag(_config(rateStability=True)) == "o4_rs0.2"


def testRateStabilityNonDefaultThresholdTag():
    assert cliTag(_config(rateStability=True, rateStabilityThreshold=0.3)) == "o4_rs0.3"


def testRateStabilityNonDefaultFloorTag():
    assert cliTag(_config(rateStability=True, rateStabilityFloor=8.0)) == "o4_rs0.2_rsf8.0"


def testRateStabilityFloorIgnoredWhenDisabled():
    # Floor only appears when the gate is on.
    assert cliTag(_config(rateStability=False, rateStabilityFloor=8.0)) == "o4"
