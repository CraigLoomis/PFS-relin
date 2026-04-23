"""Smoke test: the package imports cleanly."""

def test_packageImports():
    import nirLinearity
    assert nirLinearity is not None


def test_modelsSubpackageImports():
    import nirLinearity.models
    assert nirLinearity.models is not None


def test_publicApiExports():
    import nirLinearity

    for attr in [
        "Ramp",
        "LinearizedRamp",
        "Diagnostics",
        "LinearityCorrection",
        "Model",
        "PolynomialModel",
        "fit",
        "apply",
        "applyFrame",
        "saveFits",
        "loadFits",
        "MASKED_BY_INPUT",
        "INSUFFICIENT_POINTS",
        "FIT_FAILED",
        "NON_MONOTONIC",
        "BORDER_PIX",
        "ABOVE_VALID_RANGE",
        "BELOW_VALID_RANGE",
    ]:
        assert hasattr(nirLinearity, attr), f"nirLinearity.{attr} missing"
