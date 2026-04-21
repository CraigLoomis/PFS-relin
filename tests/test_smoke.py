"""Smoke test: the package imports cleanly."""

def test_packageImports():
    import relin
    assert relin is not None


def test_modelsSubpackageImports():
    import relin.models
    assert relin.models is not None


def test_publicApiExports():
    import relin

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
    ]:
        assert hasattr(relin, attr), f"relin.{attr} missing"
