"""Smoke test: the package imports cleanly."""

def test_packageImports():
    import relin
    assert relin is not None


def test_modelsSubpackageImports():
    import relin.models
    assert relin.models is not None
