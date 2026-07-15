"""Input and output locations for the fitLinearity harness.

Lab ramps are read from a per-detector directory under the data root; every
artifact the harness writes lands under the output root, keyed by detector and
by a tag describing the fit configuration. Nothing is ever written back into
the data root.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("/work/cloomis/outputs/fitLinearity")
"""Root of the harness's persistent artifacts."""

DATA_ROOT_ENV = "FITLINEARITY_DATA"
"""Environment variable that overrides the default data root."""


def repoRoot() -> Path:
    """Return the checkout root (the directory holding ``python/``)."""
    return Path(__file__).resolve().parents[2]


def dataRoot() -> Path:
    """Return the root holding the per-detector lab ramp directories.

    ``$FITLINEARITY_DATA`` when set, otherwise ``jhu-data`` beside the checkout.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override)
    return repoRoot().parent / "jhu-data"


def inputDir(det: str, root: Path | None = None) -> Path:
    """Return the directory holding detector ``det``'s ``.npz`` ramps."""
    if not det:
        raise ValueError("detector id must be non-empty")
    base = root if root is not None else dataRoot()
    return Path(base) / det


def outputDir(det: str, cliTag: str, root: Path | None = None) -> Path:
    """Return (creating it) the output directory for one fit configuration."""
    if not det:
        raise ValueError("detector id must be non-empty")
    base = Path(root) if root is not None else DEFAULT_OUTPUT_ROOT
    out = base / det / cliTag
    out.mkdir(parents=True, exist_ok=True)
    return out
