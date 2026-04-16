"""Development convenience loaders. Production callers supply their own loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from relin.types import Ramp


def loadNpz(path: str | Path) -> tuple[Ramp, np.ndarray]:
    """Load a ``.npz`` with ``deltas`` and ``photodiode`` arrays.

    Returns a ``(Ramp, photodiode)`` pair. The caller is expected to apply
    the photodiode correction (e.g. ``deltas * photodiode[:, None, None]``)
    before passing the ramp into :func:`relin.fit.fit`.
    """
    path = Path(path)
    with np.load(path) as data:
        deltas = np.asarray(data["deltas"], dtype=np.float32)
        photodiode = np.asarray(data["photodiode"])
    return Ramp(deltas=deltas), photodiode
