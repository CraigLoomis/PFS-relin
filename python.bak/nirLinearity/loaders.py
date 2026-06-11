"""Development convenience loaders. Production callers supply their own loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Ramp


def loadNpz(path: str | Path) -> tuple[Ramp, np.ndarray]:
    """Load a ``.npz`` with ``deltas`` and ``photodiode`` arrays.

    The on-disk format stores per-read deltas with an implicit read0 = 0.
    This loader prepends the zero read and accumulates, yielding ``N+1``
    cumulative reads from ``N`` deltas. The returned photodiode array is
    canonicalized to length ``N`` (one sample per delta interval); if the
    on-disk array has ``N+1`` entries, the leading read-0 baseline sample
    is dropped. The caller is expected to apply the photodiode correction
    before passing the ramp into :func:`nirLinearity.fit.fit`.
    """
    path = Path(path)
    with np.load(path) as data:
        deltas = np.asarray(data["deltas"], dtype=np.float32)
        photodiode = np.asarray(data["photodiode"])
    nDeltas, h, w = deltas.shape
    if photodiode.shape[0] == 0:
        # No photodiode data recorded for this ramp; caller must skip the correction.
        pass
    elif photodiode.shape[0] == nDeltas + 1:
        photodiode = photodiode[1:]
    elif photodiode.shape[0] != nDeltas:
        raise ValueError(
            f"photodiode length {photodiode.shape[0]} does not match nDeltas={nDeltas} "
            f"(expected 0, {nDeltas}, or {nDeltas + 1})"
        )
    reads = np.empty((nDeltas + 1, h, w), dtype=np.float32)
    reads[0] = 0.0
    np.cumsum(deltas, axis=0, out=reads[1:])
    return Ramp(reads=reads), photodiode
