"""Loading of lab ``.npz`` ramps for the harness.

Handles the photodiode-shape variants that turn up in the lab files (``N``,
``N+1``, or length-0), which upstream ``h4Linearity.loaders`` does not, and
applies the standard illumination-drift photodiode correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lsst.obs.pfs.h4Linearity.types import Ramp


def loadNpz(path: str | Path) -> tuple[Ramp, np.ndarray]:
    """Load a ``.npz`` with ``deltas`` and ``photodiode`` arrays.

    The on-disk format stores per-read deltas with an implicit read0 = 0.
    This loader prepends the zero read and accumulates, yielding ``N+1``
    cumulative reads from ``N`` deltas. The returned photodiode array is
    canonicalized to length ``N`` (one sample per delta interval); if the
    on-disk array has ``N+1`` entries, the leading read-0 baseline sample
    is dropped. A length-0 photodiode array (no PD recorded for this
    ramp) is passed through; callers are expected to skip the photodiode
    correction in that case.
    """
    path = Path(path)
    with np.load(path) as data:
        deltas = np.asarray(data["deltas"], dtype=np.float32)
        photodiode = np.asarray(data["photodiode"])
    nDeltas, h, w = deltas.shape
    if photodiode.shape[0] == 0:
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


def loadCorrectedRamp(
    path: str | Path, noPhotodiode: bool = False
) -> tuple[Ramp, np.ndarray]:
    """Load a ramp and normalize it for illumination drift.

    Each read's increment is scaled by the photodiode ratio relative to the
    first sample, then re-accumulated, so a ramp taken under a drifting lamp
    reads as one taken at the first read's illumination. The correction is
    skipped when ``noPhotodiode`` is set or when the file records no photodiode
    samples. Returns the corrected ramp and the raw photodiode array.
    """
    ramp, photodiode = loadNpz(path)
    if noPhotodiode or photodiode.shape[0] == 0:
        return ramp, photodiode

    scale = (photodiode[0] / photodiode).astype(np.float32)  # (N,)
    deltas = np.diff(ramp.reads, axis=0)  # (N, H, W)
    correctedReads = np.empty_like(ramp.reads)
    correctedReads[0] = 0.0
    np.cumsum(deltas * scale[:, None, None], axis=0, out=correctedReads[1:])
    return Ramp(reads=correctedReads), photodiode
