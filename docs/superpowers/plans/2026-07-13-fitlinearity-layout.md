# fitLinearity Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the harness to `fitLinearity`, turn `examples/` into a package at `python/fitLinearity/` with thin `bin/` wrappers, read inputs from `jhu-data/` by detector id, write outputs under `/work/cloomis/outputs/fitLinearity/`, and resolve `obs_pfs` / `drp_stella` through EUPS instead of a hardcoded `pythonpath`.

**Architecture:** All importable logic moves into `python/fitLinearity/`. A new `paths.py` is the single owner of the input and output roots. Each `bin/` script owns its own `argparse` parser, builds a config object, resolves paths through `paths.py`, and calls into the package. The photodiode illumination-drift correction — currently copy-pasted into `sanity_check.py` and both benchmarks — is consolidated into `loader.loadCorrectedRamp`.

**Tech Stack:** Python 3.12, numpy, astropy, matplotlib, pytest. Upstream `lsst.obs.pfs.h4Linearity` from `../PIPE2D-1844/obs_pfs`, activated with EUPS.

## Global Constraints

- **camelCase** for function names, method names, and variables — tests included. Classes PascalCase, constants UPPER_SNAKE_CASE. Do not "correct" existing camelCase to snake_case.
- Package source lives under `python/`, not `src/`. Tests at `tests/`.
- Comments and docstrings describe what the code does **now**. No "was", "previously", "moved from", "used to live in `examples/`".
- Never activate a local checkout via `PYTHONPATH` — EUPS `setup` only. The single `pythonpath = ["python"]` entry in `pyproject.toml` is this repo's own source tree and is the sole exception.
- Line length 110 (ruff). Rule set `["E", "F", "W", "I"]`; pep8-naming (`N`) stays off.
- Input root default: `../jhu-data` relative to the checkout root, overridden by `$FITLINEARITY_DATA`.
- Output root default: `/work/cloomis/outputs/fitLinearity`, overridden by `--out-root`.
- Every shell that runs tests or `bin/` scripts must first run:
  ```bash
  source /work/stack/loadLSST.bash
  setup pfs_pipe2d
  setup -j -r ../PIPE2D-1844/obs_pfs
  setup -j -r ../PIPE2D-1844/drp_stella
  ```
  Use the LSST-env `python` directly. Do **not** use `uv run` — it builds an isolated venv that cannot see the LSST conda site-packages.

---

### Task 1: Project rename and package skeleton

Rename the project, create the empty package tree, remove the dead `python.bak/`, and cut the machine-specific `pythonpath` entry.

**Files:**
- Modify: `pyproject.toml`
- Create: `python/fitLinearity/__init__.py`
- Create: `python/fitLinearity/benchmarks/__init__.py`
- Delete: `python.bak/` (whole tree)

**Interfaces:**
- Consumes: nothing.
- Produces: an importable empty package `fitLinearity`; `pyproject.toml` with `pythonpath = ["python"]` and no absolute paths.

- [ ] **Step 1: Delete the dead package copy**

```bash
git rm -r --quiet python.bak
```

- [ ] **Step 2: Rewrite pyproject.toml**

Replace the whole file with:

```toml
[project]
name = "fitLinearity"
version = "0.1.0"
description = "Validation harness for the lsst.obs.pfs.h4Linearity package"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "astropy>=6.0",
]

[dependency-groups]
dev = [
    "matplotlib>=3.10.8",
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.5",
]

[tool.uv]
# The package under test (lsst.obs.pfs.h4Linearity) is an EUPS product in a
# sibling checkout, not a pip dependency; uv should not try to build/install us.
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
pythonpath = ["python"]

[tool.ruff]
line-length = 110

[tool.ruff.lint]
# Keep the default E/F/W/I rules; do NOT enable pep8-naming ("N") rules
# because this project uses camelCase for functions/variables.
select = ["E", "F", "W", "I"]
```

The `pythonpath` list carries `"python"` only. `obs_pfs` and `drp_stella` resolve through EUPS.

- [ ] **Step 3: Create the package `__init__.py` files**

`python/fitLinearity/__init__.py`:

```python
"""Validation harness for lsst.obs.pfs.h4Linearity.

Loaders, synthetic-ramp generation, the sanity check, and the fit benchmarks.
The linearity fit and apply implementations themselves live upstream in
``lsst.obs.pfs.h4Linearity``.
"""
```

`python/fitLinearity/benchmarks/__init__.py`:

```python
"""Wall-clock benchmarks for ``lsst.obs.pfs.h4Linearity.fit``."""
```

- [ ] **Step 4: Verify the package imports**

Run (in an EUPS-setup shell):
```bash
python -c "import sys; sys.path.insert(0, 'python'); import fitLinearity; print(fitLinearity.__doc__.splitlines()[0])"
```
Expected: `Validation harness for lsst.obs.pfs.h4Linearity.`

- [ ] **Step 5: Verify the existing tests still pass**

Run: `pytest`
Expected: all tests PASS. They import only `lsst.obs.pfs.h4Linearity`, which now resolves through EUPS rather than the deleted `pythonpath` entry. A `ModuleNotFoundError: lsst.obs.pfs` here means the EUPS setup chain in Global Constraints was not run.

- [ ] **Step 6: Commit**

```bash
git add -A pyproject.toml python/ python.bak
git commit -m "Rename project to fitLinearity and add the package skeleton

Resolve obs_pfs and drp_stella through EUPS rather than a hardcoded
absolute pythonpath, and drop the dead python.bak copy of the package
that now lives upstream."
```

---

### Task 2: paths.py

The single owner of the input and output roots.

**Files:**
- Create: `python/fitLinearity/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Consumes: `fitLinearity` package from Task 1.
- Produces:
  - `DEFAULT_OUTPUT_ROOT: Path` — `/work/cloomis/outputs/fitLinearity`
  - `repoRoot() -> Path`
  - `dataRoot() -> Path`
  - `inputDir(det: str, root: Path | None = None) -> Path`
  - `outputDir(det: str, cliTag: str, root: Path | None = None) -> Path`

The `root` parameters are deliberately **not** named `dataRoot` / `outRoot`: a
parameter named `dataRoot` would shadow the module-level `dataRoot()` function.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
"""Tests for input/output path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from fitLinearity.paths import (
    DEFAULT_OUTPUT_ROOT,
    dataRoot,
    inputDir,
    outputDir,
    repoRoot,
)


def testRepoRootHoldsThePackage():
    assert (repoRoot() / "python" / "fitLinearity" / "paths.py").is_file()


def testDataRootDefaultsToSiblingJhuData(monkeypatch):
    monkeypatch.delenv("FITLINEARITY_DATA", raising=False)
    assert dataRoot() == repoRoot().parent / "jhu-data"


def testDataRootHonorsEnvVar(monkeypatch, tmp_path):
    monkeypatch.setenv("FITLINEARITY_DATA", str(tmp_path))
    assert dataRoot() == tmp_path


def testInputDirIsRootSlashDet(monkeypatch, tmp_path):
    monkeypatch.setenv("FITLINEARITY_DATA", str(tmp_path))
    assert inputDir("18734") == tmp_path / "18734"


def testInputDirAcceptsExplicitRoot(tmp_path):
    assert inputDir("18660", root=tmp_path) == tmp_path / "18660"


def testDefaultOutputRoot():
    assert DEFAULT_OUTPUT_ROOT == Path("/work/cloomis/outputs/fitLinearity")


def testOutputDirIsDetSlashTagAndIsCreated(tmp_path):
    out = outputDir("18734", "o5_dev0.45", root=tmp_path)
    assert out == tmp_path / "18734" / "o5_dev0.45"
    assert out.is_dir()


def testOutputDirIsIdempotent(tmp_path):
    outputDir("18734", "o4", root=tmp_path)
    out = outputDir("18734", "o4", root=tmp_path)
    assert out.is_dir()


def testOutputDirNeverWritesUnderTheDataRoot(monkeypatch, tmp_path):
    dataDir = tmp_path / "jhu-data"
    outRoot = tmp_path / "outputs"
    monkeypatch.setenv("FITLINEARITY_DATA", str(dataDir))
    out = outputDir("18734", "o4", root=outRoot)
    assert dataDir not in out.parents
    assert not dataDir.exists()


def testInputDirRejectsEmptyDet():
    with pytest.raises(ValueError, match="detector id must be non-empty"):
        inputDir("")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fitLinearity.paths'`

- [ ] **Step 3: Write paths.py**

```python
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
```

The override parameters are named `root`, not `dataRoot` / `outRoot`: a
parameter named `dataRoot` would shadow the module-level `dataRoot()` function
and force an awkward `globals()` lookup to call it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_paths.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint**

Run: `ruff check python/fitLinearity/paths.py tests/test_paths.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add python/fitLinearity/paths.py tests/test_paths.py
git commit -m "Add paths module owning the data and output roots

Inputs are addressed by detector id under the data root (jhu-data by
default, \$FITLINEARITY_DATA to override); outputs land under
/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/."
```

---

### Task 3: loader.py, with the photodiode correction consolidated

`examples/_loader.py` becomes `python/fitLinearity/loader.py`. The
illumination-drift photodiode correction is currently duplicated verbatim in
`sanity_check.py` and both benchmarks; it moves here as `loadCorrectedRamp`.

**Files:**
- Create: `python/fitLinearity/loader.py`
- Delete: `examples/_loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `lsst.obs.pfs.h4Linearity.types.Ramp`.
- Produces:
  - `loadNpz(path: str | Path) -> tuple[Ramp, np.ndarray]` — unchanged behavior
  - `loadCorrectedRamp(path: str | Path, noPhotodiode: bool = False) -> tuple[Ramp, np.ndarray]` — returns the photodiode-corrected ramp and the raw photodiode array

- [ ] **Step 1: Write the failing tests**

Create `tests/test_loader.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fitLinearity.loader'`

- [ ] **Step 3: Write loader.py**

Copy `examples/_loader.py` to `python/fitLinearity/loader.py` verbatim, then
replace the module docstring and append `loadCorrectedRamp`. Full file:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: 7 passed.

- [ ] **Step 5: Remove the old loader**

```bash
git rm --quiet examples/_loader.py
```

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all PASS. `tests/test_loaders.py` (plural — the upstream loader's tests) is untouched and must still pass alongside the new `tests/test_loader.py`.

- [ ] **Step 7: Commit**

```bash
git add python/fitLinearity/loader.py tests/test_loader.py examples/_loader.py
git commit -m "Move the npz loader into the package and fold in the photodiode correction

loadCorrectedRamp applies the illumination-drift normalization that the
sanity check and both benchmarks each carried their own copy of."
```

---

### Task 4: syntheticRamp and sanityCheck, plus bin/sanityCheck.py

The 1021-line `examples/sanity_check.py` splits into a library module exposing
`runFit` / `runPlot` and a thin `bin/` wrapper that owns the parser.

**Files:**
- Create: `python/fitLinearity/syntheticRamp.py` (move of `examples/syntheticRamp.py`, unchanged content)
- Create: `python/fitLinearity/sanityCheck.py`
- Create: `bin/sanityCheck.py`
- Delete: `examples/syntheticRamp.py`, `examples/sanity_check.py`
- Test: `tests/test_sanityCheck.py`

**Interfaces:**
- Consumes: `fitLinearity.loader.loadCorrectedRamp`, `fitLinearity.paths.inputDir` / `outputDir` / `DEFAULT_OUTPUT_ROOT`.
- Produces:
  - `SanityCheckConfig` dataclass with fields: `order: int`, `deviationLimit: float | None`, `deviationStart: float`, `saturationLevel: float | None`, `lowFluxFraction: float`, `saturationKnee: float | None`, `badLinearityMultiplier: float | None`, `noPhotodiode: bool`, `seed: int`, `nplot: int`, `plotFormat: str`, `fitrangeMin: float | None`, `fitrangeMax: float | None`
  - `cliTag(config: SanityCheckConfig) -> str`
  - `runFit(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None`
  - `runPlot(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None`

- [ ] **Step 1: Write the failing test for cliTag**

`cliTag` is the one piece of `sanityCheck.py` with pure, testable logic, and it
is what decides the output directory name — worth pinning. Create
`tests/test_sanityCheck.py`:

```python
"""Tests for the sanity check's configuration tag."""

from __future__ import annotations

from fitLinearity.sanityCheck import SanityCheckConfig, cliTag


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_sanityCheck.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fitLinearity.sanityCheck'`

- [ ] **Step 3: Move syntheticRamp into the package**

```bash
git mv examples/syntheticRamp.py python/fitLinearity/syntheticRamp.py
```

Then edit its docstring — the current first line says "shared by the example
benchmarks", which no longer describes where it lives:

```python
"""Synthetic ramp generation for the fit benchmarks.

Produces a 4096x4096x30 cumulative-DN cube with plausible IR-detector
behavior, so benchmarks can run without a lab NPZ being on disk.
"""
```

Leave the body of `syntheticRamp()` untouched.

- [ ] **Step 4: Move sanity_check.py into the package**

```bash
git mv examples/sanity_check.py python/fitLinearity/sanityCheck.py
```

- [ ] **Step 5: Restructure sanityCheck.py**

Make exactly these changes to `python/fitLinearity/sanityCheck.py`. Every other
function in the file (`_t`, `_applyBridge`, `_plotDiagnostic`, and the rest of
the plotting helpers) keeps its current body.

**5a.** Replace the module docstring and imports at the top of the file:

```python
"""Fit, save, reload, apply, and diagnose a linearity correction on a lab ramp.

Runs the full chain on a real 4096x4096 up-the-ramp exposure and reports
residuals and per-population diagnostics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import lsst.obs.pfs.h4Linearity as nirLinearity
from fitLinearity.loader import loadCorrectedRamp
from lsst.obs.pfs.h4Linearity.types import (
    BORDER_PIX,
    FIT_FAILED,
    HIGH_FIT_RESIDUAL,
    INSUFFICIENT_POINTS,
    MASKED_BY_INPUT,
    NON_MONOTONIC,
    Ramp,
)
```

`argparse` stays imported here only if a helper still needs it; if ruff reports
it unused after 5d, delete the import. `Ramp` likewise — keep it only if a
remaining function annotates with it.

**5b.** Add the config dataclass and `cliTag` immediately after the imports:

```python
@dataclass
class SanityCheckConfig:
    """Fit and plot parameters for one sanity-check run."""

    order: int
    deviationLimit: float | None
    deviationStart: float
    saturationLevel: float | None
    lowFluxFraction: float
    saturationKnee: float | None
    badLinearityMultiplier: float | None
    noPhotodiode: bool
    seed: int
    nplot: int
    plotFormat: str
    fitrangeMin: float | None
    fitrangeMax: float | None


def cliTag(config: SanityCheckConfig) -> str:
    """Build the output-directory name describing this run's fit configuration.

    Only non-default parameters appear, so a default run tags as ``o4``.
    """
    tag = f"o{config.order}"
    if config.noPhotodiode:
        tag += "_pdOff"
    if config.deviationLimit is not None:
        tag += f"_dev{config.deviationLimit}"
    if config.deviationStart != 0.5:
        tag += f"_ds{config.deviationStart}"
    if config.saturationLevel is not None:
        tag += f"_sat{int(config.saturationLevel)}"
    if config.lowFluxFraction != 0.5:
        tag += f"_lff{config.lowFluxFraction}"
    if config.saturationKnee != 0.5:
        tag += f"_knee{config.saturationKnee}" if config.saturationKnee is not None else "_kneeOff"
    if config.badLinearityMultiplier != 5.0:
        tag += (
            f"_blm{config.badLinearityMultiplier}"
            if config.badLinearityMultiplier is not None
            else "_blmOff"
        )
    return tag
```

This is the tag logic from the old `main()` (the `cliTag` local), moved verbatim
except that `pdTag` is inlined as the `noPhotodiode` branch.

**5c.** Replace `_loadInputRamp` with a version that takes the input directory
and delegates the photodiode correction to the loader:

```python
def _loadInputRamp(inputDir: Path, noPhotodiode: bool) -> tuple[Ramp, Path]:
    """Load the single ``.npz`` ramp in ``inputDir``, photodiode-corrected."""
    npzFiles = sorted(inputDir.glob("*.npz"))
    if not npzFiles:
        raise FileNotFoundError(f"No .npz files found in {inputDir}")
    dataPath = npzFiles[0]
    t0 = time.perf_counter()
    print(f"Loading {dataPath} ...", flush=True)
    correctedRamp, photodiode = loadCorrectedRamp(dataPath, noPhotodiode=noPhotodiode)
    if photodiode.shape[0] and not noPhotodiode:
        scale = photodiode[0] / photodiode
        print(
            f"  photodiode scale range: min={scale.min():.6f} max={scale.max():.6f}",
            flush=True,
        )
    _t("load + photodiode correction", t0)
    return correctedRamp, dataPath
```

The inline `np.diff` / `np.cumsum` block that used to sit here is gone — it is
`loadCorrectedRamp`.

**5d.** Split `main()` into `runFit` and `runPlot`. Delete `main()`, the
`argparse` parser, the `if __name__ == "__main__":` block, the `_buildTag`
helper (its FITS-header-derived tag is unused now that `cliTag` owns naming),
and the `outDir` / `fitsPath` / `dataDir` / `detName` derivation. In their place:

```python
def runFit(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None:
    """Fit the ramp in ``inputDir`` and write the correction FITS to ``outDir``."""
    det = outDir.parent.name
    fitsPath = outDir / f"{det}_linearity.fits"
    ...
```

and

```python
def runPlot(config: SanityCheckConfig, inputDir: Path, outDir: Path) -> None:
    """Plot diagnostics for the correction already written to ``outDir``."""
    det = outDir.parent.name
    fitsPath = outDir / f"{det}_linearity.fits"
    ...
```

For each, take the body of the old `if args.fit:` / `if args.plot:` block
verbatim and apply these mechanical substitutions throughout:

| old | new |
|---|---|
| `args.order` | `config.order` |
| `args.deviation_limit` | `config.deviationLimit` |
| `args.deviation_start` | `config.deviationStart` |
| `args.saturation_level` | `config.saturationLevel` |
| `args.low_flux_fraction` | `config.lowFluxFraction` |
| `saturationKnee` (local) | `config.saturationKnee` |
| `badLinearityMultiplier` (local) | `config.badLinearityMultiplier` |
| `args.no_photodiode` | `config.noPhotodiode` |
| `args.seed` | `config.seed` |
| `args.nplot` | `config.nplot` |
| `args.plot_format` | `config.plotFormat` |
| `args.fitrange_min` | `config.fitrangeMin` |
| `args.fitrange_max` | `config.fitrangeMax` |
| `dataDir` (as ramp source) | `inputDir` |
| `detId` / `detName` | `det` |
| `outDir.mkdir(exist_ok=True)` | *delete — `paths.outputDir` already created it* |

The `-1 means None` mapping for `saturationKnee` and `badLinearityMultiplier`
is **not** done here — the `bin/` script does it when building the config, so
the package sees only `float | None`.

- [ ] **Step 6: Run the cliTag tests**

Run: `pytest tests/test_sanityCheck.py -v`
Expected: 7 passed.

- [ ] **Step 7: Write bin/sanityCheck.py**

```python
#!/usr/bin/env python
"""Run the linearity sanity check on one detector's lab ramp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity import paths  # noqa: E402
from fitLinearity.sanityCheck import SanityCheckConfig, cliTag, runFit, runPlot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", type=str, required=True,
                        help="Detector id, naming the input subdirectory (e.g. 18734)")
    parser.add_argument("--data-root", type=str, default=None,
                        help=f"Root holding the per-detector ramp dirs "
                             f"(default: {paths.dataRoot()})")
    parser.add_argument("--out-root", type=str, default=None,
                        help=f"Root for output artifacts (default: {paths.DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--fit", action="store_true",
                        help="Fit and save a linearity correction FITS file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots (reads existing FITS + input data)")
    parser.add_argument("--plot-format", type=str, default="png",
                        help="Plot file format: png, pdf, svg (default: png)")
    parser.add_argument("--nplot", type=int, default=1000,
                        help="Number of pixels to plot (default: 1000)")
    parser.add_argument("--deviation-limit", type=float, default=None,
                        help="Fractional deviation limit for the fit range (default: None)")
    parser.add_argument("--deviation-start", type=float, default=0.5,
                        help="Fraction of the ramp at which deviation is measured (default: 0.5)")
    parser.add_argument("--order", type=int, default=4,
                        help="Polynomial order (default: 4)")
    parser.add_argument("--saturation-level", type=float, default=None,
                        help="Absolute saturation level in DN (default: None)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for pixel sampling (default: 0)")
    parser.add_argument("--no-photodiode", action="store_true",
                        help="Skip the illumination-drift photodiode correction")
    parser.add_argument("--low-flux-fraction", type=float, default=0.5,
                        help="Low-flux rejection fraction (default: 0.5)")
    parser.add_argument("--saturation-knee", type=float, default=0.5,
                        help="Per-pixel saturation knee (default: 0.5; -1 to disable)")
    parser.add_argument("--bad-linearity-multiplier", type=float, default=5.0,
                        help="Flag HIGH_FIT_RESIDUAL when residualRms > multiplier × median(good "
                             "residualRms) (default: 5.0; -1 to disable)")
    parser.add_argument("--fitrange-min", type=float, default=None,
                        help="Lower bound (DN) for the fit-range histogram (default: auto p0.05)")
    parser.add_argument("--fitrange-max", type=float, default=None,
                        help="Upper bound (DN) for the fit-range histogram (default: auto p99.95)")
    args = parser.parse_args()

    if not args.fit and not args.plot:
        args.fit = True

    config = SanityCheckConfig(
        order=args.order,
        deviationLimit=args.deviation_limit,
        deviationStart=args.deviation_start,
        saturationLevel=args.saturation_level,
        lowFluxFraction=args.low_flux_fraction,
        # CLI value -1 disables the gate; the API spells that None.
        saturationKnee=None if args.saturation_knee < 0 else args.saturation_knee,
        badLinearityMultiplier=(
            None if args.bad_linearity_multiplier < 0 else args.bad_linearity_multiplier
        ),
        noPhotodiode=args.no_photodiode,
        seed=args.seed,
        nplot=args.nplot,
        plotFormat=args.plot_format,
        fitrangeMin=args.fitrange_min,
        fitrangeMax=args.fitrange_max,
    )

    dataRoot = Path(args.data_root) if args.data_root else None
    outRoot = Path(args.out_root) if args.out_root else None
    inputDir = paths.inputDir(args.det, root=dataRoot)
    if not inputDir.is_dir():
        parser.error(f"input directory does not exist: {inputDir}")
    outDir = paths.outputDir(args.det, cliTag(config), root=outRoot)

    print(f"  in : {inputDir}", flush=True)
    print(f"  out: {outDir}", flush=True)

    if args.fit:
        runFit(config, inputDir, outDir)
    if args.plot:
        runPlot(config, inputDir, outDir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
chmod +x bin/sanityCheck.py
```

The `sys.path.insert` is what lets the script find its own package without an
install step. It points at this repo's `python/`, not at any external checkout.

- [ ] **Step 8: Verify the CLI wiring without running a fit**

Run: `bin/sanityCheck.py --help`
Expected: the usage block prints, listing `--det`, `--data-root`, `--out-root`,
and the fit options; the `--data-root` help line shows the resolved default
(`.../linfit/jhu-data`).

Run: `bin/sanityCheck.py --det 99999 --fit`
Expected: exit 2, `error: input directory does not exist: .../jhu-data/99999`

- [ ] **Step 9: Run the full suite and lint**

Run: `pytest && ruff check python bin tests`
Expected: all tests PASS, `All checks passed!`

- [ ] **Step 10: Commit**

```bash
git add -A python/fitLinearity bin/sanityCheck.py tests/test_sanityCheck.py examples/
git commit -m "Move the sanity check into the package behind a thin bin/ wrapper

sanityCheck exposes runFit/runPlot over a SanityCheckConfig; bin/sanityCheck.py
owns the argument parser and resolves the input and output directories through
paths. Outputs land under the output root rather than inside the data dir."
```

---

### Task 5: Benchmarks

**Files:**
- Create: `python/fitLinearity/benchmarks/fitThreading.py`, `python/fitLinearity/benchmarks/fitBlocksize.py`
- Create: `bin/benchmarkThreading.py`, `bin/benchmarkBlocksize.py`
- Delete: `examples/benchmark_fit_threading.py`, `examples/benchmark_fit_blocksize.py`, and the now-empty `examples/`

**Interfaces:**
- Consumes: `fitLinearity.loader.loadCorrectedRamp`, `fitLinearity.syntheticRamp.syntheticRamp`.
- Produces:
  - `fitThreading.runBenchmark(dataPath: Path | None) -> int`
  - `fitBlocksize.runBenchmark(dataPath: Path | None, sizes: list[int], trials: int, workers: int) -> int`

- [ ] **Step 1: Move the benchmark modules**

```bash
git mv examples/benchmark_fit_threading.py python/fitLinearity/benchmarks/fitThreading.py
git mv examples/benchmark_fit_blocksize.py python/fitLinearity/benchmarks/fitBlocksize.py
```

- [ ] **Step 2: Rewrite each module's head and entry point**

In **both** files:

- Replace `from _loader import loadNpz` with `from fitLinearity.loader import loadCorrectedRamp`.
- Replace `from syntheticRamp import syntheticRamp` with `from fitLinearity.syntheticRamp import syntheticRamp`.
- Delete the `Usage:` block from the module docstring — it names `uv run python examples/...`, which is neither the path nor the interpreter now. State what the benchmark measures and nothing more.
- Delete `import argparse` and the `main()` parser; rename `main()` to `runBenchmark` with the signature in **Interfaces** above, taking what it used to read off `args`.
- Delete the `if __name__ == "__main__":` block.
- Replace the ramp-loading branch — in both files it is the same eight lines of `loadNpz` + `np.diff` + `np.cumsum` — with:

```python
    if dataPath is None:
        print("No --data-path provided; generating synthetic ramp ...", flush=True)
        correctedRamp = syntheticRamp()
    else:
        dataPath = Path(dataPath)
        if not dataPath.exists():
            print(f"Data file missing: {dataPath}")
            return 1
        print(f"Loading {dataPath} ...", flush=True)
        correctedRamp, _ = loadCorrectedRamp(dataPath)
```

- Drop the `Ramp` and `np` imports if ruff then reports them unused.

Keep `_parseSizes` in `fitBlocksize.py` — but it raises
`argparse.ArgumentTypeError`, so move it to `bin/benchmarkBlocksize.py`, where
`argparse` now lives.

- [ ] **Step 3: Write bin/benchmarkThreading.py**

```python
#!/usr/bin/env python
"""Benchmark fit() wall-clock across worker counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity.benchmarks.fitThreading import runBenchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 ramp is generated",
    )
    args = parser.parse_args()
    return runBenchmark(Path(args.data_path) if args.data_path else None)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write bin/benchmarkBlocksize.py**

```python
#!/usr/bin/env python
"""Benchmark fit() wall-clock across blockSize values.

Defaults probe a geometric sweep [32, 64, 128, 256, 512] with 2 trials, which
takes 5-10 minutes on a typical workstation and is enough to pick a reasonable
blockSize. Tune with the same worker count you will use in production — the
optimum shifts with thread count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fitLinearity.benchmarks.fitBlocksize import runBenchmark  # noqa: E402


def _parseSizes(s: str) -> list[int]:
    sizes = [int(x) for x in s.split(",") if x.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("--sizes must list at least one integer")
    if any(b < 1 for b in sizes):
        raise argparse.ArgumentTypeError("--sizes values must be >= 1")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to a lab NPZ ramp; if omitted, a synthetic 4096x4096x30 ramp is generated",
    )
    parser.add_argument("--sizes", type=_parseSizes, default=[32, 64, 128, 256, 512],
                        help="Comma-separated blockSize values (default: 32,64,128,256,512)")
    parser.add_argument("--trials", type=int, default=2,
                        help="Trials per blockSize (default: 2)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker threads (default: 8)")
    args = parser.parse_args()
    return runBenchmark(
        Path(args.data_path) if args.data_path else None,
        sizes=args.sizes, trials=args.trials, workers=args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

Before writing this file, open the old `examples/benchmark_fit_blocksize.py`
parser and confirm the `--trials` and `--workers` defaults above match it; if
they differ, the old file's values win.

```bash
chmod +x bin/benchmarkThreading.py bin/benchmarkBlocksize.py
```

- [ ] **Step 5: Remove the empty examples directory**

```bash
git status --short examples/   # expect nothing left
rmdir examples 2>/dev/null || true
```

- [ ] **Step 6: Verify both benchmarks wire up**

Run: `bin/benchmarkThreading.py --help && bin/benchmarkBlocksize.py --help`
Expected: both usage blocks print, no import errors.

- [ ] **Step 7: Lint and test**

Run: `pytest && ruff check python bin tests`
Expected: all PASS, `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add -A python/fitLinearity/benchmarks bin/ examples/
git commit -m "Move the fit benchmarks into the package behind bin/ wrappers

Both benchmarks now take their ramp from loadCorrectedRamp instead of each
carrying a copy of the photodiode correction."
```

---

### Task 6: Documentation, gitignore, and end-to-end verification

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `.gitignore`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing importable.

- [ ] **Step 1: Update .gitignore**

The `examples/**` patterns name a directory that no longer exists, and outputs
now live outside the repo entirely. Replace the data and plot sections:

```
# Large data files — keep out of git
*.npz
*.fits
*.h5

# uv / venv
.venv/
.python-version
uv.lock

# Python build / cache
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.pytest_cache/
.coverage
htmlcov/

# Editor / OS
.DS_Store
.vscode/
.idea/
*.swp

# Claude Code session state
.claude/
```

The plot-output section goes away: diagnostics land under
`/work/cloomis/outputs/fitLinearity/`, outside the working tree.

- [ ] **Step 2: Rewrite CLAUDE.md**

```markdown
# fitLinearity

Validation harness for `lsst.obs.pfs.h4Linearity` — per-pixel polynomial
nonlinearity correction for IR-detector up-the-ramp data. The fit and apply
implementations live upstream; this repo holds the loaders, the sanity check,
and the benchmarks.

## Conventions

- **camelCase** for Python function names, method names, and variable names — not snake_case. Applies to tests too. Classes remain PascalCase. Constants remain UPPER_SNAKE_CASE. Module filenames stay short/single-word where possible; use camelCase only for compound module names. Do not "correct" existing camelCase to snake_case.
- Package source lives at `python/fitLinearity/` (not `src/`). Executables at `bin/`, tests at `tests/`.

## Environment

The harness imports `lsst.obs.pfs.h4Linearity`, which pulls in the LSST stack.
Every shell that runs tests or `bin/` scripts must set up EUPS first:

```bash
source /work/stack/loadLSST.bash
setup pfs_pipe2d
setup -j -r ../PIPE2D-1844/obs_pfs
setup -j -r ../PIPE2D-1844/drp_stella
```

Use the LSST-env `python` directly. **Do not use `uv run`** — it builds an
isolated venv from `pyproject.toml` alone and cannot see the LSST conda
site-packages. Never put a local checkout on `PYTHONPATH`; go through EUPS.

## Data and outputs

- Inputs: `../jhu-data/<det>/*.npz`, addressed by `--det` (override the root with
  `--data-root` or `$FITLINEARITY_DATA`).
- Outputs: `/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/` (override with
  `--out-root`). Nothing is written back into the data directory.

## Real-data check

```bash
bin/sanityCheck.py --det 18734 --fit --plot
```

Runs the full fit → save → load → apply chain on a 4096² lab ramp. Use it after
changes that touch fit/apply/io upstream.

## Docs layout

- Design specs: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
- Implementation plans: `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
```

- [ ] **Step 3: Update README.md**

Open `README.md`. Replace every occurrence of `nirLinearity` (as the *project*
name — not `lsst.obs.pfs.h4Linearity` references) with `fitLinearity`, replace
any `uv run python examples/sanity_check.py ...` invocation with
`bin/sanityCheck.py --det <det> ...`, and replace any `examples/linearity/...`
input path with `../jhu-data/<det>/`. If the README documents where outputs
land, point it at `/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/`.

- [ ] **Step 4: Verify no stale references survive**

Run:
```bash
grep -rn "examples/\|python\.bak\|/Users/cloomis\|nirLinearity-validation\|uv run" \
  --include=*.py --include=*.toml --include=*.md . | grep -v docs/superpowers
```
Expected: no output. (`docs/superpowers/` is excluded — the older specs and plans
are historical records and are not rewritten.)

Note: `import lsst.obs.pfs.h4Linearity as nirLinearity` inside `sanityCheck.py`
is an alias for the *upstream* package, not the project name — leave it.

- [ ] **Step 5: End-to-end verification — the real point of this plan**

Run, in an EUPS-setup shell, from the checkout root:
```bash
bin/sanityCheck.py --det 18734 --fit
```
Expected:
- It prints `in : /work/cloomis/claude/linfit/jhu-data/18734` and
  `out: /work/cloomis/outputs/fitLinearity/18734/o4`.
- It completes the fit → saveFits → loadFits → apply chain and reports
  `coefficients bitwise-equal: True`, `badPixelMask bitwise-equal: True`.
- `/work/cloomis/outputs/fitLinearity/18734/o4/18734_linearity.fits` exists.

Then confirm the input tree was not written to:
```bash
find ../jhu-data/18734 -newer pyproject.toml
```
Expected: no output.

Then the plot path:
```bash
bin/sanityCheck.py --det 18734 --plot --plot-format pdf
ls /work/cloomis/outputs/fitLinearity/18734/o4/
```
Expected: `18734_linearity.fits` and `diagnostic_18734.pdf`.

If the fit's summary diagnostics are available, compare against the known-good
18660 parity numbers before declaring victory — but 18734 at `o4` has no
recorded baseline, so a clean run plus the bitwise-equal checks is the bar here.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md .gitignore
git commit -m "Document the fitLinearity layout, EUPS setup, and data/output roots"
```

---

## Self-Review

**Spec coverage.** Rename → Task 1. Package + `bin/` → Tasks 1, 4, 5. `paths.py` →
Task 2. jhu-data inputs by `--det` → Tasks 2, 4. Outputs under
`/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/` → Tasks 2, 4. EUPS instead
of `pythonpath` → Task 1, documented in Task 6. Delete `python.bak/` → Task 1.
Verification (pytest green + end-to-end sanity check) → Tasks 1 and 6, Step 5.

**Type consistency.** `cliTag(config)` is defined in Task 4 and called in Task 4's
`bin/` script only. `loadCorrectedRamp(path, noPhotodiode=False)` is defined in
Task 3 and consumed in Tasks 4 and 5 with that exact signature.
`paths.inputDir(det, root=None)` / `paths.outputDir(det, cliTag, root=None)`
are defined in Task 2 and called with the `root=` keyword in Task 4.
`runBenchmark` differs by module (Task 5's Interfaces block gives both
signatures) — each has exactly one caller.

**Known soft spot.** Task 4 Step 5d is the largest single edit in the plan: a
1021-line file restructured by substitution table rather than by full listing.
Listing the whole file would run to hundreds of lines of unchanged plotting
code, so the table is the honest trade. The `cliTag` tests in Step 1 pin the one
piece of pure logic; the end-to-end run in Task 6 Step 5 is what actually proves
the restructure preserved behavior. Treat a diff in Task 4 that touches anything
inside `_plotDiagnostic` as a mistake.
