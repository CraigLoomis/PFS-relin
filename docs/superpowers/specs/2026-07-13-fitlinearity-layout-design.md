# fitLinearity: package layout and path resolution

**Date:** 2026-07-13
**Status:** approved

## Purpose

The validation harness for `lsst.obs.pfs.h4Linearity` is renamed to
**fitLinearity** and restructured into a normal Python package with thin
executable wrappers. Input ramps, upstream products, and output artifacts each
get a single well-defined location, none of which is a machine-specific
absolute path baked into a config file.

## Scope

- Rename the project to `fitLinearity`.
- Turn `examples/` into a package at `python/fitLinearity/`, with executables in `bin/`.
- Read fit inputs from `jhu-data/`, addressed by detector id.
- Write outputs under `/work/cloomis/outputs/fitLinearity/`.
- Resolve `obs_pfs` and `drp_stella` through EUPS `setup` from `/work/cloomis/claude/PIPE2D-1844/`.
- Delete `python.bak/`.

Out of scope: the git checkout directory keeps its current name (`relin/`); no
changes to fit, apply, or io logic, all of which live upstream.

## Layout

```
relin/                              git root
  pyproject.toml                    name = "fitLinearity"
  CLAUDE.md  README.md
  python/fitLinearity/
      __init__.py
      loader.py                     photodiode-canonicalizing loadNpz
      syntheticRamp.py
      sanityCheck.py                runFit(), runPlot()
      paths.py                      data-root / output-root resolution
      benchmarks/
          __init__.py
          fitThreading.py
          fitBlocksize.py
  bin/
      sanityCheck.py                argparse wrapper -> sanityCheck.runFit/runPlot
      benchmarkThreading.py
      benchmarkBlocksize.py
  tests/                            unchanged
```

`python/fitLinearity/` holds importable logic only. Each `bin/` script is a thin
argparse front end that parses arguments, resolves paths through `paths.py`, and
calls into the package. Scripts are run directly (`bin/sanityCheck.py --det
18734 --fit`); there is no install step and no `[project.scripts]` block.

`paths.py` is the single place that knows the input and output roots and the
`detId -> (inputDir, outputDir)` mapping, so the three `bin/` scripts do not each
re-derive them.

`tests/` imports only `lsst.obs.pfs.h4Linearity`, never the harness package, so
the move does not touch it.

`loader.py` is imported by name (`from fitLinearity.loader import loadNpz`). It
is a distinct module from upstream `h4Linearity.loaders`, not a shadow of it, so
no import-order constraint applies.

## Path resolution

### Inputs

Two arguments: `--data-root` (default `../jhu-data` relative to the checkout
root, overridden by `$FITLINEARITY_DATA` when set) and `--det` (e.g. `18734`).
The input directory is `dataRoot/det`, inside which the scripts glob `*.npz` as
they do today.

`detId` comes from `--det` directly. It is not derived from the input
directory's basename, so an input directory need not be named after the detector
it holds.

### Outputs

```
/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/
    <det>_linearity.fits
    diagnostic_<det>.pdf
    ...
```

The root is `/work/cloomis/outputs/fitLinearity`, overridable with
`--out-root`. `cliTag` is built from the fit configuration exactly as it is now
(`o5_dev0.45`, `o4_pdOff`, ...). Nothing is written into `jhu-data/`.

### Upstream products

`obs_pfs` and `drp_stella` come from `/work/cloomis/claude/PIPE2D-1844/` and are activated through
EUPS, never through `PYTHONPATH`:

```bash
source /work/stack/loadLSST.bash
setup pfs_pipe2d
setup -j -r /work/cloomis/claude/PIPE2D-1844/obs_pfs
setup -j -r /work/cloomis/claude/PIPE2D-1844/drp_stella
```

The hardcoded absolute `pythonpath` entry is removed from `pyproject.toml`. The
one entry that remains is `"python"`, which makes the harness's own source tree
importable; that is this repo's source, not a local-checkout activation, so it
does not conflict with the EUPS-only rule. The setup chain above is documented
in `CLAUDE.md`.

## Verification

A layout and path change has no unit-test surface of its own. It is verified by:

1. `pytest` passes green in an EUPS-setup shell, with no `pythonpath` entry
   pointing at obs_pfs — proving the products resolve through EUPS alone.
2. `bin/sanityCheck.py --det 18734 --fit` runs end to end and writes its FITS
   and diagnostics under `/work/cloomis/outputs/fitLinearity/18734/o4/`, with
   `jhu-data/18734/` left unmodified.
