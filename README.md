# fitLinearity

Validation harness for `lsst.obs.pfs.h4Linearity` — per-pixel polynomial
nonlinearity correction for IR-detector up-the-ramp data. The fit and apply
implementations live upstream; this repo holds the loaders, the sanity check,
and the benchmarks.

## Install (development)

```bash
uv sync
```

`uv sync` only installs the dev tooling (pytest, ruff, matplotlib) into a local
venv for editing. It is not used to run anything: `lsst.obs.pfs.h4Linearity`
is an EUPS product, not a pip dependency, so tests and `bin/` scripts must run
under an EUPS-setup shell instead.

## Environment

```bash
source /work/stack/loadLSST.bash
setup pfs_pipe2d
setup -j -r /work/cloomis/claude/PIPE2D-1844/obs_pfs
setup -j -r /work/cloomis/claude/PIPE2D-1844/drp_stella
```

Use the LSST-env `python` directly — do not use `uv run`.

## Run tests

```bash
pytest
```

## Real-data check

```bash
bin/sanityCheck.py --det 18734 --fit --plot
```

Reads `../jhu-data/<det>/*.npz` and writes the corrected FITS file and
diagnostic plot under `/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/`.

## Design

See `docs/superpowers/specs/2026-07-13-fitlinearity-layout-design.md`.
