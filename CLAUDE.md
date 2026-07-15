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
setup -j -r /work/cloomis/claude/PIPE2D-1844/obs_pfs
setup -j -r /work/cloomis/claude/PIPE2D-1844/drp_stella
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
