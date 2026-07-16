# fitLinearity

Validation harness for `lsst.obs.pfs.h4Linearity` — per-pixel polynomial
nonlinearity correction for IR-detector up-the-ramp data. The fit and apply
implementations live upstream; this repo holds the loaders, the fit driver,
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

## Threading

`fit()` parallelizes over image tiles (auto `workers = min(cpu_count, 8)`) while
numpy's BLAS spawns its own threads, so the two multiply. On a many-core host an
uncapped BLAS pool oversubscribes badly. `bin/fitLinearity.py` therefore caps
BLAS/OpenMP to one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`), set before numpy is imported and via
`setdefault` so an explicit environment value still wins. Single-threaded BLAS is
fastest at every core budget; the fit is memory-bandwidth bound, so tile-worker
speedup flattens past ~8–16 workers. The `bin/benchmark*.py` scripts control
threads themselves and are not subject to the cap.

## Data and outputs

- Inputs: `../jhu-data/<det>/*.npz`, addressed by `--det` (override the root with
  `--data-root` or `$FITLINEARITY_DATA`).
- Outputs: `/work/cloomis/outputs/fitLinearity/<det>/<cliTag>/` (override with
  `--out-root`). Nothing is written back into the data directory.

## Real-data check

```bash
bin/fitLinearity.py --det 18734 --fit --plot
```

Runs the full fit → save → load → apply chain on a 4096² lab ramp. Use it after
changes that touch fit/apply/io upstream.

## Rate-stability gate

`--rate-stability` runs the upstream split-half `detectRateInstability` gate on
the linearized ramp after the fit and folds `RATE_UNSTABLE` into the saved
correction's bad-pixel mask (re-saving the FITS). It tests each good pixel's
per-read rate for half-vs-half consistency over the valid prefix of the ramp:
reads up to the first one whose raw signal crosses above `fitMax`. Everything
from that first crossing on is masked, so the near-saturation extrapolation —
and the rollover where a saturating pixel's raw signal dips back below `fitMax` —
never enters the test. Knobs: `--rate-stability-threshold` (default 0.20) and
`--rate-stability-floor` (default 5.0 DN); the cliTag gains `_rs<threshold>`, plus
`_rsf<floor>` when the floor is non-default.

The gate needs a clean-linear fit. An order-4 correction under-corrects the
near-saturation knee, leaving in-range droop that makes the gate reject most good
pixels; order 5 removes it. A single anomalous low read just below `fitMax` (a
dropout the production CR/glitch mask would catch) can still trip the gate — this
harness passes an all-zeros CR flagMask, so only the fitMax clip is applied.

## Docs layout

- Design specs: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
- Implementation plans: `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
