# nirLinearity

Per-pixel polynomial nonlinearity correction for IR-detector up-the-ramp data.

## Conventions

- **camelCase** for Python function names, method names, and variable names — not snake_case. Applies to tests too. Classes remain PascalCase. Constants remain UPPER_SNAKE_CASE. Module filenames stay short/single-word where possible; use camelCase only for compound module names. Do not "correct" existing camelCase to snake_case.
- Package source lives at `python/nirLinearity/` (not `src/`). Tests at `tests/`.

## Tooling

- Dependency manager: `uv`. Run things as `uv run pytest`, `uv run python examples/sanity_check.py`, etc.
- Python 3.12.

## Docs layout

- Design specs: `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`
- Implementation plans: `docs/superpowers/plans/YYYY-MM-DD-<topic>.md`
- New specs/plans follow these paths.

## Real-data check

`examples/sanity_check.py` runs the full fit→save→load→apply chain on the 4096² lab ramp at `examples/linearity/18734/18734_164220.npz` (not in repo). Use it after changes that touch fit/apply/io.
