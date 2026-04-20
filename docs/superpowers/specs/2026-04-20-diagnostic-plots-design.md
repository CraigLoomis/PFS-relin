# Diagnostic Plots for Sanity Check

## Summary

Add a `--plot` flag to `examples/sanity_check.py` that produces two diagnostic
PNG files after the fit-save-load-apply pipeline completes. No changes to the
relin library itself.

## Plot 1: Before/After Linearization (`diagnostic_before_after.png`)

2x1 figure (stacked vertically).

**Top subplot — accumulated flux vs read number:**
- N (default 1000) random good pixels.
- Raw cumulative `m` (before) and linearized cumulative `t` (after) vs read.
- Thin translucent lines so overlapping traces show density.
- For each pixel, segments where `m` is outside `[fitMin, fitMax]` drawn in red;
  in-range segments in the normal color. Applies to both raw and linearized traces.
- Straight reference line showing ideal `rate * n` target.

**Bottom subplot — delta flux (per-read differences):**
- Same N pixels.
- Raw deltas and linearized deltas (diff of cumulative).
- Out-of-range segments in red, same as above.
- Horizontal reference line at the median rate.

Default N = 1000. Saved next to the FITS file.

## Plot 2: Fit Rejection Diagnostics (`diagnostic_rejections.png`)

2x1 figure.

**Top subplot — bar chart of rejection categories:**
- One bar per flag type: MASKED_BY_INPUT, INSUFFICIENT_POINTS, FIT_FAILED,
  NON_MONOTONIC, plus "good" (no flags).
- Heights are pixel counts. Percentage labels on each bar.
- Flags are bit-flags so bars are not mutually exclusive.

**Bottom subplot — failed pixel traces:**
- Up to N (default 1000) pixels with any bad-pixel flag set (excluding
  MASKED_BY_INPUT which has no useful signal).
- Plot raw cumulative `m` vs read number.
- Color-coded by failure type (FIT_FAILED, NON_MONOTONIC, INSUFFICIENT_POINTS).
- Shows why fitting failed.

Default N = 1000 (or all failed pixels if fewer). Saved next to the FITS file.

## Implementation

- All code in `examples/sanity_check.py` — no library changes.
- `--plot` flag via `argparse`. When absent, behavior is unchanged.
- `matplotlib` used only in the example script, not added to package dependencies.
- PNGs saved to the same directory as the FITS output.
