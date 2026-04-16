# `fit()` Tile Threading — Design

**Status:** Approved 2026-04-16.
**Scope:** `python/relin/fit.py` only.

## Motivation

`fit()` iterates tiles sequentially. On the reference 29 × 4096 × 4096 lab
ramp this takes ~146 s wall-clock (64 tiles of 512², single-threaded).
Each tile's `model.fitBlock` call is independent pure numpy, so tile-level
threading is the natural first speedup. Target: ~5-7× wall-clock reduction
on 4096² frames with no behavior change for existing callers.

## Public API change

New keyword-only — well, positional-or-keyword to keep it consistent with
the existing signature style — parameter `workers`:

```python
def fit(
    ramps: Sequence[Ramp],
    model: Model | None = None,
    blockSize: tuple[int, int] = (512, 512),
    workers: int | None = None,
    conditionNumberLimit: float = 1e12,
) -> LinearityCorrection
```

## Effective-worker resolution

Resolved once at call time from the first ramp's frame size and
`os.cpu_count()`:

```python
if workers is None:
    H, W = ramps[0].deltas.shape[1:]
    if H * W < 1_000_000:
        effective = 1                              # small frame → no threading
    else:
        effective = min(os.cpu_count() or 1, 8)    # auto default, cap at 8
else:
    effective = workers                             # explicit wins; no clamp
if effective < 1:
    raise ValueError("workers must be >= 1")
```

Rationale:
- **Default is heuristic, not fixed.** The 1 M-pixel threshold avoids
  creating an 8-thread pool to process a 64×64 test frame. The cap at 8
  for auto-detection avoids runaway worker counts on 32- or 64-core CI
  hosts where BLAS oversubscription risk climbs.
- **Explicit `workers=N` is honored as-written.** A caller passing
  `workers=16` intentionally on a 32-core node gets 16 workers. Silently
  clamping the caller's explicit request would be a surprise.
- **`effective < 1` is an error, not a no-op.** `workers=0` is
  ambiguous ("use no workers"? "let the library decide"?); rejecting it
  outright removes the ambiguity.

## Control flow

```
if effective == 1:
    <existing sequential tile loop — unchanged>
else:
    with ThreadPoolExecutor(max_workers=effective) as ex:
        futures = {
            ex.submit(
                model.fitBlock,
                m=mTile, t=tConcat, valid=validTile,
                conditionNumberLimit=conditionNumberLimit,
            ): (rowStart, rowEnd, colStart, colEnd)
            for <each tile, with mTile and validTile assembled on the
                 submitting thread before submission>
        }
        for fut in as_completed(futures):
            rs, re, cs, ce = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                for other in futures:
                    other.cancel()
                raise RuntimeError(
                    f"fitBlock failed at tile "
                    f"[rows {rs}:{re}, cols {cs}:{ce}]"
                ) from e
            <write result into disjoint slices of the preallocated
             output arrays — identical to the sequential stitch-back>
```

### Why split sequential and threaded paths

Keeping `effective == 1` on the existing loop (rather than routing it
through a one-worker `ThreadPoolExecutor`) has three benefits:

- Zero overhead for the fast path (no thread creation / future bookkeeping).
- Smallest possible diff risk for callers who never opt in.
- Clear separation makes it easy to read either path on its own.

### Why assemble tiles on the submitting thread

Per-tile `m` and `valid` array assembly (slicing `cumulatives[k]`,
concatenating across ramps, building default-`valid` arrays for ramps
without `validMask`) happens on the submitting (main) thread, not inside
worker threads. This keeps `model.fitBlock` invocations identical to the
sequential path — workers receive fully-formed numpy arrays and do pure
compute — and bounds peak memory to `effective` in-flight tiles rather
than one.

## Thread safety

- `model.fitBlock` is pure: reads its inputs, returns a new
  `BlockFitResult`. Verified by inspection of
  `python/relin/models/polynomial.py`.
- `PolynomialModel` is a frozen dataclass with no shared state.
- Preallocated output arrays (`coefficients`, `fitMin`, `fitMax`,
  `badPixelMask`, diagnostic arrays) are **only** written on the main
  thread, inside the `as_completed` consumer loop. Workers never touch
  shared arrays.
- Input arrays (`cumulatives`, `ramp.validMask`) are read-only to workers,
  and each worker reads a disjoint slice.

The net effect is that no numpy array is concurrently accessed from
multiple threads in a way that could race. The GIL handles the small
Python-level bookkeeping (future completion, dict lookups) safely.

## Determinism

Because each tile writes to disjoint output slices and all writes happen
on the main thread, the final output arrays are **byte-identical
regardless of the effective worker count or tile completion order**. The
tests assert this property directly.

## Error propagation

If any `fitBlock` future raises:

1. The first exception observed in the `as_completed` loop is caught.
2. `cancel()` is called on all remaining futures (this prevents any
   not-yet-started tasks from running; in-flight tasks run to completion).
3. A `RuntimeError` is raised with the offending tile's row/col range in
   the message, chained from the original exception via `raise ... from
   e` so the underlying traceback is preserved.

The `with ThreadPoolExecutor(...)` block guarantees the pool is shut down
on both the success and failure paths.

## Testing

Added to `tests/test_fit_threading.py` (new file — keeps `test_fit.py`
focused on the sequential semantics):

1. **Byte-identical output.** For the existing `smallSyntheticRamp` plus
   one larger synthetic fixture (e.g. 1024×1024), assert
   `fit(ramps, workers=1)` and `fit(ramps, workers=4)` produce every
   array field bit-equal via `np.array_equal`, and the summary dict is
   equal as a dict.
2. **Worker-resolution heuristic.** Three unit tests that monkeypatch
   `os.cpu_count` and a module-level `_executorFactory` pointer in
   `relin.fit` (recording the `max_workers` it was constructed with):
   - Small frame + `workers=None` → effective worker count is 1 and no
     executor is constructed (sequential path).
   - Large frame + `workers=None` + `cpu_count()=16` → executor built
     with `max_workers=8` (cap hit).
   - Large frame + `workers=None` + `cpu_count()=4` → executor built
     with `max_workers=4` (cap not hit).
3. **Explicit `workers` bypasses the cap.** `fit(ramps, workers=16)` on
   a (mocked) 4-core box records `max_workers=16`.
4. **`workers=0` raises.** `pytest.raises(ValueError, match="workers")`.
5. **Worker exception is re-raised with tile coords.** A test-only
   `Model` subclass whose `fitBlock` raises on a specific tile range;
   assert the resulting `RuntimeError` message contains "rows" and
   "cols" substrings matching the injected tile's coordinates, and that
   `__cause__` is the injected exception.

The `_executorFactory` indirection is introduced specifically to make
tests 2 and 3 possible without patching `concurrent.futures` globally.
It defaults to `concurrent.futures.ThreadPoolExecutor` and is not part
of the public API.

## Benchmark

`examples/benchmark_fit_threading.py` — standalone script, not a pytest
test — measures wall-clock for `workers ∈ {1, 2, 4, 8}` on the real
4096² ramp (`examples/linearity/18734/18734_164220.npz`) and prints a
small table. This is the evidence that will drive the follow-up decision
on whether `threadpoolctl` is worth adding; not a pass/fail gate for
this PR.

## Documentation

`fit()` docstring gets one new paragraph covering:

- What `workers` does, and what `None` means (cite the heuristic).
- Note: BLAS/LAPACK may spawn additional threads internally; if threaded
  speedup plateaus, try setting `OMP_NUM_THREADS=1` /
  `MKL_NUM_THREADS=1` in the process environment.
- Note: output is worker-count-independent and deterministic.

## Dependencies

No new runtime dependency. `concurrent.futures.ThreadPoolExecutor` is
stdlib. `os.cpu_count` is stdlib.

## Explicitly out of scope

- `threadpoolctl` integration (deferred; decision driven by the
  benchmark).
- Multiprocessing / Dask / ray-style parallelism.
- Threading inside `apply()` (separate concern; `apply` is not tiled).
- Progress reporting (callback / tqdm).
- Persistent / reusable thread pool across multiple `fit` calls.
