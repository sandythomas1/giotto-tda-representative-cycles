# Decision Log

Append-only log of significant technical decisions across specs. Each entry: what was decided, why, alternatives considered, and which spec/task it came from.

<!-- New entries go at the top, most recent first -->

## 2026-08-17 — One Rips graph per fit, but only from the second feature on
**Spec:** 001-representative-cycle-fidelity (T7)

**Decided:** `fit()` builds one filtration-sorted edge list (`RipsGraphCache`,
float32 keys + int32 endpoints) capped at the largest birth radius any feature asks
for, visits features in ascending birth radius, and falls back to the old per-feature
dense mask below `_MIN_FEATURES_FOR_GRAPH_CACHE = 2`.

**Why:** Measured 11× faster on the 1500-point torus (424 features, 2.54 s → 0.23 s)
at 0.11× the memory of the dense mask it replaces, with zero differences in cycle
length or verification status. The threshold exists because one sort over every edge
costs more than the single mask it would replace when there is only one feature.

**Alternatives:** (a) Always build the cache — rejected, it is a slowdown for
single-feature fits. (b) Cache without the radius cap — rejected, the unbounded edge
list is ~6× the dense mask and would fail the spec's memory budget.

**Tradeoff:** The graph is mutated in place to mask each birth edge (restored on exit),
so it is not thread-safe; the library is single-threaded by construction.

## 2026-08-17 — The overview shares one projection plane, and says so
**Spec:** 001-representative-cycle-fidelity (T13)

**Decided:** `plot_overview()` fits a single best-fit plane over the union of the drawn
cycles' vertices and annotates the figure with the retained fraction plus a pointer to
`plot_cycle()`. `plot_cycle()` keeps each loop's own plane. `plot_cycle` rejects
negative indices instead of wrapping them.

**Why:** One pair of axes can only carry one plane — per-cycle planes share no
coordinate system, so overlaying them would place loops relative to each other
arbitrarily. The cost is real (a loop can be foreshortened), so under the
constitution's no-silent-degradation rule it is printed on the figure rather than left
for the reader to discover. `plot_cycle(-1)` would silently draw the *least* persistent
cycle from a most-persistent-first list.

**Alternatives:** (a) Naive `X[:, :2]` for the overlay — rejected, that is the V4 defect
this spec removed from the panels. (b) One sub-panel per plane — rejected, that is
`plot_matplotlib()`, and the overview exists to answer a different question.

**Tradeoff:** Two views of the same loop can look different; the annotation is what
makes that legible instead of confusing.

## 2026-08-17 — Axes-level drawing seams for the composite views
**Spec:** 001-representative-cycle-fidelity (T13)

**Decided:** Extracted `draw_barcode(ax, ...)` out of `plot_barcode()` and added
`draw_cycle_overlay(ax, coords, feature, color)` to `panels.py`; `plot_overview()`
composes those rather than re-implementing either. The overview figure uses matplotlib's
constrained layout instead of `tight_layout()`.

**Why:** A second barcode implementation inside the overview would drift from the real
one. Constrained layout is needed because the persistence diagram carries a colorbar,
and a colorbar axes inside a nested gridspec is exactly the case `tight_layout` warns it
cannot handle — a warning the user could not act on.

**Alternatives:** Duplicating the drawing code in `overview.py` — rejected outright.

**Tradeoff:** `overview.py` imports three underscore-prefixed helpers from `panels.py`;
that is deliberate coupling within one subpackage, and the alternative (a fourth
`_common.py`) buys nothing today.

## 2026-08-16 — Split the single module into a package behind a compatibility shim
**Spec:** 001-representative-cycle-fidelity

**Decided:** `representative_cycles.py` (30 KB, computation + matplotlib + plotly in one file)
becomes a `repcycles/` package (`core.py`, `reconstruction.py`, `projection.py`,
`palette.py`, `plotting_mpl.py`, `plotting_plotly.py`), with `representative_cycles.py`
retained as a thin re-export shim.

**Why:** The file mixes three concerns with different change rates and different dependency
weights, and the constitution requires that computation never import matplotlib at module
scope. Splitting also lets independent workstreams touch disjoint files.

**Alternatives:** (a) Keep one file — rejected, forces serialised edits and keeps the
matplotlib import coupling. (b) Split without a shim — rejected, breaks the documented
public import path, which the constitution names a stable contract.

**Tradeoff:** More files to navigate; one extra indirection when tracing an import.
