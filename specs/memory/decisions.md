# Decision Log

Append-only log of significant technical decisions across specs. Each entry: what was decided, why, alternatives considered, and which spec/task it came from.

<!-- New entries go at the top, most recent first -->

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
