# Plan 001: Representative Cycle Fidelity — Correct Extraction, Honest Views

## Approach

Two independent halves — extraction and view — connected by one new contract: the
`CycleFeature` dataclass and a colour-assignment function derived from it. Get that contract
right first, and the halves can be built in parallel against it.

**Extraction.** The current `fit()` walks `gens[1][0]` and trusts row *i* of the generator
array to correspond to row *i* of `dgms[1]`. Measurement shows that holds today for
all-finite diagrams, but it is undocumented ripser behaviour and it demonstrably does *not*
hold once essential bars enter (`dgms[1]` gained the infinite row while `gens[1][0]` did
not). The fix is to stop relying on order: for each generator, compute the birth-edge length
`D[u, v]` from our own float64 distance matrix and match it against unclaimed diagram rows
by value. That single change makes essential support (F1–F3) a straightforward extension —
essential generators from `gens[3]` match the diagram rows whose death is `inf` — and turns
a class of silent mispairing into a loud `CycleReconstructionError`.

Reconstruction changes from "build a dense mask and a fresh CSR per feature" to "build one
filtration-sorted edge list per fit, then slice it". Sort all `n(n-1)/2` upper-triangle
edges by length once; for a feature born at radius *r*, `np.searchsorted` gives the prefix
of edges present in Rips(*r*), and a CSR graph is assembled from that prefix. Features
sharing a birth radius (common — ripser births cluster) reuse the same graph via a small
cache keyed on the searchsorted cut index. Cycle *verification* is then cheap and worth
doing unconditionally: even-degree check plus a max-edge-length check against the birth
radius, recorded on the feature rather than assumed.

**View.** Three changes carry most of the value. First, a single `cycle_colors(features)`
function is the only place a colour is chosen; every renderer takes its colour from there,
which is what makes a figure readable across panels. Second, panels for ≥3-D clouds project
each cycle onto *its own* best-fit plane via SVD of the cycle's centred vertices, and print
the retained-variance fraction on the panel — measured to lift torus tube-loops from 0.50 to
0.96. The projection is fitted on the cycle's vertices and then applied to the *whole* cloud,
so the loop keeps its context instead of floating in isolation. Third, infinite bars get an
explicit visual language: a labelled "∞" band in the diagram, arrow caps in the barcode.

**Structure.** `representative_cycles.py` becomes a package. The shim keeps the public import
path, the constitution's stated contract. This is done as the very first task so that
subsequent work lands in disjoint files.

## Architecture

```
repcycles/
  __init__.py          public surface: RepresentativeCycles, CycleFeature, errors
  errors.py            CycleReconstructionError, CycleReconstructionWarning
  validation.py        input validation: shape, finiteness, symmetry, max_points guard
  core.py              RepresentativeCycles: fit(), pairing, summary(), to_dataframe()
                       — no matplotlib / plotly import at module scope (constitution)
  feature.py           CycleFeature dataclass (the shared contract)
  reconstruction.py    RipsGraph: sorted-edge build once, searchsorted slice, Dijkstra,
                       path trace, verification
  projection.py        best-fit-plane projection + retained-variance metric
  palette.py           cycle_colors(): colourblind-safe discrete assignment
  plotting/
    __init__.py        re-exports the plot_* functions
    diagram.py         persistence-diagram panel (finite + ∞ band)
    barcode.py         barcode with ∞ arrows and shown-bar markers
    panels.py          per-cycle panels + plot_matplotlib
    overview.py        plot_overview, plot_cycle
    interactive.py     plot_plotly (hover metadata, skeleton underlay)

representative_cycles.py   re-export shim  (public path — unchanged for users)
tests/
  test_representative_cycles.py   existing suite, kept green
  test_extraction.py              essential bars, pairing, verification, determinism
  test_reconstruction_perf.py     graph reuse, benchmark, output-identity
  test_projection.py              variance-retention thresholds
  test_plotting.py                colour consistency, ∞ rendering, Agg-backend structure
```

Dependency direction is strictly one-way: `plotting/* → palette/projection → feature`, and
`core → reconstruction/validation → feature`. Nothing in the computation column imports the
visualisation column.

The plotting subpackage is split by *view*, not lumped into one module, for a practical
reason beyond tidiness: each view is an independent unit of work touching its own file, so
the view lane parallelises instead of serialising on one 800-line module. `panels.py` is
the only view that depends on `projection.py`.

## Alternatives Considered

- **Algebraic cycle representatives (boundary-matrix reduction), matching GUDHI/JavaPlex** —
  rejected. It would mean reimplementing persistence reduction on top of ripser's output,
  which is a different project, and the README already positions the shortest-path heuristic
  as a deliberate choice that yields tighter, more interpretable loops. The right fix for
  trust is verification and honest labelling, not swapping the algorithm.
- **Keeping the single 30 KB module and serialising all edits** — rejected. It forces
  matplotlib into the import path of anyone who only wants the topology, and it makes the
  visualisation and extraction work contend for the same file.
- **Incremental graph construction** — process features in ascending birth radius and grow
  one edge set monotonically, never re-slicing. This is the textbook approach and the
  genuine competitor to slice-per-feature; it does strictly less work in the limit, since
  every edge is inserted exactly once across the whole fit. **Adopted in part**: features
  are processed in ascending birth order and the CSR graph is rebuilt only when the
  searchsorted cut index actually changes, which captures the same win (ripser births
  cluster heavily — 426 features on the torus share far fewer distinct radii) without
  needing an incremental CSR structure, which SciPy does not provide cheaply. If the ≥2×
  target is missed, full incremental construction with a union-find-backed adjacency
  structure is the next step, not more micro-optimisation.
- **Sparse threshold graph built lazily per unique birth radius, no global sort** — this is
  close to what the adopted design does; kept here only to note that without the global sort
  each radius costs an O(n²) scan, which is exactly the per-feature cost being eliminated.
- **3-D matplotlib axes (`mplot3d`) for 3-D clouds instead of per-cycle plane projection** —
  rejected as the default. A static 3-D axes at panel size is harder to read than a
  well-chosen 2-D projection, and the interactive plotly view already covers true 3-D
  inspection. The per-cycle plane is measurable (retained variance) in a way "looks 3-D"
  is not.
- **Asking the user whether essential bars should default on** — decided instead: default
  on, with `include_essential=False` available. Silently dropping the most persistent
  feature in the dataset is the more surprising behaviour.

## Tradeoffs

Optimises for **trustworthiness per figure** over raw throughput. Verification runs on every
feature (an even-degree check and a max over the cycle's edges — negligible against Dijkstra),
and the by-value diagram pairing costs a small search per generator instead of an array index.
Both are paid so that no feature can be silently wrong, which the constitution's scientific
correctness bar demands.

**The performance change trades memory for time, and that was not obvious.** A fully sorted
edge list is measured at **200 MB at n=5000** (float64 key + int64 index pair over 12.5 M
edges) against **25 MB** for the dense boolean mask it replaces — 8× worse, in a change that
looks like a pure win. float32 keys and int32 indices bring it to 100 MB, and building the
list only up to `max_edge_length` cuts it further on the realistic path. The residual
regression is accepted because the dense mask was allocated *per feature* (426 times on the
torus) while the sorted list is allocated once, so peak-to-work ratio still improves — but
anyone fitting >10⁴ points should set `max_edge_length`, and the docstring says so.

Given up: positional pairing's simplicity, and a flat single-file module that was easy to
copy into another project. The shim keeps the copy-one-file story alive only for the import
path, not the file itself — anyone vendoring `representative_cycles.py` alone will now need
the package. That is a real cost, accepted because the file had already outgrown the format.

The package split also front-loads risk: task 1 touches everything before any feature work
lands. It is mitigated by making that task a pure move with zero behaviour change, gated on
the existing 34-test suite passing untouched.

## Risks

- **Pairing tolerance too tight or too loose.** ripser computes in float32; our distances are
  float64, so birth values differ in the last bits. Mitigation: match with a relative
  tolerance calibrated to float32 epsilon (~1e-6 relative, floored at 1e-9 absolute), and
  when several rows are within tolerance, disambiguate by preferring an unclaimed row and
  matching death values too. Test with duplicated/degenerate distances.
- **Essential features break downstream arithmetic.** `persistence = inf` propagates into
  sorting, colour normalisation, and axis limits. Mitigation: every consumer of `persistence`
  gets an explicit finite/infinite branch, and the barcode/diagram tests include an essential
  bar specifically to catch `inf` leaking into a limit computation.
- **Perf target not met by graph reuse alone.** If 1.2 s proves out of reach, the fallback is
  grouping features by identical birth-radius cut index and reusing one Dijkstra run per
  source vertex. Decide only after measuring — do not pre-optimise into complexity.
- **The move task silently changes behaviour.** Mitigation: task 1 is a pure move; the
  existing test file is not edited in that task at all, and its result is the gate.
- **Projection helper degenerates on tiny cycles** (3-vertex loops, collinear points).
  Mitigation: fall back to the first two coordinates when the cycle has < 3 vertices or the
  second singular value is ~0, and report the retained variance honestly rather than
  claiming 1.0.
