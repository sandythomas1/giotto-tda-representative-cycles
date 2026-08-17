# Spec 001: Representative Cycle Fidelity — Correct Extraction, Honest Views

Status: Reviewed — revised per `review.md` (all 7 blocking findings addressed)

## Problem

`RepresentativeCycles` is the whole point of this repository: it turns ripser's birth/death
simplex pairs into loops a human can look at. Today it does that for the easy case only, and
the views it produces can mislead.

Three measured failures. **(1)** Essential H₁ classes — bars that never die within the
filtration — are dropped entirely, because `fit()` reads only `gens[1]` (finite pairs) and
never `gens[3]` (essential pairs). A 60-point circle fitted with `max_edge_length=1.0`, the
exact setting a user reaches for on a large cloud, returns **zero** features while its one
obvious loop sits in `gens[3]` unread. **(2)** Every plot flattens a point cloud with
`X[:, :2]`, silently discarding the third coordinate. On a 600-point torus the tube-loops
retain only **0.50–0.69** of their variance under that projection versus **0.92–0.98** under
a per-cycle best-fit plane — the loop is drawn as a near-degenerate smear and the user reads
it as noise. **(3)** The persistence diagram highlights "shown" features by indexing
finite-filtered arrays with full-diagram row indices, which silently marks the wrong point,
or raises `IndexError`, as soon as an infinite bar is present.

The audience is researchers who will publish figures made by this tool and trust its
`features_` list as ground truth. A wrong loop that looks plausible is the worst outcome
this library can produce.

## Goals

- Every H₁ class ripser reports — finite **and** essential — appears in `features_` with a
  reconstructed cycle.
- Pairing between persistence-diagram rows and generator rows is verified, not assumed from
  undocumented ripser row ordering.
- Every returned cycle carries the evidence needed to trust it: closure, radius conformance,
  geodesic length, and an explicit flag when reconstruction degraded.
- 3-D point clouds are viewed without silent information loss.
- One cycle has one colour in every view — diagram, barcode, panel, and interactive figure.
- Cycle reconstruction stops rebuilding an O(n²) dense mask per feature.
- The documented import path `from representative_cycles import RepresentativeCycles`
  keeps working unchanged.

## Non-Goals

- H₂ (voids) or dimensions above 1. `maxdim=1` stays hard-wired; the sphere example keeps
  showing H₁ only.
- Algebraic (boundary-matrix-reduction) cycle representatives à la JavaPlex/GUDHI. The
  shortest-path heuristic stays the method; this spec makes it *honest and verified*, not
  canonical.
- Minimal / optimal homologous cycle computation (an NP-hard problem in general over ℤ).
- Replacing `gph-ripser`, or vendoring any part of it.
- A GUI, a web app, or a notebook widget. Views remain matplotlib figures and plotly figures.
- Publishing to PyPI.

## Requirements

### Functional

**Extraction method**

- **F1** — `fit()` reads essential H₁ generators from `gens[3]` and emits a `CycleFeature`
  for each, with `death = np.inf` and `persistence = np.inf`.
- **F2** — Essential features are reconstructed at their birth radius by the same
  shortest-path procedure as finite features, and are sorted first (infinite persistence
  sorts above every finite value).
- **F3** — `min_persistence` never filters out an essential feature (∞ ≥ any threshold).
  A separate `include_essential: bool = True` constructor flag opts out.
- **F4** — Each generator row is paired to its diagram row by matching the birth-edge length
  `D[u, v]` against the diagram's birth value within a tolerance, rather than by positional
  index. Tolerance is relative, calibrated to float32 epsilon (`rtol=1e-5`, `atol=1e-9`),
  because ripser computes in float32 while our distances are float64.
- **F4a** — Ties are expected, not exceptional. Congruent features produce diagram rows
  identical in *both* birth and death (measured: three congruent hexagonal holes give three
  identical rows `[1.0, 1.7320508]`). Such rows are **interchangeable** — a persistence
  diagram is a multiset, and the pairing carries no information beyond the values. Matching
  therefore claims the **lowest unclaimed row index** among all rows within tolerance, and
  this is not an error condition.
- **F4b** — `CycleReconstructionError` is raised only when **zero** unclaimed rows match a
  generator, naming the generator, its birth-edge length, and the candidate rows. A silently
  mispaired feature is never emitted.
- **F5** — `CycleFeature` gains: `cycle_length: float` (sum of geodesic edge weights),
  `cycle_path: np.ndarray` (vertices in loop traversal order, first vertex repeated at the
  end), `is_essential: bool`, and `is_verified: bool`.
- **F6** — `is_verified` is `True` only when the reconstructed loop passes both checks:
  every vertex has even degree (closed over ℤ/2) **and** every edge length ≤ birth radius
  (1 + 1e-9 relative tolerance). Otherwise `False`.
- **F7** — When no u→v path exists without the birth edge, the feature is still returned
  with the birth edge alone, `is_verified=False`, and a `warnings.warn` of a dedicated
  `CycleReconstructionWarning` category. No bare `print`.
- **F8** — Dijkstra tie-breaking is deterministic: two `fit()` calls on identical input,
  within the same implementation, return byte-identical `cycle_edges` arrays. This is a
  *self*-consistency guarantee. It deliberately does **not** promise byte-identity with the
  pre-optimisation implementation: the shortest-path *distance* is unique but the argmin
  *path* is not, so any change to graph construction may legitimately select a different
  equal-cost path.
- **F8a** — Feature ordering is fully determined, with no reliance on sort stability or on
  ripser's emission order. Sort key: essential before finite, then persistence descending,
  then birth ascending, then generator index ascending. (Without this, multiple essential
  features — all with `persistence = inf` — have undefined relative order.)
- **F9** — `to_dataframe()` returns one row per feature with all scalar fields, plus
  `n_edges`. pandas is imported lazily and its absence raises a clear `ImportError`.
- **F10** — Non-finite input is handled by *kind*, not rejected wholesale:
  - `NaN` is rejected always, in both modes, with a `ValueError` naming the first offending
    index. NaN has no topological meaning and silently poisons every comparison.
  - `+inf` is **accepted** in a precomputed distance matrix and documented as "no edge
    between these points". This is the standard encoding for geodesic distances on a
    disconnected mesh — a use case the README advertises — and `gph` accepts it (verified).
    Such pairs are simply absent from the Rips graph at every radius.
  - `±inf` is rejected in Euclidean *coordinates*, where it is meaningless.
  - Negative distances are rejected in precomputed matrices.
  - **Consequence found during implementation:** accepting `+inf` is not sufficient on its
    own, because the MDS embedding used to *position* points for plotting is scikit-learn's,
    which rejects any non-finite value — so `fit()` would validate happily and then crash in
    visualisation. Since the embedding is layout-only and never topological, infinite
    distances are replaced by twice the largest finite distance **for the embedding alone**.
    That substitution misrepresents the geometry, so it is documented in the method's
    docstring rather than hidden: disconnected points are drawn far apart, but their drawn
    separation carries no metric meaning. Every topological quantity uses the original matrix.
  - Symmetry is checked with `np.isclose`, not `max(abs(D - D.T))`: `inf - inf` is `NaN`, so
    the subtraction form reports a spurious `nan` asymmetry on exactly the inputs this
    requirement exists to support.

**View**

- **V1** — The persistence-diagram panel highlights exactly the features present in
  `features_`, correct for any mix of finite and infinite bars; infinite bars are drawn on a
  clearly-labelled "∞" band above the finite range rather than dropped.
- **V2** — A single `cycle_colors()` mapping assigns each feature a colour; the persistence
  diagram, barcode, matplotlib panels, and plotly traces all use it. Cycle *k* has the same
  colour in every view produced from the same fit. The canonical representation is a **hex
  string** (`'#4c72b0'`), because matplotlib artists carry RGBA float tuples and plotly
  carries CSS strings, which are never directly `==`; each renderer converts from hex, and
  equality is asserted after normalising both back through `matplotlib.colors.to_hex`.
- **V3** — The palette is a discrete, colourblind-safe qualitative set, taken by explicit
  index (not by continuously sampling a qualitative colormap, which is what
  `cm.tab10(np.linspace(...))` does today).
- **V4** — For point clouds with ≥ 3 dimensions, each cycle panel projects onto that cycle's
  own best-fit plane (PCA on the cycle's vertices) and annotates the panel with the fraction
  of variance retained. Two criteria, one relative and one absolute:
  - *Relative (always holds, cannot flake):* retained variance is ≥ that of the current
    `X[:, :2]` projection for every cycle. This is guaranteed by SVD optimality, so it is
    a real invariant rather than a tuned threshold.
  - *Absolute (benchmark):* ≥ 0.90 on the torus fixture, where `X[:, :2]` retains as little
    as 0.50. (Probed for safety on the sphere fixture too, where genuinely 3-D loops still
    retain 0.964–0.996 — the threshold is not at risk there.)

    **Fixture sensitivity, measured during implementation and recorded here rather than
    papered over:** across 24 torus samples (n=600, noise 0.04–0.05, seeds 0–11) the top-6
    minimum ranged 0.867–0.965, and 5 of 24 fell below 0.90. Those dips are real geometry,
    not a projection defect — occasionally a top-6 class is a spiral winding around the tube
    *and* partway around the main circle, which no plane can capture (its third singular
    value reaches 40–60% of its first). Denser sampling removes them; at n=800 nearly every
    seed clears 0.90. The benchmark is therefore pinned to a named fixture (n=600, seed 8,
    measured top-6 minimum 0.951), and the *relative* criterion above remains the one that
    holds universally. This is exactly why the spec carries both.
- **V5** — `plot_overview()` renders one figure combining the point cloud with all selected
  cycles overlaid in their assigned colours, the persistence diagram, and the barcode —
  colour-linked across all three.
- **V6** — `plot_barcode()` draws infinite bars as right-pointing arrows, marks bars that
  correspond to `features_`, and uses the shared colour mapping.
- **V7** — `plot_plotly()` attaches hover text to every cycle trace: feature index, birth,
  death, persistence, cycle length, edge count, verified flag.
- **V8** — `plot_cycle(index)` renders a single cycle at full figure size with local context
  (neighbouring points within the birth radius), for figure-quality output.
- **V9** — `plot_plotly(show_skeleton=True)` optionally draws the Rips 1-skeleton at the
  focused cycle's birth radius as a faint underlay, so the user can see what the algorithm
  saw. Off by default. Capped by `skeleton_max_edges: int = 20_000`; when the skeleton
  exceeds the budget, the **shortest** 20 000 edges are drawn (deterministic, and the short
  edges are the ones that carry local structure) and the figure title is annotated
  `skeleton truncated: 20000 / N edges`. Silent truncation is forbidden by the
  constitution's no-silent-degradation rule.
- **V10** — No deprecated matplotlib APIs; the module emits zero `DeprecationWarning` /
  `MatplotlibDeprecationWarning` under `-W error::DeprecationWarning` on the pinned versions.
- **V11** — Plot methods never call `plt.show()` and always return the figure.

**Structure**

- **F11** — Code is reorganised into a `repcycles/` package separating computation from
  visualisation; `representative_cycles.py` remains as a re-export shim so that
  `from representative_cycles import RepresentativeCycles, CycleFeature` is unchanged.
- **F12** — `repcycles.core` imports neither matplotlib nor plotly at module scope.

### Non-Functional

- **Security** — Beyond the constitution baseline: `fit()` validates shape, dtype,
  finiteness (per F10), and symmetry/zero-diagonal for precomputed matrices *before*
  allocating the O(n²) distance matrix. Memory exhaustion is handled without a behaviour
  break: `max_points: int | None = None` (no cap by default, preserving today's behaviour);
  above 5000 points a `ResourceWarning` states the projected float64 distance-matrix cost
  (n=5000 → ~200 MB); a caller who wants a hard limit sets `max_points` and gets a
  `ValueError`. `save_path` / `save_html` are caller-supplied paths in a library whose
  caller *is* the trust boundary — there is no privilege gradient to defend, so the
  requirement is usability, not traversal defence: fail with a clear message naming the
  missing parent directory rather than surfacing a raw `OSError`.
- **Performance/Scale** — The Rips graph is built **once per fit** and reused across all
  features, replacing the current per-feature dense `n × n` boolean mask. Recorded context
  (this machine, not an assertion): 2.37 s reconstruction on a 1500-point torus, 426
  features. The *assertion* is a ratio measured in-process on the same fixture — the new
  path must be **≥ 2× faster** than the legacy dense-mask path — because a wall-clock target
  hard-codes one laptop into the suite. Marked `@pytest.mark.perf` and deselectable.
  Equivalence is asserted on **total geodesic cycle length** (within 1e-9) and
  `is_verified` status, not on byte-identical edge arrays (see F8).
  Memory: the sorted-edge structure must not regress memory by more than 2× against the
  dense mask it replaces. Measured naively it is 8× worse (200 MB vs 25 MB at n=5000, using
  float64 keys + int64 indices), so float32 keys and int32 indices are required, and the
  edge list is built only up to `max_edge_length` when one is set.
- **Observability** — Degraded reconstruction surfaces as a `CycleReconstructionWarning`
  and an `is_verified=False` field. `summary()` shows a verification column. No `print` in
  library code paths other than `summary()` itself.

## Breaking Changes

Users upgrading should know about exactly these. Everything else is additive.

- **B1** — `features_` now contains essential (infinite-death) features by default. Code
  that assumes `np.isfinite(f.death)` for every feature must branch, or pass
  `include_essential=False`. This is the intended headline change.
  *Blast radius, measured:* with the default `max_edge_length=inf` the Rips complex
  eventually becomes a full simplex, every class dies, and **no essential bars arise at
  all** — verified on a 60-point circle (0 essential generators at `thresh=inf`, 1 at
  `thresh=1.0`). So B1 changes results only for callers who set `max_edge_length`, which is
  precisely the population currently losing its most persistent loops in silence.
- **B2** — `persistence` may be `inf`. Any downstream arithmetic (normalisation, sums,
  colour scaling) must handle it.
- **B3** — Above 5000 points, `fit()` emits a `ResourceWarning`. It does not change results
  and does not raise unless `max_points` is set explicitly.
- **B4** — `CycleFeature` gains fields. Positional construction of the dataclass past
  `death_edge` still works; new fields are keyword-with-defaults and appended last.
- **B5** — Reconstruction may return a *different but equally valid* representative cycle
  than the previous release for the same input (same class, same geodesic length). Cycle
  edge arrays are therefore not comparable across versions.

## Acceptance Criteria

1. Circle of 60 points fitted with `max_edge_length=1.0` yields exactly one feature, with
   `is_essential=True`, `death=inf`, and `len(cycle_edges) >= 3`. (Today: zero features.)
2. The Zhu §2.3 rectangle still gives birth = 2.000000 and death = √5 to within 1e-4, with a
   4-edge cycle — unchanged from the current suite.
3. A dataset producing both finite and essential bars returns features whose `birth` equals
   `D[u, v]` of its own birth edge for every feature, within 1e-4.
4. Pairing is tested as a **pure function** on synthetic arrays, not via a real dataset
   (no real input can be coerced into producing an unmatched generator):
   a. Generator whose birth-edge length matches no diagram row → `CycleReconstructionError`
      naming the generator and its length.
   b. Three congruent hexagonal holes (birth 1.0, death √3 for all three) → three features,
      no error, each claiming a distinct diagram row.
   c. Generators presented out of diagram order → each still pairs to its own row.
5. Every feature returned for the circle, figure-eight, torus, and annulus fixtures has
   `is_verified=True`, and `cycle_length` equals the sum of its edge lengths within 1e-9.
6. Two consecutive `fit()` calls on the same input produce `np.array_equal` cycle edges.
7. Input validation, one assertion each: `np.nan` in coordinates → `ValueError` naming the
   index; `np.inf` in coordinates → `ValueError`; `np.inf` in a *precomputed* matrix →
   accepted, treated as no-edge, fit completes; negative precomputed distance → `ValueError`;
   >5000 points → `ResourceWarning` but successful fit; exceeding an explicitly-set
   `max_points` → `ValueError`.
8. Torus benchmark projection, both criteria: each of the top 6 cycles retains ≥ 0.90 of its
   variance under the best-fit plane, **and** retains ≥ the `X[:, :2]` value for every cycle
   on every 3-D fixture (the SVD-optimality invariant, which cannot flake).
9. Colour assigned to feature *k* is identical across `plot_matplotlib`, `plot_barcode`,
   `plot_overview`, and `plot_plotly` for one fit — asserted by extracting artist and trace
   colours and normalising both through `matplotlib.colors.to_hex` before comparison.
10. `plot_barcode()` on a fit containing an essential bar renders an arrow artist for it and
    does not raise.
11. `plot_matplotlib()` on a fit containing an essential bar highlights the correct diagram
    point and does not raise `IndexError`.
12. `plot_cycle(0)` and `plot_plotly(show_skeleton=True)` return figures for 2-D and 3-D
    inputs without error.
13. Performance, asserted as an in-process ratio on the 1500-point torus fixture (never as
    wall-clock, which would hard-code one machine into the suite): the reused-graph path is
    ≥ 2× faster than the legacy dense-mask path run in the same process, and produces cycles
    of identical total geodesic length (within 1e-9) and identical `is_verified` status.
    Test marked `@pytest.mark.perf`.
13a. Memory: the sorted-edge structure uses float32 keys and int32 indices, verified by
    asserting the dtypes and total `nbytes` stay within 2× the dense mask it replaces.
14. `from representative_cycles import RepresentativeCycles, CycleFeature` works, and the
    entire pre-existing test suite (34 passed, 1 skipped) still passes untouched except
    where a test asserted a behaviour this spec deliberately changes.
15. `pytest -W error::DeprecationWarning` passes for the plotting modules.
16. `python examples.py` and `python zhu_paper_replication.py` both run to completion and
    the Zhu replication reports all five examples passing.

## Open Questions

- None blocking. One judgement call recorded rather than asked: essential features are
  **included by default** (`include_essential=True`), on the grounds that dropping a user's
  most persistent loop is a worse default than showing a bar with an infinite death. The
  flag exists for anyone who disagrees.
