# Review: Spec 001

Verdict: **NEEDS REVISION** → **RESOLVED** (all 11 findings applied to `spec.md` / `plan.md`
on 2026-08-16; spec status is now *Reviewed*. Findings retained below as the record of why
the spec reads the way it does.)

Seven blocking findings, four non-blocking. Three of them come from probes run against the
spec's own assumptions, which turned out to be wrong.

## Findings

### 1. F4's by-value pairing is genuinely ambiguous on congruent features — BLOCKING

**What's wrong.** F4 replaces positional pairing with "match birth-edge length `D[u,v]`
against the diagram's birth value", and plan.md's risk section says ties are disambiguated
"by matching death values too". Measured: three congruent hexagonal holes produce

```
[[1.0, 1.7320508], [1.0, 1.7320508], [1.0, 1.7320508]]
```

— three rows identical in *both* birth and death. Death-value disambiguation does not
disambiguate anything here.

**Why it matters.** As specified, the implementer has no defined behaviour for this case.
One engineer raises `CycleReconstructionError` on a perfectly healthy lattice dataset (a
regression from today's code, which handles it fine); another silently takes the first match.
Congruent features are not exotic — they are what you get from any symmetric or grid-like
point cloud, which is most synthetic test data.

**Fix.** State explicitly that rows identical within tolerance in both birth and death are
*interchangeable*, because a persistence diagram is a multiset and the pairing carries no
information beyond the values. Specify: match against unclaimed rows, take the lowest
unclaimed row index among those matching, and raise only when **zero** rows match. Add an
acceptance criterion covering the congruent-holes case explicitly.

### 2. AC13 over-constrains the optimisation to byte-identical output — BLOCKING

**What's wrong.** AC13 requires the optimised reconstruction's cycle edges be "identical to
those from the pre-optimisation implementation". Dijkstra's shortest *distance* is unique;
the shortest *path* is not. Rebuilding the graph from a sorted edge list changes CSR row
ordering, which changes which of several equal-cost predecessors SciPy records.

**Why it matters.** The implementer either (a) contorts the new graph builder to reproduce
the old CSR ordering — throwing away the optimisation's freedom for no scientific gain — or
(b) declares the criterion unmeetable and skips it. Both are bad outcomes from a criterion
that was trying to ask for something reasonable.

**Fix.** Replace with the property that actually matters: the optimised implementation must
produce cycles of **identical total geodesic length** (within 1e-9) and identical
`is_verified` status on a fixed-seed fixture. Keep byte-identity as a *self*-consistency
requirement (F8: two runs of the *same* implementation agree), which is achievable and is
what determinism means.

### 3. Wall-clock acceptance criteria are machine-dependent — BLOCKING (testability)

**What's wrong.** AC13's "≤ 1.2 s" hard-codes the reviewer's laptop into the test suite. The
2.37 s baseline was measured on this Windows box; on slower CI hardware the target is
unreachable, on a faster machine it passes even if the optimisation was never implemented.

**Fix.** Assert a *ratio* measured within the same test run: reconstruct with the legacy
dense-mask path and the new path in one process on the same fixture, and require the new
path be ≥ 2× faster. Mark it `@pytest.mark.perf` and allow deselection. Keep the absolute
numbers in the spec as recorded context, not as assertions.

### 4. F10 rejects `inf` in precomputed matrices, breaking a legitimate encoding — BLOCKING

**What's wrong.** F10 rejects "NaN/±inf coordinates or distances". Probed: `gph` accepts
`inf` entries in a precomputed matrix without error. `inf` is the standard encoding for
"these two points are not connected" — exactly what you get from a geodesic distance matrix
on a disconnected mesh, or a graph shortest-path matrix, both named as use cases in the
README ("geodesic distances on meshes").

**Why it matters.** A user who fits a mesh geodesic matrix today gets results; after this
spec they get a `ValueError` on valid input.

**Fix.** Split the requirement: NaN is rejected always (it has no meaning and silently
poisons comparisons). `+inf` is *accepted* in precomputed matrices and documented as
"no edge"; it is rejected in Euclidean coordinates, where it is meaningless. Negative
distances are rejected. Add a test for each.

### 5. `max_points=5000` is an undisclosed breaking change — BLOCKING

**What's wrong.** The non-functional security requirement introduces a 5000-point cap where
today there is none, and the spec never lists it as a breaking change. AC14 won't catch it
because every fixture in the suite is tiny.

**Why it matters.** Someone running an 8000-point cloud successfully today gets a hard
`ValueError` after upgrade. The guard is defensible; introducing it silently is not — and
5000 points is a 200 MB float64 distance matrix, which is large but not obviously fatal.

**Fix.** Keep the guard, but: default `max_points=None` (no cap, preserving today's
behaviour), document the O(n²) memory cost in the docstring with a worked figure, and emit a
`ResourceWarning` above 5000 points rather than raising. A user who wants a hard cap sets
one. Add this to a new "Breaking Changes" section in the spec — currently absent.

### 6. AC9's colour-consistency assertion is unwritable as specified — BLOCKING (testability)

**What's wrong.** "Colour assigned to feature *k* is identical across `plot_matplotlib` …
and `plot_plotly`". matplotlib artists carry RGBA float tuples; plotly traces carry CSS
strings like `'#636EFA'`. They are never `==`.

**Fix.** Define the contract at the source: `cycle_colors()` returns hex strings, each
renderer converts as needed, and the test asserts equality after normalising both back to
hex via `matplotlib.colors.to_hex`. Say so in V2.

### 7. V9's "edge budget" is unspecified — BLOCKING (clarity)

**What's wrong.** "capped by an edge budget to stay responsive" names no number and no
overflow behaviour. At n=1500 the 1-skeleton at a typical birth radius is easily 10⁵ edges;
whether the implementer subsamples, truncates, or refuses changes what the user sees.

**Fix.** Specify: default budget 20 000 edges; when exceeded, draw the shortest 20 000 edges
(deterministic), and annotate the figure that the skeleton is truncated. Never silently
show a partial skeleton without saying so — the constitution forbids silent degradation.

### 8. Sort order among multiple essential features is undefined — NON-BLOCKING

All essential features share `persistence = inf`, so `sort(key=persistence)` leaves their
relative order to Python's sort stability over whatever order ripser emitted. Harmless today,
but it undercuts F8's determinism claim the moment two essential bars exist. Suggest an
explicit composite key: `(is_essential desc, persistence desc, birth asc, index asc)`.

### 9. The `save_path` traversal requirement is security theatre — NON-BLOCKING

"the call must not create directories outside it" — the caller of a library *is* the
trust boundary; there is no privilege gradient between the user and their own script. This
adds code that protects nobody. Suggest reducing it to a usability requirement: fail with a
clear message naming the missing directory instead of matplotlib's raw `OSError`.

### 10. plan.md's performance alternatives omit the strongest competitor — NON-BLOCKING

The rejected alternative ("sparse threshold graph per unique birth radius, no global sort")
is close to a strawman. The real competitor is **incremental construction**: sort features by
birth radius ascending and add edges to one growing graph, never re-slicing. That is the
textbook approach and plausibly beats slice-per-feature. It should be considered on its
merits and rejected (or adopted) explicitly.

### 11. plan.md hides a memory regression — NON-BLOCKING

Measured: a fully sorted edge list costs **200 MB at n=5000** (float64 key + two int32
indices over 12.5 M edges) against **25 MB** for the dense bool mask it replaces — 8× worse,
in a plan section that presents the change as a pure win. State it, and mitigate: store the
sort key as float32 and indices as int32, or build the sorted list only up to
`max_edge_length`.

## Non-Issues Considered

- **Would the ≥0.90 projection threshold fail on genuinely 3-D loops (e.g. sphere)?** Probed
  the sphere fixture: best-fit-plane variance retained for the top 6 cycles is 0.964–0.996.
  The threshold holds. Worth noting the requirement is still better stated *relatively* —
  the best-fit plane is optimal by SVD, so "retains ≥ what `X[:, :2]` retains" is
  mathematically guaranteed and can never produce a flaky test — but the absolute threshold
  is not the trap it looked like.
- **Does the positional pairing the spec is replacing actually work today?** Probed both
  all-finite and mixed finite/essential diagrams: the finite rows did align positionally in
  every case tested. The spec is right to replace it anyway (it is undocumented behaviour
  being relied on for scientific output), but this is hardening, not a live bug — the spec's
  Problem section correctly does not claim otherwise.
- **Scope discipline on H₂.** Non-Goals excludes it cleanly and no requirement smuggles it
  back in. `maxdim=1` stays.
- **Constitution consistency.** F12 (no matplotlib in `core`) directly implements the
  architecture principle; F11's shim implements the stable-contract principle. No conflicts.
- **Essential-bar default.** Defaulting `include_essential=True` is a genuine product call
  and the spec flags it as a recorded judgement rather than burying it. Fine as decided.
