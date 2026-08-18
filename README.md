# giotto-tda-cycles

**Representative H₁ cycle visualisation for [giotto-tda](https://github.com/giotto-ai/giotto-tda) / [gph-ripser](https://github.com/giotto-ai/giotto-ph)**

`giotto-tda` computes persistent homology barcodes beautifully.  
What it does **not** give you is a way to look at the actual loops — the
geometric paths through your point cloud that correspond to each H₁ bar.

This add-on fills that gap.

---

## What it does

`gph.ripser_parallel(return_generators=True)` returns four integers per H₁
bar: the birth edge `(u, v)` and the death triangle.  Those integers are not
a cycle — they are just the simplex that *created* the homological class.
To get a drawable loop you need a second step.

`RepresentativeCycles` provides that step: it builds the Vietoris-Rips
1-skeleton at the birth radius, runs **scipy's C-level Dijkstra** from `u`
to `v` (excluding the birth edge so the path is forced around the hole), and
closes the loop with the birth edge.  The result is a valid representative of
the H₁ class — a closed edge-path you can actually plot.

Every loop it returns is then **re-checked** and reported with the evidence:
`is_verified` is `True` only when the loop is closed over ℤ/2 (every vertex
has even degree) *and* every one of its edges exists in the complex at the
birth radius.  A loop that fails either check is still returned — with
`is_verified=False`, a `CycleReconstructionWarning`, and a badge on the
figure — because a plausible-looking loop that is quietly wrong is the worst
output this library could produce.

```python
from representative_cycles import RepresentativeCycles
import numpy as np

theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
X = np.column_stack([np.cos(theta), np.sin(theta)])

rc = RepresentativeCycles(min_persistence=0.2)
rc.fit(X)
rc.summary()                # table, with a verification column
rc.plot_overview()          # cloud + all cycles + diagram + barcode
rc.plot_cycle(0)            # the most persistent loop, full size
rc.plot_plotly().show()     # interactive
```

---

## Example output

The figure below replicates examples A–E from Zhu (2013)
*"Persistent Homology: An Introduction and a New Text Representation for NLP"*,
the same paper whose barcodes were originally produced by JavaPlex.
All five expected topologies are recovered correctly.

![Zhu-paper replication](assets/demo.png)

| Example | Shape | Expected β₁ | Observed β₁ |
|---------|-------|:-----------:|:-----------:|
| A | Rectangle (exact Zhu §2.3 example) | 1 | 1 ✓ |
| B | Circle S¹ | 1 | 1 ✓ |
| C | Figure-eight S¹∨S¹ | 2 | 2 ✓ |
| D | Torus T² (H₁(T²) = ℤ×ℤ) | 2 | 2 ✓ |
| E | Sphere S² (H₁ = 0) | 0 | 0 ✓ |

Example A is verifiable to floating-point precision: birth = **2.000000**,
death = **√5 ≈ 2.236068**, exactly matching the Zhu §2.3 prediction.

Regenerate it with `python zhu_paper_replication.py`, which prints the
expected-vs-observed table above and exits only after all five pass.

---

## Installation

```bash
pip install -r requirements.txt
# then clone / copy the repcycles/ package into your project
```

Or install the package itself:

```bash
pip install -e .                  # core
pip install -e ".[dataframe]"     # plus pandas, for to_dataframe()
```

`pandas` is optional and used by `to_dataframe()` alone; without it that one
method raises an `ImportError` naming the install command, and everything else
works unchanged.

---

## Usage

### Point cloud (Euclidean)

```python
rc = RepresentativeCycles(
    min_persistence=0.1,     # ignore short-lived noise bars
    max_edge_length=2.0,     # cap the filtration (see "Essential cycles")
    metric="euclidean",      # default
)
rc.fit(X)                    # X: (n_points, n_dims)
rc.summary()                 # birth/death/persistence/length/verified table
rc.plot_overview()           # cloud + all cycles + diagram + barcode
rc.plot_matplotlib()         # diagram + one projected panel per cycle
rc.plot_cycle(0)             # one cycle, full size, with local context
rc.plot_barcode()            # barcode, essential bars as arrows
rc.plot_plotly()             # interactive, toggleable cycles
```

### Pre-computed distance matrix

Any metric space — cosine similarity on text, geodesic distances on meshes,
correlation matrices, etc.:

```python
from scipy.spatial.distance import pdist, squareform

D = squareform(pdist(X, metric="cosine"))

rc = RepresentativeCycles(metric="precomputed")
rc.fit(D)      # D: (n, n) symmetric, zeros on diagonal
rc.summary()
rc.plot_matplotlib()   # visualises an MDS embedding of D
```

`+inf` is accepted in a precomputed matrix and means *"no edge between these
two points"* — the standard encoding for geodesic distances on a disconnected
mesh.  `NaN` and negative distances are rejected, naming the offending index.

### Essential cycles (loops that never die)

Setting `max_edge_length` truncates the filtration, so a loop can survive to
the end of it.  Such a class is **essential**: `death = inf`,
`persistence = inf`.  These are included by default and are usually the most
important loops in the data.

```python
rc = RepresentativeCycles(max_edge_length=1.0, min_persistence=0.3)
rc.fit(X)

[f.is_essential for f in rc.features_]   # -> [True]
rc.features_[0].death                    # -> inf
```

`min_persistence` never filters an essential feature.  Pass
`include_essential=False` for the old behaviour.

### Accessing cycle data directly

```python
for f in rc.features_:
    print(f.birth, f.death, f.persistence)
    print("birth edge:  ", f.birth_edge)      # (u, v)
    print("cycle edges: ", f.cycle_edges)     # (n_edges, 2) int array
    print("traversal:   ", f.cycle_path)      # closed loop order, plottable
    print("length:      ", f.cycle_length)    # summed edge lengths
    print("verified:    ", f.is_verified)

df = rc.to_dataframe()                        # requires pandas
```

---

## API reference

### `RepresentativeCycles(...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_persistence` | `0.0` | Drop features below this persistence threshold. Never drops an essential feature |
| `max_edge_length` | `np.inf` | Cap on the Rips filtration (speeds up large inputs, and is what makes essential classes possible) |
| `metric` | `'euclidean'` | `'euclidean'` for point clouds; `'precomputed'` for distance matrices |
| `coeff` | `2` | Coefficient field — use 2 for ℤ/2ℤ |
| `n_threads` | `1` | Threads passed to `ripser_parallel` |
| `reconstruct_cycles` | `True` | Set `False` to skip cycle reconstruction (barcode only) |
| `include_essential` | `True` | Include classes that never die (`death = inf`) |
| `max_points` | `None` | Hard cap on input size. `None` = no cap; above 5000 points a `ResourceWarning` states the projected distance-matrix cost |

### `.fit(X)` → `self`

Validates `X`, runs Vietoris-Rips persistent homology, pairs each ripser
generator to its diagram row **by value** (not by position — row order
agreement between `dgms` and `gens` is undocumented ripser behaviour and does
not survive essential bars), and reconstructs one loop per surviving class.
An unmatchable generator raises `CycleReconstructionError` rather than
reporting a cycle under another class's birth and death.

Validation happens *before* the O(n²) distance matrix is allocated.

### `.features_` → `list[CycleFeature]`

Sorted essential-first, then persistence descending, then birth ascending,
then diagram row — a total order, so the list is reproducible across runs.

| Field | Description |
|-------|-------------|
| `index` | Row index into the H₁ persistence diagram |
| `birth`, `death`, `persistence` | Filtration values; `death`/`persistence` are `inf` for essential classes |
| `birth_edge`, `death_edge` | `(u, v)` pairs. `death_edge` is `(-1, -1)` when essential |
| `cycle_edges` | `(n_edges, 2)` int array, in traversal order |
| `cycle_vertices` | Unique vertex indices, sorted |
| `cycle_path` | Vertices in traversal order, first repeated at the end — plots directly as a closed polygon |
| `cycle_length` | Summed edge lengths of the loop |
| `is_essential` | The class never dies within the filtration |
| `is_verified` | The loop passed both checks: closed over ℤ/2, and every edge within the birth radius |
| `n_edges` | `len(cycle_edges)` |

### `.diagrams_` → `list[np.ndarray]`

Standard giotto-tda / Ripser persistence diagrams.

### Output methods

| Method | What it gives you |
|--------|-------------------|
| `.summary()` | Printed table; the `Ok` column is `is_verified`, and `∞` renders for essential deaths |
| `.to_dataframe()` | One row per feature, every scalar field plus `n_edges` (needs pandas) |
| `.plot_overview(max_cycles=6, figsize=None, title=..., save_path=None)` | One figure: cloud with all cycles overlaid, persistence diagram, barcode — colour-linked |
| `.plot_matplotlib(max_cycles=6, figsize=None, title=..., save_path=None)` | Persistence diagram + one panel per cycle, each on its own best-fit plane |
| `.plot_cycle(index, figsize=(7, 7), save_path=None, context_radius=None)` | One cycle at full figure size with the points inside its birth radius picked out |
| `.plot_barcode(figsize=(8, 4), save_path=None)` | Barcode; essential bars are right-pointing arrows, drawn cycles are marked |
| `.plot_plotly(max_cycles=6, title=..., save_html=None, show_skeleton=False, skeleton_max_edges=20_000)` | Interactive figure; hover gives index, birth, death, persistence, cycle length, edge count and verified flag |

Every plot method **returns its figure and never calls `plt.show()`**, so the
caller decides when and whether to display or close it.

Cycle *k* has the same colour in every view of the same fit — the palette is a
discrete, colourblind-safe qualitative set indexed by position, and the same
list is handed to all views.

### Viewing loops in ≥ 3 dimensions

Panels do not drop the third coordinate.  Each cycle panel projects onto that
**cycle's own best-fit plane** (rank-2 SVD of its vertices, applied to the
whole cloud so the loop keeps its context) and prints the fraction of the
cycle's variance the plane retained, alongside the `X[:, :2]` figure it
replaces.  On the torus fixture that is the difference between ~0.50 and
≥ 0.90 retained.

`plot_overview()` cannot do this — one pair of axes carries one plane — so it
fits a single plane across all the drawn loops and says so on the figure,
pointing at `plot_cycle()` for any loop whose shape matters.

### Errors and warnings

| Type | Raised when |
|------|-------------|
| `CycleReconstructionError` | A ripser generator matches no diagram row |
| `CycleReconstructionWarning` | The birth edge's endpoints are disconnected at the birth radius, so no loop can be traced; the result degrades to the birth edge alone with `is_verified=False` |
| `ResourceWarning` | Input exceeds 5000 points (states the projected memory cost; does not change results) |
| `ValueError` | NaN/±inf coordinates, a malformed precomputed matrix, `max_points` exceeded, or a `save_path` whose parent directory does not exist |

Import them from `repcycles.errors`.

---

## Breaking changes (0.1 → 0.2)

- **B1** — `features_` now contains essential (infinite-death) features by
  default. Code that assumes `np.isfinite(f.death)` must branch, or pass
  `include_essential=False`. With the default `max_edge_length=inf` the
  complex eventually becomes a full simplex and no essential bars arise at
  all, so this changes results only for callers who set `max_edge_length` —
  precisely the callers who were silently losing their most persistent loops.
- **B2** — `persistence` may be `inf`. Downstream arithmetic (normalisation,
  sums, colour scaling) must handle it.
- **B3** — Above 5000 points `fit()` emits a `ResourceWarning`. It does not
  change results and raises only if you set `max_points`.
- **B4** — `CycleFeature` gained fields (`cycle_length`, `cycle_path`,
  `is_essential`, `is_verified`). They are keyword-with-defaults appended
  last, so positional construction through `death_edge` still works.
- **B5** — Reconstruction may return a *different but equally valid*
  representative cycle than 0.1 for the same input: same homology class, same
  geodesic length, possibly different edges. Equal-cost shortest paths are not
  unique, so `cycle_length` — not the edge array — is the quantity to compare
  across versions.

`from representative_cycles import RepresentativeCycles, CycleFeature`
is unchanged, and remains the supported import path.

---

## Performance

The Rips graph is built **once per fit**, as a filtration-sorted edge list
(float32 keys, int32 endpoints) sliced by `searchsorted`, instead of a dense
`n × n` boolean mask per feature. On the 1500-point torus benchmark (424 H₁
classes) that is an 11× reconstruction speed-up on the development machine;
the test suite asserts the machine-independent part — a ≥ 2× in-process ratio
against the old path, with identical cycle lengths and verification status:

```bash
pytest -m perf          # the benchmark
pytest -m "not perf"    # everything else
```

The edge list is capped at the largest birth radius any feature asks for, so
it stays smaller than the mask it replaces (246 KB vs 2.25 MB at n=1500).
Above ~10⁴ points, set `max_edge_length`.

---

## How this compares to alternatives

| | **This library** | **JavaPlex** | **GUDHI** |
|---|---|---|---|
| Language | Python | Java / MATLAB | C++ / Python |
| Representative cycles | Shortest-geodesic-path heuristic | Algebraic boundary-matrix reduction | `persistence_generator_extension` |
| Same homology class? | Yes | Yes | Yes |
| Identical cycle edges? | Not necessarily | — | Not necessarily |
| Essential (infinite) classes | Included, drawn, reconstructed | Yes | Yes |
| giotto-tda integration | Native (uses `gph-ripser`) | None | Via `gudhi.representations` |
| Interactive visualisation | Plotly + matplotlib | None built-in | None built-in |
| Pre-computed distance matrices | Yes (`metric='precomputed'`) | Yes | Yes |

**Cycle representative note:**  
JavaPlex and GUDHI compute cycle representatives via algebraic boundary-matrix
reduction — the canonical pivot-column representative.  This library uses a
shortest-geodesic-path **heuristic**: the output is a *valid* representative of
the same homology class, but the specific edges may differ, and it is not the
minimal loop in the class (that problem is NP-hard over ℤ in general).  The
heuristic tends to produce geometrically tighter (visually cleaner) loops.

**float32 note:**  
`gph-ripser` requires float32 inputs.  Coordinates are cast to float32 only
for the `ripser_parallel` call; all distance computations for cycle
reconstruction run in float64 to preserve accuracy.

---

## Running the examples

```bash
# seven examples: circle, figure-eight, torus, annulus, sphere,
# an essential bar, and the overview / single-cycle figures
python examples.py

# Zhu (2013) paper replication with pass/fail checks
python zhu_paper_replication.py

# outputs saved to ./output/
```

---

## Project structure

```
repcycles/                  the package
    core.py                 RepresentativeCycles: fit, summary, to_dataframe
    feature.py              CycleFeature — the shared data contract
    pairing.py              generator → diagram-row matching, by value
    reconstruction.py       shortest-path loops + the shared Rips graph
    projection.py           per-cycle best-fit plane
    palette.py              the one colour mapping every view uses
    validation.py           input validation at the trust boundary
    errors.py               CycleReconstructionError / Warning
    plotting/               diagram, barcode, panels, overview, interactive
representative_cycles.py    re-export shim: the stable import path
examples.py                 gallery of seven examples
zhu_paper_replication.py    Zhu (2013) replication with pass/fail checks
specs/                      spec-driven development history
tests/                      pytest suite
output/                     generated figures (gitignored)
```

Computation modules import neither matplotlib nor plotly at module scope; the
`plot_*` methods import their backend lazily, so the library stays usable in
an environment with no plotting stack installed.

---

## References

- Zhu, X. (2013). *Persistent Homology: An Introduction and a New Text
  Representation for Natural Language Processing.*
  IJCAI 2013, pp. 1953–1959.
  [(paper)](https://pages.cs.wisc.edu/~jerryzhu/pub/homology.pdf)
- Tausz, A., Vejdemo-Johansson, M., & Adams, H. (2011). JavaPlex: A research
  software package for persistent (co)homology.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
  Introduction.* AMS.
- Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence
  barcodes. *Journal of Applied and Computational Topology*, 5(3), 391–423.

---

## License

MIT
