# Tasks: Spec 001

16 tasks. The dependency graph is deliberately shaped so that after the contract lands (T2),
a method lane and a view lane run in parallel on **disjoint files** — no two concurrent tasks
write the same module.

Execution waves (each wave's tasks are mutually independent):

| Wave | Tasks | Lane |
|------|-------|------|
| 0 | T1 | foundation (pure move — gates everything) |
| 1 | T2 | shared contract |
| 2 | T3, T4, T6, T9, T10, T11, T14 | method + view in parallel |
| 3 | T5, T7, T12 | |
| 4 | T8, T13 | |
| 5 | T15, T16 | integration |

---

## T1: Split the module into a package behind a compatibility shim
- Description: Pure structural move, **zero behaviour change**. `representative_cycles.py`
  becomes `repcycles/` (`__init__.py`, `core.py`, `reconstruction.py`, `plotting/`), and
  `representative_cycles.py` is reduced to a re-export shim. No logic is edited, renamed, or
  "improved" in this task — anything tempting is left for the task that owns it.
- Files likely touched: `repcycles/**`, `representative_cycles.py`, `pyproject.toml` (packages)
- Depends on: none
- Security-sensitive: no
- Acceptance criteria:
  - The existing `tests/test_representative_cycles.py` passes **completely untouched** (34 passed, 1 skipped).
  - `from representative_cycles import RepresentativeCycles, CycleFeature` works.
  - `import repcycles.core` does not import matplotlib or plotly (assert via `sys.modules`).
  - `git diff` shows no changed logic lines, only moves and import rewiring.
- Status: done

## T2: Shared contract — CycleFeature, errors, palette
- Description: The data structure and colour mapping both lanes build against. Extend
  `CycleFeature` with `cycle_length`, `cycle_path`, `is_essential`, `is_verified` (F5,
  keyword-with-defaults, appended last per B4). Add `errors.py` with
  `CycleReconstructionError` and `CycleReconstructionWarning`. Add `palette.py` with
  `cycle_colors(features) -> list[str]` returning **hex strings** (V2, V3).
- Files likely touched: `repcycles/feature.py`, `repcycles/errors.py`, `repcycles/palette.py`, `tests/test_contract.py`
- Depends on: T1
- Security-sensitive: no
- Acceptance criteria:
  - `CycleFeature(...)` constructible positionally through `death_edge` as before (B4).
  - New fields default to `0.0` / empty array / `False` / `False`.
  - `cycle_colors` returns hex strings, is deterministic, and cycles the palette for >N features.
  - Palette is a named colourblind-safe discrete set, indexed by position — not a continuous
    sample of a qualitative colormap.
- Status: done

## T3: Input validation
- Description: Implement F10 exactly as specified — NaN rejected everywhere; `+inf` **accepted**
  in precomputed matrices as "no edge"; `±inf` rejected in Euclidean coordinates; negative
  precomputed distances rejected; `max_points=None` default with `ResourceWarning` above 5000
  points (B3); clear error naming a missing parent directory for `save_path`.
- Files likely touched: `repcycles/validation.py`, `repcycles/core.py` (call site), `tests/test_validation.py`
- Depends on: T2
- Security-sensitive: **yes** — this is the untrusted-input boundary. Numeric input from
  callers, memory-exhaustion guard on the O(n²) allocation, and filesystem paths. Per the
  constitution, validation happens *before* the large allocation, not after.
- Acceptance criteria:
  - Every bullet of AC7 has a passing test.
  - `inf` in a precomputed matrix completes a fit and the affected pair is absent from the
    Rips graph at every radius.
  - Validation runs before the `cdist` allocation (assert via a monkeypatched `cdist` that fails).
- Status: done

## T4: Verified diagram ↔ generator pairing
- Description: Replace positional pairing with by-value matching (F4/F4a/F4b). Pure function
  `pair_generators(gens, dgm, D) -> list[PairedRow]`, testable on synthetic arrays.
  Ties among rows identical in birth *and* death are interchangeable — claim the lowest
  unclaimed index, never error. Error only on zero matches.
- Files likely touched: `repcycles/pairing.py`, `repcycles/core.py`, `tests/test_pairing.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC4 a/b/c all pass, tested as a pure function on synthetic arrays.
  - Congruent hexagonal holes (three identical `[1.0, √3]` rows) → 3 features, no error.
  - Tolerance is `rtol=1e-5, atol=1e-9`; a test documents why (float32 vs float64).
- Status: done

## T5: Essential H₁ classes
- Description: Read `gens[3][0]` (shape `(k, 2)` — birth edge only, verified by probe), emit
  features with `death=inf`, `persistence=inf`, `is_essential=True`, reconstruct at birth
  radius (F1, F2). `include_essential=True` constructor flag (F3). Implement the total sort
  order of F8a so multiple essential features have defined order.
- Files likely touched: `repcycles/core.py`, `tests/test_extraction.py`
- Depends on: T4
- Security-sensitive: no
- Acceptance criteria:
  - AC1: 60-point circle, `max_edge_length=1.0` → exactly 1 feature, essential, ≥3 edges.
  - `include_essential=False` reproduces today's feature list exactly.
  - `min_persistence` never filters an essential feature.
  - Two essential features sort deterministically (birth ascending).
- Status: done

## T6: Cycle verification and metadata
- Description: `cycle_length`, `cycle_path` (traversal order, first vertex repeated), and
  `is_verified` computed from the even-degree check plus the max-edge ≤ birth-radius check
  (F5, F6). Replace the silent birth-edge fallback's `print` with a
  `CycleReconstructionWarning` and `is_verified=False` (F7). Deterministic tie-breaking (F8).
- Files likely touched: `repcycles/reconstruction.py`, `tests/test_reconstruction.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC5: circle/figure-eight/torus/annulus fixtures all `is_verified=True`; `cycle_length`
    equals the summed edge lengths within 1e-9.
  - AC6: two fits produce `np.array_equal` cycle edges.
  - Disconnected input triggers `CycleReconstructionWarning` and `is_verified=False`; no `print`.
  - `cycle_path` traverses the loop and closes on itself.
- Status: done

## T7: Reconstruction performance — build the graph once
- Description: Replace the per-feature dense `n × n` mask with one filtration-sorted edge
  array per fit (float32 keys, int32 indices per the memory requirement), sliced by
  `searchsorted`; process features in ascending birth radius and rebuild the CSR only when
  the cut index changes.
- Files likely touched: `repcycles/reconstruction.py`, `tests/test_reconstruction_perf.py`
- Depends on: T6
- Security-sensitive: no
- Acceptance criteria:
  - AC13: in-process ratio ≥ 2× against the legacy dense-mask path on the 1500-point torus,
    marked `@pytest.mark.perf`. **No wall-clock assertion.**
  - Equivalence asserted on total geodesic length (1e-9) and `is_verified`, not on edge arrays.
  - AC13a: dtypes are float32/int32 and `nbytes` within 2× the dense mask.
- Status: in-progress

## T8: `to_dataframe()` and a verification column in `summary()`
- Description: F9 export with lazy pandas import and a clear `ImportError`; `summary()` gains
  a verified column and renders `inf` deaths as `∞` without breaking column alignment.
- Files likely touched: `repcycles/core.py`, `tests/test_export.py`
- Depends on: T5, T6
- Security-sensitive: no
- Acceptance criteria:
  - One row per feature; columns cover every scalar field plus `n_edges`.
  - Essential features render as `inf` in the frame and `∞` in `summary()`.
  - Absent pandas raises `ImportError` naming the install command (simulated via import hook).
- Status: done

## T9: Best-fit-plane projection helper
- Description: V4's projection: SVD of the cycle's centred vertices, applied to the whole
  cloud so context is preserved; returns coordinates plus retained-variance fraction.
  Degenerate cases (<3 vertices, near-zero second singular value) fall back to the first two
  coordinates and report retained variance honestly.
- Files likely touched: `repcycles/projection.py`, `tests/test_projection.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC8 absolute: top 6 torus cycles each retain ≥ 0.90.
  - AC8 relative: retained variance ≥ the `X[:, :2]` value for every cycle on every 3-D
    fixture (SVD-optimality invariant).
  - Collinear and 3-vertex cycles return a usable projection without raising.
  - 2-D input is passed through unchanged.
- Status: done

## T10: Persistence-diagram panel
- Description: Fix the highlight index bug (V1) — full-diagram indices are currently used
  against finite-filtered arrays. Draw infinite bars on a labelled "∞" band above the finite
  range. Take colours from `cycle_colors` (V2).
- Files likely touched: `repcycles/plotting/diagram.py`, `tests/test_plotting_diagram.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC11: a fit containing an essential bar highlights the correct point, no `IndexError`.
  - Highlighted points correspond exactly to `features_` for any finite/infinite mix.
  - `inf` never leaks into an axis limit.
- Status: done

## T11: Barcode view
- Description: V6 — infinite bars as right-pointing arrows, markers on bars present in
  `features_`, shared colours, sorted by persistence with `inf` first.
- Files likely touched: `repcycles/plotting/barcode.py`, `tests/test_plotting_barcode.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC10: essential bar renders an arrow artist, no raise.
  - Bar colours match `cycle_colors` after `to_hex` normalisation.
  - `pers.max()` normalisation does not divide by `inf` or `0`.
- Status: done

## T12: Cycle panels with 3-D-aware projection
- Description: Rewire `plot_matplotlib`'s per-cycle panels onto `projection.py` (V4), annotate
  each panel with retained variance, use shared colours, and drop deprecated matplotlib APIs
  (V10) including the `cm.tab10(np.linspace(...))` misuse. Never call `plt.show()` (V11).
- Files likely touched: `repcycles/plotting/panels.py`, `tests/test_plotting_panels.py`
- Depends on: T9, T10
- Security-sensitive: no
- Acceptance criteria:
  - 3-D fixtures render panels using the per-cycle plane; the variance annotation is present.
  - AC15: no `DeprecationWarning` under `-W error::DeprecationWarning`.
  - Returns a figure; `plt.show` is never called (assert via monkeypatch).
- Status: in-progress

## T13: `plot_overview()` and `plot_cycle()`
- Description: V5 — one figure combining cloud + all cycles overlaid + diagram + barcode,
  colour-linked. V8 — `plot_cycle(index)` at full size with local context (neighbours within
  the birth radius).
- Files likely touched: `repcycles/plotting/overview.py`, `tests/test_plotting_overview.py`
- Depends on: T10, T11, T12
- Security-sensitive: no
- Acceptance criteria:
  - AC9 (matplotlib half): colour of feature *k* identical across panels, barcode, overview.
  - AC12 (matplotlib half): `plot_cycle(0)` returns a figure for 2-D and 3-D input.
  - `plot_overview()` on a fit with zero features renders an empty-state figure, not a crash.
- Status: todo

## T14: Interactive plotly view
- Description: V7 hover metadata on every cycle trace (index, birth, death, persistence,
  cycle length, edge count, verified). V9 `show_skeleton` with the 20 000-edge budget,
  shortest-edges-first truncation, and the mandatory truncation annotation. Shared colours.
- Files likely touched: `repcycles/plotting/interactive.py`, `tests/test_plotting_plotly.py`
- Depends on: T2
- Security-sensitive: no
- Acceptance criteria:
  - AC12 (plotly half): `plot_plotly(show_skeleton=True)` returns a figure for 2-D and 3-D.
  - Hover text contains all seven fields.
  - Exceeding the edge budget annotates the figure; the drawn edges are the shortest ones.
  - AC9 (plotly half): trace colours match `cycle_colors` after `to_hex` normalisation.
- Status: done

## T15: Examples, README, regenerated outputs
- Description: Add an essential-bar example and an overview-figure example to `examples.py`;
  update `README.md` for the new API surface, the Breaking Changes section, and the
  verification story; regenerate `output/` and `assets/demo.png`.
- Files likely touched: `examples.py`, `zhu_paper_replication.py`, `README.md`, `output/**`, `assets/demo.png`
- Depends on: T13, T14, T8
- Security-sensitive: no
- Acceptance criteria:
  - AC16: both scripts run to completion; Zhu replication reports all five examples passing.
  - README documents every new parameter, field, and method, and the B1–B5 breaking changes.
  - No README claim is made that the tests don't verify.
- Status: todo

## T16: Integration gate
- Description: Whole-suite verification against the acceptance criteria, cross-lane colour
  consistency (AC9 end-to-end), the deprecation gate (AC15), and public-API compatibility
  (AC14). This is the task that catches what passed in isolation but breaks in composition.
- Files likely touched: `tests/test_integration.py`, `pyproject.toml` (pytest markers)
- Depends on: T3, T5, T7, T8, T12, T13, T14, T15
- Security-sensitive: no
- Acceptance criteria:
  - All 16 acceptance criteria in `spec.md` map to a passing test, listed explicitly.
  - Full suite green, including the pre-existing 35 tests.
  - `pytest -W error::DeprecationWarning` passes.
- Status: todo
