# Learnings

Concepts, patterns, and AI-native practices picked up while building this project. This is a running curriculum — review it periodically.

<!-- New entries go at the top, most recent first -->

## 2026-08-17 — A half-finished optimisation can leave a green suite
`RipsGraphCache` was fully written, fully documented, and completely dead: it called a
`_floor_to_float32` helper that did not exist, and `core.fit()` never passed a cache, so
the suite was green while the code path had never executed once. Two habits catch this,
and both are now in the suite: a test that asserts the *fast path is the one taken*
(`fit()` builds exactly one cache), and an equivalence test that forces the old path
through the same code (monkeypatching the threshold) and compares results. Coverage of a
module is not evidence that the module is reachable.

## 2026-08-17 — Measure the criterion before writing the assertion
AC13a demanded the sorted-edge structure stay "within 2× the dense mask", and the class
docstring said it was ~6× *worse* when unbounded. Both were true: the answer was that
`fit()` caps the edge list at the largest birth radius any feature needs, which on the
1500-point torus is 247 KB against the mask's 2.25 MB. Measuring first turned an
apparent contradiction in the spec into the design constraint that resolved it —
guessing would have produced either a weakened criterion or a wrong implementation.

## 2026-08-16 — Probe before you spec
Every requirement in spec 001 was derived from a measurement, not from reading the code and
guessing. Three throwaway probe scripts produced the numbers that became acceptance criteria:
essential bars vanishing under `max_edge_length` (0 features found where 1 exists), 2-D
projection variance loss on torus tube-loops (0.50–0.69 kept vs 0.92–0.98 for the best-fit
plane), and reconstruction cost (2.37 s at n=1500). A spec whose acceptance criteria are
measured numbers is testable; one whose criteria are adjectives is not.
