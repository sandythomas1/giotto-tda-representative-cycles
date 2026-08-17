# Project Constitution

Durable engineering principles for this project. Unlike specs (which describe one feature), this file rarely changes — treat edits to it as deliberate, discussed decisions, not implementation detail.

## Engineering Standards
- Code is written for a senior-engineering bar: readable, scalable, secure, idiomatic to the stack.
- No code ships without unit tests covering its behavior, including edge cases and failure modes.
- Every subtask implementation includes a rationale + tradeoffs summary.

## Scientific Correctness Bar

This is a computational-topology library. Wrong output is worse than no output, because a
plausible-looking loop drawn on a point cloud will be believed.

- Any claim about a topological object (a cycle, a bar, a Betti number) must be either
  provable from the algorithm or verified at runtime and reported as unverified.
- Heuristics are allowed, but must be named as heuristics in the docstring, with a statement
  of what they do and do not guarantee.
- Analytically-known examples (e.g. the Zhu §2.3 rectangle: birth = 2, death = √5) are
  regression tests, not demos. They must stay green.
- Silent fallbacks are forbidden. If reconstruction degrades, the degraded result must be
  flagged in the returned data structure, not just in a print statement.

## Security Bar
- Treat all external input as untrusted. Validate at system boundaries.
- No secrets, credentials, or tokens committed to the repo or logged.
- Every `/spec-implement` run triggers an automated security review pass before a task is marked done.
- OWASP Top 10 classes are the minimum bar for review (injection, auth, exposure, SSRF, etc.).
- For this project specifically, the realistic attack surface is *untrusted numeric input*:
  NaN/inf coordinates, adversarially large `n` (memory exhaustion via O(n²) distance
  matrices), non-finite distance matrices, and file paths passed to `save_path` /
  `save_html`. Validate these at `fit()` and at plot entry points.

## Testing Bar
- Unit tests are mandatory for new logic; prefer testing behavior over implementation detail.
- Test framework: **pytest** (detected: `tests/test_representative_cycles.py`, `[tool.pytest.ini_options]` in `pyproject.toml`, `pytest>=7.0` dev extra).
- Numerical assertions use explicit tolerances; no bare `==` on floats.
- Plotting code is tested with the `Agg` backend for artifact structure (trace counts, colours,
  data ranges) — not by eyeballing images.
- Coverage expectations: every public method on `RepresentativeCycles` has at least one
  behavioural test; every documented failure mode has a test that triggers it.

## Architecture Principles
- The public import path `from representative_cycles import RepresentativeCycles` is a
  **stable contract**. Internal reorganisation must preserve it.
- Topology computation and visualisation are separate concerns and live in separate modules.
  Nothing in the computation layer may import matplotlib or plotly at module scope.
- Optional heavy dependencies (plotly, pandas, scikit-learn) are imported lazily inside the
  functions that need them, so the core stays importable without them.

## Tech Stack
- Python ≥ 3.9 (dev environment: CPython 3.12.9, Windows).
- `gph-ripser` (giotto-ph) for Vietoris-Rips persistent homology with `return_generators=True`.
- NumPy ≥ 1.24, SciPy ≥ 1.10 (`scipy.sparse.csgraph` for Dijkstra), scikit-learn ≥ 1.3 (MDS).
- matplotlib ≥ 3.7 (static views), plotly ≥ 5.0 (interactive views).
- pytest for tests. Packaging via setuptools/`pyproject.toml`.

## Decision Log Pointer
See `specs/memory/decisions.md` for the running log of significant architectural decisions and their rationale.
