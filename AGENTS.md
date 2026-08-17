# giotto-tda-cycles — Agent Instructions

Representative H₁ cycle extraction and visualisation on top of `gph-ripser` / giotto-tda.

## Spec-Driven Pipeline

This project uses the spec-driven pipeline. **`specs/constitution.md` takes precedence over
the generic global defaults** — read it before starting any spec work here, in particular
its Scientific Correctness Bar (heuristics must be labelled; no silent fallbacks) and its
architecture rule that computation modules never import matplotlib or plotly at module scope.

- Specs live in `specs/NNN-slug/` (`spec.md`, `plan.md`, `tasks.md`).
- Running decision log: `specs/memory/decisions.md`.
- Running learnings log: `specs/memory/learnings.md`.

Commands: `/spec-new`, `/spec-review`, `/spec-decompose`, `/spec-implement`, `/spec-status`,
`/spec-retro`.

## Environment note

`python` is not on PATH in this environment. Use
`$env:LOCALAPPDATA\Programs\Python\Python312\python.exe` (PowerShell) to run tests and
examples.

## Stable contract

`from representative_cycles import RepresentativeCycles, CycleFeature` must keep working —
it is the documented public API in the README.
