# Interface Contract: Spec 001

Binding signatures for parallel implementation. Tasks are implemented concurrently by
separate agents, each owning its own files, so these signatures are the integration surface.
**Do not change a signature here without saying so explicitly in your report** — another
task is already coding against it.

Existing shared types: `repcycles.feature.CycleFeature`,
`repcycles.errors.{CycleReconstructionError, CycleReconstructionWarning}`,
`repcycles.palette.cycle_colors(n_or_features) -> list[str]` (hex strings).

---

## T3 — `repcycles/validation.py`

```python
def validate_point_cloud(X: np.ndarray) -> np.ndarray:
    """Return float64 (n, d) coords. Rejects: not 2-D, NaN, ±inf, n < 1."""

def validate_precomputed(D: np.ndarray) -> np.ndarray:
    """Return float64 (n, n). Rejects: non-square, asymmetric, non-zero diagonal,
    NaN, negative. ACCEPTS +inf (means 'no edge')."""

def check_size(n: int, max_points: int | None = None) -> None:
    """ResourceWarning above SIZE_WARN_THRESHOLD (5000); ValueError if n > max_points."""

def validate_save_path(path: str | None) -> str | None:
    """Return path unchanged; ValueError naming the missing parent directory."""

SIZE_WARN_THRESHOLD: int = 5000
```

Error messages must name the offending index (`"NaN at row 17, column 2"`).

## T4 — `repcycles/pairing.py`

```python
@dataclass(frozen=True)
class PairedRow:
    generator_index: int      # row within its own generator array
    diagram_index: int        # claimed row of the H1 diagram
    birth: float              # taken from the diagram row
    death: float              # np.inf for essential
    birth_edge: tuple[int, int]
    death_edge: tuple[int, int]   # (-1, -1) for essential
    is_essential: bool

def pair_generators(
    finite_gens: np.ndarray,      # (k, 4) [b_v0, b_v1, d_v0, d_v1]
    essential_gens: np.ndarray,   # (m, 2) [b_v0, b_v1]
    diagram: np.ndarray,          # (n, 2) H1 diagram, may contain inf deaths
    dist_matrix: np.ndarray,      # (n_pts, n_pts) float64
    rtol: float = 1e-5,
    atol: float = 1e-9,
) -> list[PairedRow]:
    """Match each generator to an unclaimed diagram row by birth-edge length.
    Ties (rows equal in birth AND death) are interchangeable: claim the lowest
    unclaimed index. Zero matches -> CycleReconstructionError."""
```

## T6 / T7 — `repcycles/reconstruction.py`

```python
@dataclass
class CycleResult:
    edges: np.ndarray        # (n_edges, 2) int
    path: np.ndarray         # (n_vertices + 1,) int, closed loop order
    length: float            # sum of edge lengths
    is_verified: bool

def reconstruct_cycle(
    dist_matrix: np.ndarray,
    birth_edge: tuple[int, int],
    birth_radius: float,
    graph_cache: "RipsGraphCache | None" = None,
) -> CycleResult:
    """Shortest u->v path at birth radius excluding the birth edge, closed with it."""

class RipsGraphCache:                 # T7
    def __init__(self, dist_matrix: np.ndarray, max_edge_length: float = np.inf): ...
    def graph_at(self, radius: float): ...   # -> csr_matrix, rebuilt only when the cut moves
```

**T6 must keep `reconstruct_cycle` working with `graph_cache=None`** (build per call) so T7
can be added without changing callers. T6 owns the file first; T7 edits it afterwards.

Existing callers use the old return type (a bare edge array). T6 changes the return type to
`CycleResult`; `repcycles/core.py` is updated by the integrator, not by T6.

## T9 — `repcycles/projection.py`

```python
@dataclass(frozen=True)
class Projection:
    coords: np.ndarray            # (n_points, 2) — ALL points, not just the cycle
    variance_retained: float      # fraction of the CYCLE's variance kept, in [0, 1]
    baseline_retained: float      # same fraction under the naive X[:, :2]
    is_degenerate: bool           # fell back (too few vertices / rank-deficient)

def project_for_cycle(X: np.ndarray, cycle_vertices: np.ndarray) -> Projection:
    """Best-fit plane through the cycle's vertices (SVD), applied to the whole cloud so
    the loop keeps its context. 2-D input passes through with variance_retained == 1.0."""
```

## T10 — `repcycles/plotting/diagram.py`

```python
def draw_persistence_diagram(
    ax,
    diagram: np.ndarray,            # (n, 2) H1 diagram, may contain inf
    features: list[CycleFeature],   # the shown subset
    colors: list[str] | None = None,
) -> None:
```

`features[k].index` is a row index into `diagram`. Highlight by that row index — the current
bug is indexing the *finite-filtered* arrays with it.

## T11 — `repcycles/plotting/barcode.py`

```python
def plot_barcode(
    diagram: np.ndarray,
    features: list[CycleFeature] | None = None,
    colors: list[str] | None = None,
    figsize: tuple[int, int] = (8, 4),
    save_path: str | None = None,
):  # -> matplotlib Figure
```

## T12 — `repcycles/plotting/panels.py`

```python
def plot_matplotlib(rc, max_cycles=6, figsize=None, title=..., save_path=None): ...
def draw_cycle_panel(ax, X, feature, color: str, rank: int, projection=None) -> None: ...
```

## T13 — `repcycles/plotting/overview.py`

```python
def plot_overview(rc, max_cycles=6, figsize=None, title=..., save_path=None): ...
def plot_cycle(rc, index: int, figsize=(7, 7), save_path=None, context_radius=None): ...
```

## T14 — `repcycles/plotting/interactive.py`

```python
def plot_plotly(
    rc,
    max_cycles: int = 6,
    title: str = "Representative H₁ Cycles (Interactive)",
    save_html: str | None = None,
    show_skeleton: bool = False,
    skeleton_max_edges: int = 20_000,
):  # -> plotly Figure
```

Reads `rc.point_cloud_`, `rc.features_`, `rc._dist_matrix_` (needed for the skeleton).

---

## Rules for every task

1. **Only touch the files your task lists.** `repcycles/core.py` belongs to the integrator;
   if your change needs a core wiring change, say so in your report instead of editing it.
2. Tests go in the test file named by your task; do not edit
   `tests/test_representative_cycles.py` (it is the backwards-compatibility gate).
3. Run: `& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pytest tests/<your file> -q`
   (`python` is not on PATH). Before finishing, also run the full suite and report — but do
   **not** fix failures in files you don't own; report them.
4. Plot tests use the `Agg` backend and assert artifact structure (artist counts, colours,
   data ranges) — never image comparison.
5. Follow `specs/constitution.md`: no silent degradation, heuristics labelled as heuristics,
   no `print` in library code, computation modules import no plotting stack.
