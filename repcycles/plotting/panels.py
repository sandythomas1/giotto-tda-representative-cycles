"""Per-cycle panels and the composite matplotlib figure (spec 001, V2-V4, V10, V11).

Each panel answers one question: *where does this loop actually live in the
cloud?*  Three things had to change before it could answer honestly.

**The third coordinate is no longer thrown away (V4).**  Every panel used to
draw ``X[:, :2]``.  On the 600-point torus fixture that keeps only 0.46-0.67
of a tube-loop's variance: the loop is rendered as a near-degenerate smear
and a reader dismisses it as noise.  Panels now project through
:func:`repcycles.projection.project_for_cycle`, which fits the plane that is
optimal *for this cycle* (rank-2 truncated SVD) and applies it to the
**whole** cloud, so the loop keeps its surrounding context.  The retained
fraction is printed on the panel; when no plane could be fitted the panel
says so instead of quoting a number that would imply a good fit
(``specs/constitution.md``: a degraded result is flagged, never dressed up).

**Colour comes from the shared palette (V2/V3).**  ``cm.tab10(np.linspace(0,
1, k))`` sampled a *qualitative* colormap continuously, interpolating between
the designed categorical colours: muddy, not colourblind-safe, and different
every time the cycle count changed.  Colours are now taken by index from
:func:`repcycles.palette.cycle_colors`, the same list every other view uses.

**Unverified loops do not look like verified ones.**  ``is_verified=False``
means the reconstruction did not close over ℤ/2 or used an edge longer than
the birth radius.  Such a loop is drawn dotted and carries an explicit badge,
because a plausible-looking loop that is silently wrong is the worst output
this library can produce.

Artists carry stable ``gid`` values (the ``GID_*`` constants) so tests and
composite figures can find them without depending on child-artist ordering.
Nothing here calls ``plt.show()``; every entry point returns its figure (V11).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..feature import CycleFeature
from ..palette import cycle_colors
from ..projection import Projection, project_for_cycle
from ..validation import validate_save_path
from .diagram import draw_persistence_diagram

__all__ = [
    "plot_matplotlib",
    "draw_cycle_panel",
    "GID_CLOUD",
    "GID_CYCLE",
    "GID_BIRTH_EDGE",
    "GID_CYCLE_VERTICES",
    "GID_BIRTH_VERTICES",
    "GID_PROJECTION_NOTE",
    "GID_UNVERIFIED",
]

# Stable artist identifiers.
GID_CLOUD = "panel-cloud"
GID_CYCLE = "panel-cycle"
GID_BIRTH_EDGE = "panel-birth-edge"
GID_CYCLE_VERTICES = "panel-cycle-vertices"
GID_BIRTH_VERTICES = "panel-birth-vertices"
GID_PROJECTION_NOTE = "panel-projection-note"
GID_UNVERIFIED = "panel-unverified"

#: Colour of the context cloud behind the loop.  Light enough that the loop
#: reads as the subject, dark enough to stay visible when printed.
CLOUD_COLOR = "#C8C8C8"

#: Colour of the "unverified" badge.
UNVERIFIED_COLOR = "#B00020"

#: A verified loop is solid; an unverified one is dotted.  The distinction is
#: carried by line *style*, not only by text, so it survives being read at
#: thumbnail size.
_STYLE_VERIFIED = "solid"
_STYLE_UNVERIFIED = "dotted"

_INFINITY = "∞"


def plot_matplotlib(
    rc,
    max_cycles: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "Representative H₁ Cycles",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Persistence diagram + one projected panel per representative cycle.

    Parameters
    ----------
    rc : RepresentativeCycles
        A fitted model.  Read-only: ``point_cloud_``, ``features_`` and
        ``diagrams_``.
    max_cycles : int, default 6
        Maximum number of cycles to draw, taken from the front of
        ``features_`` (which is sorted most-persistent-first).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Defaults to a width that
        grows with the panel count.
    title : str
        Figure super-title.
    save_path : str, optional
        If given, the figure is written here at 150 DPI.  Checked before any
        drawing happens, so a missing directory is reported by name instead of
        wasting the render.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.  Never shown, never closed — the caller owns it (V11).

    Raises
    ------
    RuntimeError
        If ``rc`` has not been fitted.
    TypeError, ValueError
        If ``max_cycles`` is not a non-negative integer, or ``save_path`` is
        not writable (see :func:`repcycles.validation.validate_save_path`).

    Notes
    -----
    A fit with zero features still returns a figure: the persistence-diagram
    panel renders its own empty state.  An empty figure is a statement; a
    crash in a plotting call at the end of a long fit is not.
    """
    if getattr(rc, "point_cloud_", None) is None:
        raise RuntimeError("Call fit() before plot_matplotlib().")

    n_requested = _validate_max_cycles(max_cycles)
    save_path = validate_save_path(save_path)

    X = np.asarray(rc.point_cloud_, dtype=float)
    cycles: List[CycleFeature] = list(rc.features_)[:n_requested]
    colors = cycle_colors(cycles)

    n_panels = len(cycles) + 1  # +1 for the persistence diagram
    if figsize is None:
        figsize = (4 * n_panels + 2, 5)

    # squeeze=False keeps `axes` a 2-D array whatever the panel count, so the
    # zero-feature case needs no special casing.
    fig, axes_grid = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes_grid[0]

    draw_persistence_diagram(axes[0], _h1_diagram(rc), cycles, colors)

    for rank, (feature, color) in enumerate(zip(cycles, colors), start=1):
        draw_cycle_panel(axes[rank], X, feature, color, rank=rank)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def draw_cycle_panel(
    ax: Axes,
    X: np.ndarray,
    feature: CycleFeature,
    color: str,
    rank: int,
    projection: Optional[Projection] = None,
) -> None:
    """Draw one representative cycle over the point cloud onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.  Drawn onto in place; nothing is shown or saved.
    X : np.ndarray, shape (n_points, n_dims)
        The full point cloud.  Clouds with ``n_dims >= 3`` are projected onto
        this cycle's best-fit plane; 2-D clouds pass through unchanged.
    feature : CycleFeature
        The cycle to draw.  ``death`` and ``persistence`` may be ``inf``.
    color : str
        Hex colour for this cycle, from
        :func:`repcycles.palette.cycle_colors` — the same colour the cycle
        carries in the diagram, the barcode and the plotly view (V2).
    rank : int
        1-based position in the figure, used in the panel title.
    projection : Projection, optional
        A projection already computed for this cycle, so a composite figure
        can fit the plane once and share it.  Must cover every point of *X*.
        Defaults to :func:`repcycles.projection.project_for_cycle`.

    Raises
    ------
    ValueError
        If *X* is not a 2-D array, or a supplied *projection* does not have
        one row per point of *X*.
    IndexError
        If the feature references a point outside *X*.

    Notes
    -----
    The projected coordinates cover the **whole** cloud, not just the loop:
    a loop drawn without its context cannot be judged, and the plane is
    cheap to apply to every point once it has been fitted on the cycle.
    """
    X = _as_cloud(X)
    if projection is None:
        projection = project_for_cycle(X, feature.cycle_vertices)
    elif len(projection.coords) != len(X):
        raise ValueError(
            f"projection covers {len(projection.coords)} points but X has "
            f"{len(X)}; the projection must be computed for this cloud."
        )

    coords = projection.coords
    _check_indices(feature, len(coords))

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=10,
        c=CLOUD_COLOR,
        linewidths=0.0,
        zorder=1,
        gid=GID_CLOUD,
    )

    _draw_cycle_edges(ax, coords, feature, color)
    _draw_marked_vertices(ax, coords, feature, color)

    ax.set_title(_panel_title(feature, rank), fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale_view()

    _annotate_projection(ax, X.shape[1], projection)
    if not feature.is_verified:
        _annotate_unverified(ax)
    _add_legend(ax, color, feature.is_verified)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_cycle_edges(
    ax: Axes, coords: np.ndarray, feature: CycleFeature, color: str
) -> None:
    """Draw the loop, with its birth edge picked out in dashed black."""
    if feature.n_edges == 0:
        return

    birth_key = _edge_key(feature.birth_edge)
    loop_segments: List = []
    birth_segments: List = []
    for u, v in np.asarray(feature.cycle_edges, dtype=int):
        segment = [coords[u], coords[v]]
        if _edge_key((u, v)) == birth_key:
            birth_segments.append(segment)
        else:
            loop_segments.append(segment)

    if loop_segments:
        ax.add_collection(
            mc.LineCollection(
                loop_segments,
                colors=[color],
                linewidths=2.0,
                linestyles=(
                    _STYLE_VERIFIED if feature.is_verified else _STYLE_UNVERIFIED
                ),
                alpha=0.85,
                zorder=2,
                gid=GID_CYCLE,
            )
        )
    if birth_segments:
        ax.add_collection(
            mc.LineCollection(
                birth_segments,
                colors=["black"],
                linewidths=2.5,
                linestyles="dashed",
                zorder=3,
                gid=GID_BIRTH_EDGE,
            )
        )


def _draw_marked_vertices(
    ax: Axes, coords: np.ndarray, feature: CycleFeature, color: str
) -> None:
    """Highlight the cycle's vertices, and star the birth edge's endpoints."""
    vertices = np.asarray(feature.cycle_vertices, dtype=int)
    if vertices.size:
        ax.scatter(
            coords[vertices, 0],
            coords[vertices, 1],
            s=40,
            c=color,
            edgecolors="k",
            linewidths=0.4,
            zorder=4,
            gid=GID_CYCLE_VERTICES,
        )

    u, v = (int(i) for i in feature.birth_edge)
    ax.scatter(
        coords[[u, v], 0],
        coords[[u, v], 1],
        s=80,
        c="black",
        marker="*",
        zorder=5,
        gid=GID_BIRTH_VERTICES,
    )


def _add_legend(ax: Axes, color: str, is_verified: bool) -> None:
    """Explain the two encodings a reader cannot guess.

    Proxy artists are used, so the legend never disturbs artist counts.
    """
    label = "cycle path" if is_verified else "cycle path (unverified)"
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                lw=2,
                ls=_STYLE_VERIFIED if is_verified else _STYLE_UNVERIFIED,
                label=label,
            ),
            Line2D([], [], color="black", lw=2, ls="--", label="birth edge"),
        ],
        fontsize=7,
        loc="best",
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def _annotate_projection(
    ax: Axes, n_dims: int, projection: Projection
) -> None:
    """State which view this is and what it cost (V4).

    Three honest messages, never one dressed up as another:

    * ``n_dims <= 2`` — nothing was projected, so no fraction is quoted.
    * degenerate — no plane could be fitted; the fallback is named, and the
      fraction quoted is the fallback's own, not a claim about a fit.
    * otherwise — the fitted plane's retained fraction, alongside the
      ``X[:, :2]`` figure it replaces so the reader can see the difference.
    """
    if n_dims <= 2:
        message = f"{n_dims}-D input: shown as recorded"
    elif projection.is_degenerate:
        message = (
            "degenerate cycle: no plane fitted\n"
            f"first two coords keep "
            f"{_percent(projection.variance_retained)} of cycle variance"
        )
    else:
        message = (
            f"best-fit plane keeps "
            f"{_percent(projection.variance_retained)} of cycle variance\n"
            f"(first two coords: {_percent(projection.baseline_retained)})"
        )

    ax.text(
        0.02,
        0.98,
        message,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        zorder=6,
        gid=GID_PROJECTION_NOTE,
    )


def _annotate_unverified(ax: Axes) -> None:
    """Badge a loop that failed verification, so it cannot pass for one that
    did not (``specs/constitution.md``: no silent degradation)."""
    ax.text(
        0.98,
        0.02,
        "unverified loop",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color=UNVERIFIED_COLOR,
        zorder=6,
        gid=GID_UNVERIFIED,
    )


def _panel_title(feature: CycleFeature, rank: int) -> str:
    """Title text, safe for essential features.

    ``death`` and ``persistence`` may be ``inf``.  They are formatted, never
    arithmetically combined, so no ``inf - inf`` can turn into ``nan``:
    ``persistence`` is read from the feature rather than recomputed here.
    """
    return (
        f"Rank {rank} cycle  (H₁ #{feature.index})\n"
        f"birth={_filtration(feature.birth)}  "
        f"death={_filtration(feature.death)}  "
        f"p={_filtration(feature.persistence)}\n"
        f"{feature.n_edges} edges  len={_filtration(feature.cycle_length)}"
    )


def _filtration(value: float) -> str:
    """Render a filtration value, with ``inf`` as ``∞`` (never ``inf`` or ``nan``)."""
    number = float(value)
    if np.isnan(number):
        return "n/a"
    if np.isposinf(number):
        return _INFINITY
    if np.isneginf(number):
        return f"-{_INFINITY}"
    return f"{number:.3f}"


def _percent(fraction: float) -> str:
    return f"{float(fraction) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_max_cycles(max_cycles) -> int:
    """Reject anything that would silently draw the wrong panels.

    A negative value would slice features off the *end* of the list and
    quietly render a different set of cycles than the caller asked for.
    """
    if isinstance(max_cycles, bool) or not isinstance(
        max_cycles, (int, np.integer)
    ):
        raise TypeError(
            f"max_cycles must be an integer; got {type(max_cycles).__name__}."
        )
    max_cycles = int(max_cycles)
    if max_cycles < 0:
        raise ValueError(
            f"max_cycles must be non-negative; got {max_cycles}. "
            "Use 0 to draw the persistence diagram alone."
        )
    return max_cycles


def _as_cloud(X: np.ndarray) -> np.ndarray:
    """Return *X* as a float64 ``(n_points, n_dims)`` array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(
            f"X must be a 2-D array of shape (n_points, n_dims); got shape "
            f"{X.shape}."
        )
    return X


def _check_indices(feature: CycleFeature, n_points: int) -> None:
    """Reject vertex references that fall outside the cloud.

    Drawing a loop from a feature paired against a *different* cloud would
    produce a plausible figure of nothing at all, so it fails loudly.
    """
    referenced = [
        np.asarray(feature.cycle_vertices, dtype=int).ravel(),
        np.asarray(feature.cycle_edges, dtype=int).ravel(),
        np.asarray(feature.birth_edge, dtype=int).ravel(),
    ]
    for indices in referenced:
        if indices.size == 0:
            continue
        lo, hi = int(indices.min()), int(indices.max())
        if lo < 0 or hi >= n_points:
            offender = lo if lo < 0 else hi
            raise IndexError(
                f"feature references point {offender}, which is out of range "
                f"for a cloud of {n_points} points."
            )


def _edge_key(edge: Sequence[int]) -> Tuple[int, int]:
    """Undirected edge identity, so ``(u, v)`` and ``(v, u)`` compare equal."""
    u, v = int(edge[0]), int(edge[1])
    return (u, v) if u <= v else (v, u)


def _h1_diagram(rc) -> Optional[np.ndarray]:
    """The H₁ diagram of a fitted model, or ``None`` when there is none."""
    diagrams = getattr(rc, "diagrams_", None)
    if diagrams is None or len(diagrams) < 2:
        return None
    return diagrams[1]
