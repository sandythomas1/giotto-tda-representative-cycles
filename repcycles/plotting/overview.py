"""Composite overview and single-cycle figures (spec 001, V5 and V8).

Two views that the per-cycle panels cannot give:

**The overview (V5)** puts the cloud with *every* selected loop overlaid, the
persistence diagram, and the barcode into one figure, colour-linked.  A grid
of panels answers "what does cycle 3 look like?"; the overview answers "how do
these loops sit relative to each other, and which bars are they?" — the
question a reader actually opens the figure with.

The overlay pays a price the panels do not, and says so on the figure.  All
loops share one pair of axes, so they must share one plane; that plane is
fitted to the union of the drawn cycles' vertices, and it is optimal for the
group rather than for any individual loop.  A cycle that is well-captured in
its own panel can be foreshortened here.  The retained fraction is printed,
and the annotation names the per-cycle views as the place to check a loop
whose shape matters.

**The single-cycle figure (V8)** is the same panel :mod:`repcycles.plotting.
panels` draws, given the whole figure and one extra layer: the points within
the birth radius of the loop, picked out from the rest of the cloud.  That
neighbourhood is the material the shortest-path reconstruction actually had
to work with, so showing it is the difference between "here is a loop" and
"here is why this loop and not a tighter one".

Neighbourhoods are measured in the **original** distance matrix, never in the
projected plane: the projection is a viewing convenience, and a point that is
far away in the data must not be drawn into the neighbourhood by a rotation.

Nothing here calls ``plt.show()``; every entry point returns its figure (V11).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..feature import CycleFeature
from ..palette import cycle_colors
from ..projection import Projection, project_for_cycle
from ..validation import validate_save_path
from .barcode import draw_barcode
from .diagram import draw_persistence_diagram
# The underscore-prefixed names are shared within the plotting package
# deliberately: cloud coercion, diagram lookup and `max_cycles` validation must
# behave identically in every view, and a second copy would drift.
from .panels import (
    CLOUD_COLOR,
    _as_cloud,
    _h1_diagram,
    _validate_max_cycles,
    draw_cycle_overlay,
    draw_cycle_panel,
)

__all__ = [
    "plot_overview",
    "plot_cycle",
    "GID_OVERVIEW_CLOUD",
    "GID_OVERVIEW_NOTE",
    "GID_CONTEXT",
    "GID_EMPTY_BARCODE",
]

# Stable artist identifiers, so tests and callers can find artists without
# depending on child ordering.
GID_OVERVIEW_CLOUD = "overview-cloud"
GID_OVERVIEW_NOTE = "overview-projection-note"
GID_CONTEXT = "cycle-context"
GID_EMPTY_BARCODE = "barcode-empty"

#: Colour of the points inside the focused cycle's birth radius.  Darker than
#: :data:`repcycles.plotting.panels.CLOUD_COLOR` so the neighbourhood reads as
#: a distinct layer between the background cloud and the loop.
CONTEXT_COLOR = "#8A8A8A"


def plot_overview(
    rc,
    max_cycles: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
    title: str = "Representative H₁ Cycles — Overview",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cloud with all cycles overlaid, plus the diagram and the barcode (V5).

    Parameters
    ----------
    rc : RepresentativeCycles
        A fitted model.  Read-only: ``point_cloud_``, ``features_``,
        ``diagrams_``.
    max_cycles : int, default 6
        Maximum number of cycles to overlay, taken from the front of
        ``features_`` (sorted most-persistent-first).  The diagram and the
        barcode still show **every** class; only the highlighting is limited,
        so the figure never implies the omitted classes do not exist.
    figsize : tuple, optional
        ``(width, height)`` in inches.  Defaults to ``(14, 7)``.
    title : str
        Figure super-title.
    save_path : str, optional
        If given, the figure is written here at 150 DPI.  Validated before any
        drawing, so a missing directory is named rather than discovered after
        the render.

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
        not writable.

    Notes
    -----
    Colours come from :func:`repcycles.palette.cycle_colors` and are passed to
    all three sub-views, so cycle *k* is the same colour in the overlay, the
    diagram and the barcode, and the same colour it has in
    :func:`repcycles.plotting.panels.plot_matplotlib` and the plotly view
    (V2, AC9).

    A fit with no features still returns a figure: the cloud is drawn, and the
    diagram and barcode render their empty states.  A figure that says "there
    is nothing here" is a result; a traceback at the end of a long fit is not.
    """
    if getattr(rc, "point_cloud_", None) is None:
        raise RuntimeError("Call fit() before plot_overview().")

    n_requested = _validate_max_cycles(max_cycles)
    save_path = validate_save_path(save_path)

    X = _as_cloud(rc.point_cloud_)
    cycles: List[CycleFeature] = list(rc.features_)[:n_requested]
    colors = cycle_colors(cycles)
    diagram = _h1_diagram(rc)

    # Constrained layout rather than `tight_layout`: the persistence diagram
    # carries a colorbar, and a colorbar axes inside a nested gridspec is
    # exactly the case `tight_layout` warns it cannot handle.
    fig = plt.figure(figsize=figsize or (14, 7), layout="constrained")
    # The overlay is the subject of this figure, so it gets the wider column
    # and the full height; the two summary views stack beside it.
    grid = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0])
    ax_cloud = fig.add_subplot(grid[:, 0])
    ax_diagram = fig.add_subplot(grid[0, 1])
    ax_barcode = fig.add_subplot(grid[1, 1])

    _draw_overlay(ax_cloud, X, cycles, colors)
    draw_persistence_diagram(ax_diagram, diagram, cycles, colors)
    _draw_barcode_or_empty(ax_barcode, diagram, cycles, colors)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cycle(
    rc,
    index: int,
    figsize: Tuple[int, int] = (7, 7),
    save_path: Optional[str] = None,
    context_radius: Optional[float] = None,
) -> plt.Figure:
    """One cycle at full figure size, with its local context (V8).

    Parameters
    ----------
    rc : RepresentativeCycles
        A fitted model.
    index : int
        Position in ``rc.features_`` — 0 is the most persistent cycle.  This
        is the *rank*, not the feature's diagram row (``feature.index``).
    figsize : tuple, default (7, 7)
        ``(width, height)`` in inches.  Square by default: the axes are
        equal-aspect, so a square figure wastes the least space around a loop.
    save_path : str, optional
        If given, the figure is written here at 150 DPI.
    context_radius : float, optional
        Radius of the neighbourhood drawn around the loop, measured in the
        **original** metric.  Defaults to the cycle's birth radius — the
        radius at which the class appeared, and therefore the graph the
        reconstruction actually searched.  Must be finite and non-negative.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    RuntimeError
        If ``rc`` has not been fitted.
    IndexError
        If ``index`` is not a position in ``features_``.  The message names
        the valid range; a fit with no features says so explicitly, because
        "index 0 out of range" reads as a bug in the caller when it is
        usually a fit that found nothing.
    TypeError, ValueError
        If ``index`` or ``context_radius`` is the wrong type or out of range,
        or ``save_path`` is not writable.

    Notes
    -----
    The colour is taken from the full ``features_`` list, not from the
    single-element slice, so a cycle keeps the colour it has in every other
    view of the same fit (AC9).

    The projection is this cycle's **own** best-fit plane, as in the panel
    view — unlike :func:`plot_overview`, nothing is shared here, so the loop
    is shown in the plane that suits it best.
    """
    if getattr(rc, "point_cloud_", None) is None:
        raise RuntimeError("Call fit() before plot_cycle().")

    features: List[CycleFeature] = list(rc.features_)
    position = _validate_index(index, len(features))
    save_path = validate_save_path(save_path)

    feature = features[position]
    color = cycle_colors(features)[position]
    X = _as_cloud(rc.point_cloud_)

    radius = (
        _birth_radius(rc, feature)
        if context_radius is None
        else _validate_context_radius(context_radius)
    )
    neighbours = _neighbourhood(rc, X, feature, radius)

    fig, ax = plt.subplots(figsize=figsize)
    projection = project_for_cycle(X, feature.cycle_vertices)
    draw_cycle_panel(
        ax, X, feature, color, rank=position + 1, projection=projection
    )
    _draw_context(ax, projection.coords, neighbours, radius)

    # The panel title is sized for a grid of small panels; this figure is one
    # panel, so it can afford to be read at arm's length.
    ax.set_title(ax.get_title(), fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Overview internals
# ---------------------------------------------------------------------------


def _draw_overlay(
    ax: Axes,
    X: np.ndarray,
    cycles: Sequence[CycleFeature],
    colors: Sequence[str],
) -> None:
    """Draw the cloud with every selected loop on one shared projection."""
    projection = _shared_projection(X, cycles)
    coords = projection.coords

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=10,
        c=CLOUD_COLOR,
        linewidths=0.0,
        zorder=1,
        gid=GID_OVERVIEW_CLOUD,
    )

    for feature, color in zip(cycles, colors):
        draw_cycle_overlay(ax, coords, feature, color)

    ax.set_title(
        f"Point cloud with {len(cycles)} representative "
        f"{'cycle' if len(cycles) == 1 else 'cycles'}",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    _annotate_shared_projection(ax, X.shape[1], projection, len(cycles))
    _add_overlay_legend(ax, cycles, colors)


def _shared_projection(
    X: np.ndarray, cycles: Sequence[CycleFeature]
) -> Projection:
    """One plane for every drawn loop, fitted on the union of their vertices.

    Per-cycle planes cannot be combined — two loops projected through
    different planes share no coordinate system, and drawing them on one pair
    of axes would place them relative to each other arbitrarily.  So the
    overlay fits a single plane to all the vertices it is about to draw.  The
    cost is stated on the figure by :func:`_annotate_shared_projection`.
    """
    if not cycles:
        # No vertices to fit on: project_for_cycle falls back to the first two
        # coordinates and reports itself degenerate, which is the truth.
        return project_for_cycle(X, np.empty(0, dtype=int))

    vertices = np.unique(
        np.concatenate(
            [np.asarray(f.cycle_vertices, dtype=int).ravel() for f in cycles]
        )
    )
    return project_for_cycle(X, vertices)


def _annotate_shared_projection(
    ax: Axes, n_dims: int, projection: Projection, n_cycles: int
) -> None:
    """State that this plane is shared, and what it kept.

    Deliberately different wording from the panel annotation: the number here
    describes the *group*, and a reader who takes it for a per-cycle fit would
    over-trust the shape of an individual loop.
    """
    if n_dims <= 2:
        message = f"{n_dims}-D input: shown as recorded"
    elif n_cycles == 0:
        message = "no cycles to project: first two coordinates shown"
    elif projection.is_degenerate:
        message = (
            "shared plane could not be fitted: first two coordinates shown\n"
            "see plot_cycle() for each loop in its own plane"
        )
    else:
        message = (
            f"shared best-fit plane over all {n_cycles} loops keeps "
            f"{projection.variance_retained * 100:.1f}% of their variance\n"
            f"(first two coords: {projection.baseline_retained * 100:.1f}%) — "
            "individual loops may be foreshortened; see plot_cycle()"
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
        gid=GID_OVERVIEW_NOTE,
    )


def _add_overlay_legend(
    ax: Axes, cycles: Sequence[CycleFeature], colors: Sequence[str]
) -> None:
    """Name each overlaid loop by its rank, in its own colour.

    Without this the overlay is a tangle of coloured lines with no key.  The
    legend is capped: past a handful of entries it stops being a key and
    starts being a wall, and the barcode beside it already carries the
    per-cycle labels.
    """
    if not cycles:
        return

    handles = [
        Line2D(
            [],
            [],
            color=color,
            lw=2,
            ls="solid" if feature.is_verified else "dotted",
            label=(
                f"#{rank} H₁ {feature.index}"
                + ("" if feature.is_verified else " (unverified)")
            ),
        )
        for rank, (feature, color) in enumerate(zip(cycles, colors), start=1)
    ]
    ax.legend(handles=handles, fontsize=7, loc="best", ncol=2 if len(handles) > 4 else 1)


def _draw_barcode_or_empty(
    ax: Axes,
    diagram: Optional[np.ndarray],
    cycles: Sequence[CycleFeature],
    colors: Sequence[str],
) -> None:
    """Draw the barcode, or its empty state when there is no diagram.

    :func:`repcycles.plotting.barcode.draw_barcode` treats an empty diagram as
    an error — correctly, since a caller asking for a barcode of nothing has
    made a mistake.  Inside a composite figure the judgement flips: the other
    panels still have something to say, so the emptiness is reported in place.
    The check is on the diagram itself rather than on a caught ``ValueError``,
    which would also swallow a malformed diagram.
    """
    if diagram is None or len(np.asarray(diagram)) == 0:
        ax.set_title("H₁ Barcode", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No H₁ classes in this filtration",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
            gid=GID_EMPTY_BARCODE,
        )
        return

    draw_barcode(ax, diagram, cycles, colors)
    ax.set_title(ax.get_title(), fontsize=11)


# ---------------------------------------------------------------------------
# Single-cycle internals
# ---------------------------------------------------------------------------


def _draw_context(
    ax: Axes, coords: np.ndarray, neighbours: np.ndarray, radius: float
) -> None:
    """Pick the birth-radius neighbourhood out of the background cloud.

    Drawn above the cloud and below the loop, and added to the panel's own
    legend rather than replacing it, so the reader keeps the "birth edge" and
    "unverified" keys the panel supplies.
    """
    if neighbours.size:
        ax.scatter(
            coords[neighbours, 0],
            coords[neighbours, 1],
            s=18,
            c=CONTEXT_COLOR,
            linewidths=0.0,
            zorder=1.5,
            gid=GID_CONTEXT,
        )

    label = f"within birth radius ({radius:.3f})"
    proxy = Line2D(
        [],
        [],
        color=CONTEXT_COLOR,
        marker="o",
        markersize=4,
        linestyle="none",
        label=label,
    )

    legend = ax.get_legend()
    handles = list(legend.legend_handles) if legend is not None else []
    labels = (
        [text.get_text() for text in legend.get_texts()]
        if legend is not None
        else []
    )
    ax.legend(
        handles=handles + [proxy], labels=labels + [label], fontsize=7, loc="best"
    )


def _birth_radius(rc, feature: CycleFeature) -> float:
    """The radius at which this class appeared.

    Read from the distance matrix when one is available, because that exact
    float64 length — not the float32-derived diagram birth, which can sit
    ~1e-7 below it — is the radius the reconstruction cut its graph at.  The
    diagram value is the fallback for a model that carries no matrix.
    """
    D = getattr(rc, "_dist_matrix_", None)
    if D is not None:
        u, v = (int(i) for i in feature.birth_edge)
        return float(np.asarray(D)[u, v])
    return float(feature.birth)


def _neighbourhood(
    rc, X: np.ndarray, feature: CycleFeature, radius: float
) -> np.ndarray:
    """Indices of points within *radius* of any vertex of the cycle.

    Measured in the original metric.  The fitted model already holds the
    distance matrix, so this is a slice; the ``cdist`` fallback exists only
    for a model that does not carry one, and costs an ``n × n_cycle``
    allocation rather than a full ``n × n``.
    """
    vertices = np.unique(np.asarray(feature.cycle_vertices, dtype=int))
    if vertices.size == 0:
        return np.empty(0, dtype=int)

    D = getattr(rc, "_dist_matrix_", None)
    if D is not None:
        distances = np.asarray(D)[np.ix_(vertices, np.arange(len(X)))]
    else:
        from scipy.spatial.distance import cdist

        distances = cdist(X[vertices], X)

    # `<=` includes the cycle's own vertices, which is intended: they are the
    # centre of their own neighbourhood, and the loop is drawn over them.
    within = np.flatnonzero((distances <= radius).any(axis=0))
    return within


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_index(index, n_features: int) -> int:
    """Return *index* as a position in ``features_``, or raise.

    Negative indices are rejected rather than wrapped: ``plot_cycle(-1)`` most
    likely means "the last one I looked at" and would silently draw the
    *least* persistent cycle, which is the opposite of what a reader expects
    from a most-persistent-first list.
    """
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError(
            f"index must be an integer position in features_; got "
            f"{type(index).__name__}."
        )
    index = int(index)
    if n_features == 0:
        raise IndexError(
            "this fit produced no features, so there is no cycle to plot. "
            "Lower min_persistence, or check summary() first."
        )
    if not 0 <= index < n_features:
        raise IndexError(
            f"index {index} is out of range: this fit has {n_features} "
            f"features, so valid indices are 0..{n_features - 1} "
            "(0 is the most persistent)."
        )
    return index


def _validate_context_radius(context_radius) -> float:
    """Reject a radius that would draw a meaningless neighbourhood."""
    radius = float(context_radius)
    if not np.isfinite(radius):
        raise ValueError(
            f"context_radius must be finite; got {context_radius!r}. "
            "Omit it to use the cycle's birth radius."
        )
    if radius < 0:
        raise ValueError(
            f"context_radius must be non-negative; got {radius}."
        )
    return radius
