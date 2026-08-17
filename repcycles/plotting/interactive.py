"""Interactive Plotly view of representative H₁ cycles.

Two things distinguish this view from the matplotlib panels:

* **Hover metadata (V7).**  Every cycle trace carries the full evidence for
  its feature — index, birth, death, persistence, geodesic length, edge count
  and the ``is_verified`` flag — so a reader can interrogate a loop without
  cross-referencing ``summary()``.
* **The Rips 1-skeleton underlay (V9).**  With ``show_skeleton=True`` the
  figure draws the graph the reconstruction algorithm actually walked, at the
  focused cycle's birth radius.  Seeing the input to the heuristic is the
  fastest way to judge whether its output is believable.

Colours come from :func:`repcycles.palette.cycle_colors`, not from a plotly
palette, so cycle *k* is the same colour here as in every matplotlib view
(V2/V3).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..feature import CycleFeature
from ..palette import cycle_colors

__all__ = [
    "plot_plotly",
    "SKELETON_TRACE_NAME",
    "DEFAULT_SKELETON_MAX_EDGES",
]

#: Legend/trace name prefix for the Rips 1-skeleton underlay.  Stable so that
#: callers (and tests) can identify the trace without matching a radius.
SKELETON_TRACE_NAME = "Rips 1-skeleton"

#: Default edge budget for the skeleton underlay.  A 1500-point cloud at a
#: typical birth radius has on the order of 10⁵ edges; drawing all of them
#: produces a figure the browser cannot pan.
DEFAULT_SKELETON_MAX_EDGES = 20_000

_SKELETON_COLOR = "#b8b8b8"
_SKELETON_OPACITY = 0.35
_CLOUD_COLOR = "lightgray"


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plot_plotly() requires plotly. Install it: pip install plotly"
        ) from exc
    return go


def _scatter(go, is_3d: bool, x, y, z, **kwargs):
    """Return a ``Scatter3d`` or ``Scatter`` trace with the same styling.

    The 2-D/3-D split is purely about which constructor takes a ``z``; every
    other keyword is identical, so it is funnelled through one place rather
    than duplicated down two branches.
    """
    if is_3d:
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)
    return go.Scatter(x=x, y=y, **kwargs)


def _format_filtration_value(value: float) -> str:
    """Render a filtration value, showing an infinite death as ``∞``."""
    return "∞" if not np.isfinite(value) else f"{value:.4f}"


def _hover_text(feature: CycleFeature) -> str:
    """The seven fields required by V7, as one plotly hover block.

    Everything a reader needs to decide whether to trust the loop is here,
    including ``verified`` — an unverified loop is reported, never hidden.
    """
    return "<br>".join(
        (
            f"H₁ feature #{feature.index}",
            f"birth: {feature.birth:.4f}",
            f"death: {_format_filtration_value(feature.death)}",
            f"persistence: {_format_filtration_value(feature.persistence)}",
            f"cycle length: {feature.cycle_length:.4f}",
            f"edges: {feature.n_edges}",
            f"verified: {'yes' if feature.is_verified else 'no'}",
        )
    )


def _legend_label(feature: CycleFeature) -> str:
    death = _format_filtration_value(feature.death)
    persistence = _format_filtration_value(feature.persistence)
    return (
        f"H₁ #{feature.index} | b={feature.birth:.3f} "
        f"d={death} p={persistence}"
    )


def _edge_line_coords(
    X: np.ndarray, edges: np.ndarray, n_axes: int
) -> List[np.ndarray]:
    """Flatten an ``(m, 2)`` edge list into plotly line coordinates.

    Each edge becomes ``[start, end, nan]``; the ``nan`` breaks the polyline
    so one trace can carry every edge as a disconnected segment.  Vectorised
    because the skeleton can hold tens of thousands of edges.
    """
    edges = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    m = len(edges)
    coords = []
    for axis in range(n_axes):
        flat = np.empty(3 * m, dtype=np.float64)
        flat[0::3] = X[edges[:, 0], axis]
        flat[1::3] = X[edges[:, 1], axis]
        flat[2::3] = np.nan
        coords.append(flat)
    while len(coords) < 3:
        coords.append(np.empty(0, dtype=np.float64))
    return coords


# ----------------------------------------------------------------------
# Rips 1-skeleton (V9)
# ----------------------------------------------------------------------


def _skeleton_edges(
    dist_matrix: np.ndarray, radius: float, max_edges: int
) -> Tuple[np.ndarray, int]:
    """Rips 1-skeleton edges at *radius*, truncated to the shortest *max_edges*.

    Returns ``(edges, n_total)`` where ``n_total`` is the number of edges that
    exist at *radius* before any truncation, so the caller can report the
    shortfall rather than hide it.

    Truncation keeps the **shortest** edges: the choice is deterministic (no
    sampling), and short edges are the ones that carry local structure, which
    is what the underlay is for.  Non-finite entries (``+inf`` = "no edge" in
    a precomputed matrix) fail the ``<= radius`` test and are excluded.
    """
    n = dist_matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    lengths = dist_matrix[rows, cols]

    kept = np.flatnonzero(lengths <= radius)
    n_total = int(kept.size)

    if n_total > max_edges:
        # Stable sort so equal-length edges break ties by vertex order,
        # making the drawn subset reproducible run to run.
        order = np.argsort(lengths[kept], kind="stable")[:max_edges]
        kept = kept[order]

    edges = np.column_stack((rows[kept], cols[kept])).astype(np.int32)
    return edges, n_total


def _add_skeleton(
    fig,
    go,
    X: np.ndarray,
    dist_matrix: np.ndarray,
    radius: float,
    is_3d: bool,
    max_edges: int,
) -> Optional[str]:
    """Add the underlay trace; return a truncation note, or ``None``.

    The note is not cosmetic: the constitution forbids silent degradation, so
    a truncated skeleton must say so on the figure itself.
    """
    edges, n_total = _skeleton_edges(dist_matrix, radius, max_edges)
    x, y, z = _edge_line_coords(X, edges, 3 if is_3d else 2)

    fig.add_trace(
        _scatter(
            go,
            is_3d,
            x,
            y,
            z,
            mode="lines",
            line=dict(color=_SKELETON_COLOR, width=1),
            opacity=_SKELETON_OPACITY,
            name=f"{SKELETON_TRACE_NAME} (r={radius:.3f})",
            hoverinfo="skip",
        )
    )

    if len(edges) < n_total:
        return f"skeleton truncated: {len(edges)} / {n_total} edges"
    return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_plotly(
    rc,
    max_cycles: int = 6,
    title: str = "Representative H₁ Cycles (Interactive)",
    save_html: Optional[str] = None,
    show_skeleton: bool = False,
    skeleton_max_edges: int = DEFAULT_SKELETON_MAX_EDGES,
):
    """Interactive Plotly figure with toggleable, hover-annotated cycle traces.

    Each cycle's edges and vertices share a legend group, so one click shows
    or hides the whole loop.  Supports 2-D and 3-D point clouds.

    Parameters
    ----------
    rc : RepresentativeCycles
        A fitted model.  ``rc.point_cloud_``, ``rc.features_`` and
        ``rc._dist_matrix_`` are read.
    max_cycles : int, default 6
        Draw at most this many features, in ``features_`` order.
    title : str
        Figure title.  A truncation note is appended to it when the skeleton
        underlay does not fit its budget.
    save_html : str, optional
        If given, the figure is also written to this path.
    show_skeleton : bool, default False
        Draw the Vietoris-Rips 1-skeleton at the **first shown cycle's** birth
        radius as a faint underlay, so the graph the reconstruction walked is
        visible beneath its output.  Off by default because the underlay is
        dense and slow to render.
    skeleton_max_edges : int, default 20000
        Edge budget for that underlay.  When the skeleton exceeds it, the
        **shortest** ``skeleton_max_edges`` edges are drawn and the figure
        title is annotated ``skeleton truncated: N / M edges``.  The
        truncation is never silent.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    RuntimeError
        If ``rc`` has not been fitted, or if ``show_skeleton=True`` and no
        distance matrix is available.
    ValueError
        If ``skeleton_max_edges`` is not a positive integer.

    Notes
    -----
    Hover text carries the full feature record, including ``verified``.  A
    loop with ``verified: no`` is drawn like any other but must not be trusted
    as a representative — see
    :class:`repcycles.errors.CycleReconstructionWarning`.
    """
    go = _require_plotly()

    if skeleton_max_edges < 1:
        raise ValueError(
            f"skeleton_max_edges must be a positive integer, "
            f"got {skeleton_max_edges!r}."
        )

    if rc.point_cloud_ is None:
        raise RuntimeError("Call fit() before plot_plotly().")

    X = np.asarray(rc.point_cloud_)
    is_3d = X.shape[1] >= 3
    cycles: Sequence[CycleFeature] = list(rc.features_[:max_cycles])
    colors = cycle_colors(len(cycles))

    fig = go.Figure()
    notes: List[str] = []

    # The skeleton is added first so every later trace draws on top of it.
    if show_skeleton:
        if not cycles:
            notes.append("skeleton not drawn: no features to focus on")
        elif getattr(rc, "_dist_matrix_", None) is None:
            raise RuntimeError(
                "show_skeleton=True requires a distance matrix; "
                "rc._dist_matrix_ is None."
            )
        else:
            note = _add_skeleton(
                fig,
                go,
                X,
                np.asarray(rc._dist_matrix_),
                float(cycles[0].birth),
                is_3d,
                int(skeleton_max_edges),
            )
            if note:
                notes.append(note)

    fig.add_trace(
        _scatter(
            go,
            is_3d,
            X[:, 0],
            X[:, 1],
            X[:, 2] if is_3d else None,
            mode="markers",
            marker=dict(
                size=2.5 if is_3d else 5, color=_CLOUD_COLOR, opacity=0.5
            ),
            name="Point cloud",
        )
    )

    for i, feature in enumerate(cycles):
        _add_cycle_traces(fig, go, X, feature, colors[i], i, is_3d)

    fig.update_layout(
        title=_compose_title(title, notes),
        scene=dict(aspectmode="data") if is_3d else {},
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
        template="plotly_white",
    )

    if save_html:
        fig.write_html(save_html)

    return fig


def _compose_title(title: str, notes: Sequence[str]) -> str:
    if not notes:
        return title
    return f"{title}<br><sub>{'; '.join(notes)}</sub>"


def _add_cycle_traces(
    fig,
    go,
    X: np.ndarray,
    feature: CycleFeature,
    color: str,
    rank: int,
    is_3d: bool,
) -> None:
    """Draw one feature: its ordinary edges, its birth edge, its vertices."""
    group = f"cycle{rank}"
    label = _legend_label(feature)
    hover = _hover_text(feature)
    n_axes = 3 if is_3d else 2

    edges = np.asarray(feature.cycle_edges, dtype=np.intp).reshape(-1, 2)
    if len(edges):
        u, v = feature.birth_edge
        birth_key = (min(u, v), max(u, v))
        keys = np.column_stack((edges.min(axis=1), edges.max(axis=1)))
        is_birth = (keys[:, 0] == birth_key[0]) & (keys[:, 1] == birth_key[1])

        # The birth edge is drawn black and dashed in every view, which is
        # why it is split out into its own trace rather than coloured.
        for subset, birth in ((~is_birth, False), (is_birth, True)):
            if not subset.any():
                continue
            x, y, z = _edge_line_coords(X, edges[subset], n_axes)
            fig.add_trace(
                _scatter(
                    go,
                    is_3d,
                    x,
                    y,
                    z,
                    mode="lines",
                    line=dict(
                        color="black" if birth else color,
                        width=4 if birth else 3,
                        dash="dash" if birth else "solid",
                    ),
                    name="birth edge" if birth else label,
                    legendgroup=group,
                    showlegend=birth,
                    hovertext=hover,
                    hoverinfo="text",
                )
            )

    cv = np.asarray(feature.cycle_vertices, dtype=np.intp)
    fig.add_trace(
        _scatter(
            go,
            is_3d,
            X[cv, 0],
            X[cv, 1],
            X[cv, 2] if is_3d else None,
            mode="markers",
            marker=dict(
                size=5 if is_3d else 9,
                color=color,
                **({} if is_3d else {"symbol": "circle"}),
            ),
            name=label,
            legendgroup=group,
            showlegend=True,
            hovertext=hover,
            hoverinfo="text",
        )
    )
