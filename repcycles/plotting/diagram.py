"""Persistence-diagram panel (spec 001, V1/V2/V3).

The panel answers one question honestly: *which* H₁ classes did the library
turn into the cycles you are looking at?  Two properties make that answer
trustworthy.

**Highlights index the full diagram.**  A ``CycleFeature.index`` is a row
index into the H₁ diagram as ripser reported it.  The previous
implementation filtered the diagram down to its finite rows and then indexed
that shorter array with those full-diagram indices, so the moment an
essential (infinite-death) bar appeared the ring landed on the wrong point —
or raised ``IndexError``.  Nothing here indexes a filtered array; positions
are read straight from ``diagram[index]`` and only the *y* coordinate is
remapped, for essential rows, onto the ∞ band.

**Essential bars are drawn, not dropped.**  They sit on a dashed, labelled
band above the finite range.  ``inf`` never reaches an axis-limit
computation: limits come from the finite rows plus a fixed padding, and the
band is placed inside them.

Artists carry stable ``gid`` values (the ``GID_*`` constants) so tests and
downstream figures can find them without depending on child-artist ordering.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from ..feature import CycleFeature
from ..palette import cycle_colors

# Stable artist identifiers.  Structural tests and composite figures look
# artists up by these rather than by index into ``ax.collections``.
GID_DIAGONAL = "pd-diagonal"
GID_FINITE = "pd-finite"
GID_ESSENTIAL = "pd-essential"
GID_INF_BAND = "pd-inf-band"
GID_INF_LABEL = "pd-inf-label"
GID_HIGHLIGHT = "pd-highlight"
GID_EMPTY = "pd-empty"

_HIGHLIGHT_SIZE = 150.0
_POINT_SIZE = 55.0


def draw_persistence_diagram(
    ax: Axes,
    diagram: Optional[np.ndarray],
    features: Optional[Sequence[CycleFeature]] = None,
    colors: Optional[Sequence[str]] = None,
) -> None:
    """Draw an H₁ persistence diagram onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.  Drawn onto in place; nothing is shown or saved.
    diagram : np.ndarray, shape (n, 2), or None
        The H₁ persistence diagram, ``[birth, death]`` per row.  Deaths may
        be ``np.inf`` (essential classes).  ``None`` or an empty array
        renders an empty-state panel rather than raising.
    features : sequence of CycleFeature, optional
        The subset of classes that were reconstructed and shown elsewhere in
        the figure.  Each ``feature.index`` is a **row index into**
        ``diagram``; the corresponding point is ringed in that feature's
        colour.
    colors : sequence of str, optional
        Hex colours, one per feature, in feature order.  Defaults to
        :func:`repcycles.palette.cycle_colors`, which is what every other
        view uses — so cycle *k* is ringed here in the same colour it is
        drawn in on the point cloud, the barcode and the plotly figure.

    Raises
    ------
    ValueError
        If *diagram* is not an ``(n, 2)`` array, contains ``NaN``, has a
        non-finite birth or a ``-inf`` death, if a feature index is out of
        range for *diagram*, or if fewer *colors* than *features* are given.

    Notes
    -----
    Axis limits are computed from finite values only, so a diagram made
    entirely of essential classes still produces finite limits.
    """
    ax.set_title("Persistence Diagram (H₁)", fontsize=11)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    dgm = _validate_diagram(diagram)
    if dgm is None or len(dgm) == 0:
        _draw_empty_state(ax)
        return

    feature_list: List[CycleFeature] = list(features) if features else []
    highlight_colors = _resolve_colors(feature_list, colors)
    highlight_rows = _validate_feature_indices(feature_list, len(dgm))

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    is_finite = np.isfinite(deaths)

    lo, hi, pad = _finite_extent(births, deaths[is_finite])
    # Essential rows live on a band above every finite point but still
    # inside the axes, so `inf` never reaches set_ylim.
    inf_y = hi + 3.0 * pad
    has_essential = bool((~is_finite).any())
    y_top = (inf_y + pad) if has_essential else (hi + pad)

    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "k--",
        lw=0.8,
        alpha=0.5,
        zorder=1,
        gid=GID_DIAGONAL,
    )

    if is_finite.any():
        persistence = deaths[is_finite] - births[is_finite]
        scatter = ax.scatter(
            births[is_finite],
            deaths[is_finite],
            c=persistence,
            cmap="plasma",
            s=_POINT_SIZE,
            zorder=3,
            edgecolors="k",
            linewidths=0.4,
            gid=GID_FINITE,
        )
        ax.figure.colorbar(scatter, ax=ax, label="Persistence", shrink=0.85)

    if has_essential:
        _draw_infinite_band(ax, births[~is_finite], inf_y)

    if highlight_rows.size:
        # The fix: read straight from the full diagram, and lift only the
        # essential rows onto the band.  No filtered-array indexing anywhere.
        highlight_y = np.where(
            is_finite[highlight_rows], deaths[highlight_rows], inf_y
        )
        ax.scatter(
            births[highlight_rows],
            highlight_y,
            s=_HIGHLIGHT_SIZE,
            facecolors="none",
            edgecolors=list(highlight_colors),
            linewidths=1.8,
            zorder=5,
            label="shown cycles",
            gid=GID_HIGHLIGHT,
        )

    if has_essential or highlight_rows.size:
        ax.legend(fontsize=8, loc="lower right")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, y_top)
    ax.set_aspect("equal")


def _validate_diagram(diagram: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the diagram as float64, or ``None`` when there is nothing to draw.

    External numeric input is untrusted: a NaN birth or death would poison
    every comparison downstream and silently produce a blank panel, so it is
    rejected here with the offending row named.
    """
    if diagram is None:
        return None

    dgm = np.asarray(diagram, dtype=float)
    if dgm.size == 0:
        return None
    if dgm.ndim != 2 or dgm.shape[1] != 2:
        raise ValueError(
            f"diagram must have shape (n, 2), got {dgm.shape}."
        )

    nan_rows = np.nonzero(np.isnan(dgm).any(axis=1))[0]
    if nan_rows.size:
        raise ValueError(
            f"diagram contains NaN at row {int(nan_rows[0])}; "
            "NaN has no topological meaning."
        )
    inf_births = np.nonzero(~np.isfinite(dgm[:, 0]))[0]
    if inf_births.size:
        raise ValueError(
            f"diagram row {int(inf_births[0])} has a non-finite birth; "
            "every H₁ class is born at a finite filtration value."
        )
    neg_inf_deaths = np.nonzero(np.isneginf(dgm[:, 1]))[0]
    if neg_inf_deaths.size:
        raise ValueError(
            f"diagram row {int(neg_inf_deaths[0])} has death -inf; "
            "only +inf (essential) is meaningful."
        )
    return dgm


def _resolve_colors(
    features: Sequence[CycleFeature], colors: Optional[Sequence[str]]
) -> List[str]:
    """One hex colour per feature, defaulting to the shared palette (V2)."""
    if colors is None:
        return cycle_colors(len(features))
    colors = list(colors)
    if len(colors) < len(features):
        raise ValueError(
            f"colors has {len(colors)} entries for {len(features)} features; "
            "one colour per feature is required."
        )
    return colors[: len(features)]


def _validate_feature_indices(
    features: Sequence[CycleFeature], n_rows: int
) -> np.ndarray:
    """Return feature row indices, rejecting any that fall outside *diagram*.

    An out-of-range index means the caller paired features to a different
    diagram than the one being drawn.  Failing loudly beats ringing an
    arbitrary point, which is the failure mode this panel exists to remove.
    """
    if not features:
        return np.empty(0, dtype=int)

    rows = np.asarray([int(f.index) for f in features], dtype=int)
    bad = np.nonzero((rows < 0) | (rows >= n_rows))[0]
    if bad.size:
        offender = int(bad[0])
        raise ValueError(
            f"features[{offender}].index = {rows[offender]} is out of range "
            f"for a diagram with {n_rows} rows."
        )
    return rows


def _finite_extent(
    births: np.ndarray, finite_deaths: np.ndarray
) -> tuple[float, float, float]:
    """Data range and padding, computed from finite values only."""
    values = np.concatenate([births, finite_deaths])
    lo = float(values.min())
    hi = float(values.max())
    pad = max((hi - lo) * 0.05, 0.01)
    return lo, hi, pad


def _draw_infinite_band(ax: Axes, births: np.ndarray, inf_y: float) -> None:
    """Draw essential classes on a labelled band above the finite range."""
    ax.axhline(
        inf_y,
        color="0.35",
        ls="--",
        lw=0.9,
        zorder=2,
        gid=GID_INF_BAND,
    )
    ax.annotate(
        "∞",
        xy=(0.99, inf_y),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=13,
        color="0.25",
        gid=GID_INF_LABEL,
    )
    ax.scatter(
        births,
        np.full(births.shape, inf_y),
        marker="^",
        s=_POINT_SIZE + 15.0,
        c="0.35",
        edgecolors="k",
        linewidths=0.4,
        zorder=4,
        label="essential (death = ∞)",
        gid=GID_ESSENTIAL,
    )


def _draw_empty_state(ax: Axes) -> None:
    """Render 'nothing to show' as a statement, not as a blank panel."""
    ax.text(
        0.5,
        0.5,
        "No H₁ features",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="0.4",
        gid=GID_EMPTY,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
