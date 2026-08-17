"""H₁ barcode view (spec 001, V6).

A barcode is the honest view of a filtration: one horizontal bar per H₁
class, from birth to death.  Two properties matter here and are the reason
this module was rewritten.

**Essential classes are drawn, not dropped.**  A class that never dies within
the filtration has ``death = inf``.  The previous implementation filtered
those rows out, which silently deleted exactly the bars that dominate a
truncated filtration — the 60-point circle fitted with ``max_edge_length=1.0``
lost its only loop.  Here they are drawn as right-pointing arrows running to
the right edge of the axes: the arrowhead says "this bar continues past the
end of the filtration", which a bar clipped at the axis edge would not.

**Colour carries identity, not magnitude.**  Bars that became reconstructed
cycles take their colour from :func:`repcycles.palette.cycle_colors`, so the
orange bar here is the orange loop in every other view of the same fit.  Bars
with no corresponding feature are drawn in a neutral grey.  The old
persistence-driven colormap (``cm.plasma(pers / pers.max())``) broke that
identity and divided by ``inf`` or by ``0`` whenever an essential or a
zero-persistence bar was present.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from ..palette import cycle_colors

#: Colour for bars that have no corresponding entry in ``features_``.
NEUTRAL_COLOR = "#B0B0B0"

_LINEWIDTH_PLAIN = 1.8
_LINEWIDTH_FEATURE = 3.0
_MARKER_SIZE = 6.0
_ARROW_HEAD_SCALE = 14.0

#: Floor on the x-axis padding, so a degenerate diagram (one bar of zero
#: persistence) still produces finite, non-empty axis limits.
_MIN_PAD = 0.01


def plot_barcode(
    diagram: np.ndarray,
    features: Optional[Sequence] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw the H₁ barcode, essential bars included.

    Parameters
    ----------
    diagram : np.ndarray, shape (n, 2)
        The H₁ persistence diagram — ``[birth, death]`` per row.  ``death``
        may be ``np.inf`` (an essential class).  Births must be finite.
    features : sequence of CycleFeature, optional
        The subset of classes that became reconstructed cycles.  Each
        ``feature.index`` is a **row index into** ``diagram``; those rows are
        drawn in the shared cycle colours and marked, so a reader can tell
        which bars became drawn loops.  Everything else is neutral grey.
    colors : sequence of str, optional
        One hex colour per feature, in feature order.  Defaults to
        :func:`repcycles.palette.cycle_colors`.  Pass the *same* list used by
        the other views to keep one cycle one colour.
    figsize : tuple, default (8, 4)
    save_path : str, optional
        If given, the figure is written here at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.  This function never calls ``plt.show()``.

    Raises
    ------
    ValueError
        If the diagram is empty or malformed, if a feature's ``index`` is not
        a row of *diagram*, if two features claim the same row, or if
        ``colors`` does not match ``features`` in length.

    Notes
    -----
    Bars are sorted top-to-bottom by: essential first (infinite persistence
    outranks every finite value), then persistence descending, then birth
    ascending, then diagram row ascending.  The last two keys exist so the
    order is total — several essential bars all have ``persistence = inf``
    and would otherwise be in arbitrary relative order.

    A zero-persistence bar (birth == death) is drawn at its true length,
    which is zero; the round line cap renders it as a dot.  It is not widened
    to a visible minimum, because that would draw a bar the data does not
    support.
    """
    dgm = _validate_diagram(diagram)
    features = list(features) if features is not None else []
    colors = _resolve_colors(features, colors)
    marked = _map_features_to_rows(dgm, features, colors)

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    is_essential = ~np.isfinite(deaths)

    order = _bar_order(births, deaths, is_essential)
    x_lo, x_hi, arrow_tip = _x_extent(births, deaths, is_essential)

    fig, ax = plt.subplots(figsize=figsize)
    n_bars = len(dgm)
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for rank, row in enumerate(order):
        y = n_bars - 1 - rank  # rank 0 sits at the top
        entry = marked.get(row)
        color = NEUTRAL_COLOR if entry is None else entry[1]
        linewidth = _LINEWIDTH_PLAIN if entry is None else _LINEWIDTH_FEATURE

        if is_essential[row]:
            ax.add_patch(
                FancyArrowPatch(
                    (births[row], y),
                    (arrow_tip, y),
                    arrowstyle="-|>",
                    mutation_scale=_ARROW_HEAD_SCALE,
                    linewidth=linewidth,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=3,
                    gid=f"bar-infinite-{row}",
                )
            )
        else:
            ax.plot(
                [births[row], deaths[row]],
                [y, y],
                lw=linewidth,
                color=color,
                solid_capstyle="round",
                zorder=2,
                gid=f"bar-finite-{row}",
            )

        if entry is not None:
            feature_rank = entry[0]
            ax.plot(
                [births[row]],
                [y],
                marker="o",
                markersize=_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=4,
                gid=f"feature-marker-{row}",
            )
            tick_positions.append(y)
            tick_labels.append(f"#{feature_rank}")

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-1, max(n_bars, 1))
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Filtration value")
    ax.set_title("H₁ Barcode  (sorted by persistence, essential first)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    _add_legend(ax, has_features=bool(marked), has_essential=bool(is_essential.any()))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Validation — this is a public entry point, so its input is untrusted.
# ----------------------------------------------------------------------


def _validate_diagram(diagram) -> np.ndarray:
    """Return the diagram as a float64 ``(n, 2)`` array, or raise.

    An *empty* diagram is an error: there is nothing to draw and silently
    returning a blank figure would let a caller publish it.  A diagram whose
    bars are *all* essential is not an error — that is the normal shape of a
    truncated filtration, and it is precisely the case the old
    ``isfinite`` filter turned into an empty plot.
    """
    dgm = np.asarray(diagram, dtype=float)
    if dgm.size == 0:
        raise ValueError("No H1 features found.")
    if dgm.ndim != 2 or dgm.shape[1] != 2:
        raise ValueError(
            f"diagram must have shape (n, 2) of [birth, death] rows, "
            f"got shape {dgm.shape}."
        )
    nan_rows = np.flatnonzero(np.isnan(dgm).any(axis=1))
    if nan_rows.size:
        raise ValueError(
            f"diagram contains NaN at row {int(nan_rows[0])}; NaN has no "
            "topological meaning and cannot be plotted."
        )
    bad_birth = np.flatnonzero(~np.isfinite(dgm[:, 0]))
    if bad_birth.size:
        raise ValueError(
            f"birth must be finite; row {int(bad_birth[0])} has "
            f"birth={dgm[bad_birth[0], 0]}."
        )
    inverted = np.flatnonzero(dgm[:, 1] < dgm[:, 0])
    if inverted.size:
        r = int(inverted[0])
        raise ValueError(
            f"death must not precede birth; row {r} is "
            f"[{dgm[r, 0]}, {dgm[r, 1]}]."
        )
    return dgm


def _resolve_colors(features: Sequence, colors: Optional[Sequence[str]]) -> list:
    if colors is None:
        return cycle_colors(len(features))
    colors = list(colors)
    if len(colors) != len(features):
        raise ValueError(
            f"colors has {len(colors)} entries but there are "
            f"{len(features)} features; they must correspond one-to-one."
        )
    return colors


def _map_features_to_rows(
    dgm: np.ndarray, features: Sequence, colors: Sequence[str]
) -> Dict[int, Tuple[int, str]]:
    """Map diagram row -> (feature rank, colour).

    ``feature.index`` is a row index into *dgm*.  Both failure modes raise
    rather than degrade: an out-of-range index would mark an arbitrary bar,
    and two features claiming one row would hide one of them — the same class
    of silent mismarking that motivated this spec.
    """
    mapping: Dict[int, Tuple[int, str]] = {}
    for rank, (feature, color) in enumerate(zip(features, colors)):
        row = int(feature.index)
        if not 0 <= row < len(dgm):
            raise ValueError(
                f"features[{rank}].index = {row} is not a row of a diagram "
                f"with {len(dgm)} rows."
            )
        if row in mapping:
            raise ValueError(
                f"features[{rank}] and features[{mapping[row][0]}] both claim "
                f"diagram row {row}; each feature must own a distinct row."
            )
        mapping[row] = (rank, color)
    return mapping


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _bar_order(
    births: np.ndarray, deaths: np.ndarray, is_essential: np.ndarray
) -> list:
    """Row indices, top bar first.  Total order (see the docstring's Notes)."""
    persistence = deaths - births

    def key(row: int):
        if is_essential[row]:
            return (0, 0.0, float(births[row]), row)
        return (1, -float(persistence[row]), float(births[row]), row)

    return sorted(range(len(births)), key=key)


def _x_extent(
    births: np.ndarray, deaths: np.ndarray, is_essential: np.ndarray
) -> Tuple[float, float, float]:
    """Return ``(x_lo, x_hi, arrow_tip)``, all finite.

    ``inf`` never reaches an axis limit: the right edge is derived from the
    finite values only, and essential arrows are drawn to a tip just inside
    it.  The ``_MIN_PAD`` floor keeps the limits distinct when every finite
    value coincides (a single zero-persistence bar), which is where a
    span-proportional pad would collapse to zero.
    """
    finite_values = np.concatenate([births, deaths[~is_essential]])
    lo = float(min(finite_values.min(), 0.0))
    hi = float(finite_values.max())
    pad = max((hi - lo) * 0.05, _MIN_PAD)

    if is_essential.any():
        # Extra room on the right so the arrowheads are not clipped by the
        # spine while still visibly running to the edge of the plot.
        x_hi = hi + 4.0 * pad
        arrow_tip = hi + 3.0 * pad
    else:
        x_hi = hi + pad
        arrow_tip = x_hi
    return lo - pad, x_hi, arrow_tip


def _add_legend(ax, has_features: bool, has_essential: bool) -> None:
    """Explain the two encodings a reader cannot guess: the arrowhead and
    the neutral grey.  Proxy artists are not added to the axes, so they do
    not disturb artist counts."""
    handles = []
    if has_features:
        handles.append(
            Line2D(
                [], [], color="black", lw=_LINEWIDTH_FEATURE, marker="o",
                markersize=_MARKER_SIZE, markerfacecolor="white",
                label="drawn cycle",
            )
        )
        handles.append(
            Line2D([], [], color=NEUTRAL_COLOR, lw=_LINEWIDTH_PLAIN,
                   label="not drawn")
        )
    if has_essential:
        handles.append(
            Line2D([], [], color="black", lw=_LINEWIDTH_PLAIN, marker=">",
                   markevery=[-1], label="essential (never dies)")
        )
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
