"""Colour assignment for representative cycles.

One cycle, one colour, in every view.  This module is the only place a cycle
colour is chosen; the persistence diagram, barcode, matplotlib panels and
plotly traces all read from :func:`cycle_colors`.  That is what lets a reader
carry "the orange loop" from the barcode to the point cloud without
re-deriving which bar is which.

The canonical representation is a **hex string**.  matplotlib artists store
RGBA float tuples and plotly stores CSS colour strings, so the two are never
directly comparable; fixing the source representation as hex means every
renderer converts *from* one definition, and tests can compare colours after
normalising through ``matplotlib.colors.to_hex``.
"""

from __future__ import annotations

from typing import Sequence

# Okabe-Ito, the standard colourblind-safe qualitative palette
# (Okabe & Ito 2008, "Color Universal Design").  Distinguishable under
# deuteranopia, protanopia and tritanopia.
#
# Black is excluded because the birth edge is drawn in black in every view,
# and Okabe-Ito yellow (#F0E442) is excluded because it is illegible on the
# white background these figures use.  That leaves six.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
)


def cycle_colors(n_or_features) -> list[str]:
    """Return one hex colour per cycle, in feature order.

    Parameters
    ----------
    n_or_features : int | Sequence
        Either a count, or any sequence of features (only its length is used).

    Returns
    -------
    list[str]
        Hex colour strings, ``len == n``.  Deterministic: the same input
        always yields the same list, so colours are stable across the
        several figures produced from one fit.

    Notes
    -----
    Colours are taken **by index** from a discrete palette.  This is
    deliberate: sampling a qualitative colormap continuously — as
    ``cm.tab10(np.linspace(0, 1, k))`` does — interpolates *between* the
    designed categorical colours and produces muddy, non-colourblind-safe
    results that also change every time the number of cycles changes.

    With more cycles than palette entries the palette repeats.  Repetition is
    preferable to interpolation: a reader can tell two same-coloured loops
    apart from their position in the legend, but cannot recover a designed
    colour from an interpolated one.
    """
    n = n_or_features if isinstance(n_or_features, int) else len(n_or_features)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]
