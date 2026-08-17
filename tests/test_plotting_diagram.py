"""Tests for the persistence-diagram panel (spec 001 T10 — V1, V2, V3, V10, V11).

Structural assertions only: artist counts, offsets, colours and axis limits.
No image comparison anywhere (constitution, Testing Bar).

The central regression is AC11.  ``CycleFeature.index`` is a row index into
the *full* H₁ diagram; the previous implementation used it against arrays
that had already been filtered down to the finite-death rows.  With an
essential bar present that either rings the wrong point or raises
``IndexError`` — and *where* the infinite row sits in the array decides
which.  Every highlight test therefore runs with the infinite row first and
again last.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from repcycles.feature import CycleFeature  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.diagram import (  # noqa: E402
    GID_DIAGONAL,
    GID_EMPTY,
    GID_ESSENTIAL,
    GID_FINITE,
    GID_HIGHLIGHT,
    GID_INF_BAND,
    GID_INF_LABEL,
    draw_persistence_diagram,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

@pytest.fixture
def ax():
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


def collection(ax, gid):
    """The one collection carrying *gid*, or ``None``.

    Searches ``ax.collections`` rather than ``findobj`` so that the legend's
    proxy artists can never be mistaken for the real ones.
    """
    found = [c for c in ax.collections if c.get_gid() == gid]
    assert len(found) <= 1, f"expected at most one {gid}, got {len(found)}"
    return found[0] if found else None


def line(ax, gid):
    found = [ln for ln in ax.lines if ln.get_gid() == gid]
    return found[0] if found else None


def text(ax, gid):
    found = [t for t in ax.texts if t.get_gid() == gid]
    return found[0] if found else None


def offsets(coll):
    return np.asarray(coll.get_offsets(), dtype=float)


def edge_hexes(coll):
    return [to_hex(rgba) for rgba in coll.get_edgecolors()]


def make_features(*indices, diagram=None):
    """Build features directly, so failure cases are constructible."""
    out = []
    for i in indices:
        birth, death = (0.0, 1.0) if diagram is None else diagram[i]
        out.append(
            CycleFeature(
                index=int(i),
                birth=float(birth),
                death=float(death),
                persistence=float(death - birth),
                is_essential=bool(not np.isfinite(death)),
            )
        )
    return out


# Infinite row FIRST and LAST: array position is exactly what broke.
INF_FIRST = np.array([[0.50, np.inf], [0.10, 0.90], [0.30, 1.40]])
INF_LAST = np.array([[0.10, 0.90], [0.30, 1.40], [0.50, np.inf]])
MIXED = {"inf_first": INF_FIRST, "inf_last": INF_LAST}


# --------------------------------------------------------------------------
# AC11 — highlights are correct with essential bars present
# --------------------------------------------------------------------------

class TestHighlightWithInfiniteBars:

    @pytest.mark.parametrize("name", sorted(MIXED))
    def test_essential_row_is_highlighted_without_raising(self, ax, name):
        dgm = MIXED[name]
        essential_row = int(np.flatnonzero(~np.isfinite(dgm[:, 1]))[0])

        draw_persistence_diagram(ax, dgm, make_features(essential_row, diagram=dgm))

        highlight = collection(ax, GID_HIGHLIGHT)
        assert highlight is not None
        pts = offsets(highlight)
        assert pts.shape == (1, 2)
        # x is the essential row's birth, not some other row's.
        assert pts[0, 0] == pytest.approx(dgm[essential_row, 0])
        # y is the ∞ band, above every finite death, and finite.
        band = line(ax, GID_INF_BAND)
        assert pts[0, 1] == pytest.approx(band.get_ydata()[0])
        assert np.isfinite(pts[0, 1])
        assert pts[0, 1] > np.nanmax(dgm[np.isfinite(dgm[:, 1]), 1])

    @pytest.mark.parametrize("name", sorted(MIXED))
    def test_finite_row_after_an_infinite_row_lands_on_its_own_point(
        self, ax, name
    ):
        """The exact old bug: full-diagram index applied to a filtered array.

        With ``INF_FIRST`` the old code indexed a 2-element array with row 2
        (``IndexError``); with row 1 it silently ringed row 0's point.
        """
        dgm = MIXED[name]
        finite_rows = [int(i) for i in np.flatnonzero(np.isfinite(dgm[:, 1]))]
        target = finite_rows[-1]

        draw_persistence_diagram(ax, dgm, make_features(target, diagram=dgm))

        pts = offsets(collection(ax, GID_HIGHLIGHT))
        assert pts[0] == pytest.approx(dgm[target])

    @pytest.mark.parametrize("name", sorted(MIXED))
    def test_every_row_highlighted_matches_its_own_diagram_row(self, ax, name):
        dgm = MIXED[name]
        rows = list(range(len(dgm)))

        draw_persistence_diagram(ax, dgm, make_features(*rows, diagram=dgm))

        pts = offsets(collection(ax, GID_HIGHLIGHT))
        band_y = line(ax, GID_INF_BAND).get_ydata()[0]
        expected = np.column_stack(
            [dgm[:, 0], np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], band_y)]
        )
        assert pts == pytest.approx(expected)

    def test_highlight_count_equals_feature_count(self, ax):
        draw_persistence_diagram(ax, INF_FIRST, make_features(0, 2, diagram=INF_FIRST))
        assert offsets(collection(ax, GID_HIGHLIGHT)).shape[0] == 2


# --------------------------------------------------------------------------
# All-finite diagrams keep working
# --------------------------------------------------------------------------

class TestAllFiniteDiagram:

    DGM = np.array([[0.1, 0.9], [0.3, 1.4], [0.2, 0.4]])

    def test_all_points_are_scattered(self, ax):
        draw_persistence_diagram(ax, self.DGM, [])
        assert offsets(collection(ax, GID_FINITE)) == pytest.approx(self.DGM)

    def test_no_infinity_band_when_nothing_is_essential(self, ax):
        draw_persistence_diagram(ax, self.DGM, make_features(1, diagram=self.DGM))
        assert line(ax, GID_INF_BAND) is None
        assert text(ax, GID_INF_LABEL) is None
        assert collection(ax, GID_ESSENTIAL) is None

    def test_highlights_match_the_selected_rows(self, ax):
        draw_persistence_diagram(ax, self.DGM, make_features(2, 0, diagram=self.DGM))
        pts = offsets(collection(ax, GID_HIGHLIGHT))
        assert pts == pytest.approx(self.DGM[[2, 0]])

    def test_no_highlight_artist_when_no_features(self, ax):
        draw_persistence_diagram(ax, self.DGM, [])
        assert collection(ax, GID_HIGHLIGHT) is None

    def test_diagonal_is_drawn(self, ax):
        draw_persistence_diagram(ax, self.DGM, [])
        diag = line(ax, GID_DIAGONAL)
        assert diag is not None
        assert diag.get_xdata() == pytest.approx(diag.get_ydata())


# --------------------------------------------------------------------------
# Essential bars are drawn, not dropped
# --------------------------------------------------------------------------

class TestInfiniteBand:

    @pytest.mark.parametrize("name", sorted(MIXED))
    def test_essential_points_sit_on_the_band(self, ax, name):
        dgm = MIXED[name]
        draw_persistence_diagram(ax, dgm, [])

        essential = collection(ax, GID_ESSENTIAL)
        assert essential is not None
        pts = offsets(essential)
        band_y = line(ax, GID_INF_BAND).get_ydata()[0]
        expected_births = dgm[~np.isfinite(dgm[:, 1]), 0]
        assert pts[:, 0] == pytest.approx(expected_births)
        assert pts[:, 1] == pytest.approx(np.full(len(pts), band_y))

    def test_band_is_labelled_infinity(self, ax):
        draw_persistence_diagram(ax, INF_FIRST, [])
        label = text(ax, GID_INF_LABEL)
        assert label is not None
        assert label.get_text() == "∞"

    def test_finite_scatter_excludes_essential_rows(self, ax):
        draw_persistence_diagram(ax, INF_LAST, [])
        pts = offsets(collection(ax, GID_FINITE))
        assert pts.shape == (2, 2)
        assert np.isfinite(pts).all()

    def test_all_essential_diagram_renders_without_finite_points(self, ax):
        dgm = np.array([[0.2, np.inf], [0.6, np.inf]])
        draw_persistence_diagram(ax, dgm, make_features(0, 1, diagram=dgm))

        assert collection(ax, GID_FINITE) is None
        assert offsets(collection(ax, GID_ESSENTIAL)).shape == (2, 2)
        assert offsets(collection(ax, GID_HIGHLIGHT)).shape == (2, 2)
        assert np.isfinite(ax.get_xlim()).all()
        assert np.isfinite(ax.get_ylim()).all()

    def test_band_lies_inside_the_y_limits(self, ax):
        draw_persistence_diagram(ax, INF_FIRST, [])
        band_y = line(ax, GID_INF_BAND).get_ydata()[0]
        bottom, top = ax.get_ylim()
        assert bottom < band_y < top


# --------------------------------------------------------------------------
# Axis limits stay finite
# --------------------------------------------------------------------------

class TestAxisLimits:

    @pytest.mark.parametrize(
        "dgm",
        [
            INF_FIRST,
            INF_LAST,
            np.array([[0.1, 0.9], [0.3, 1.4]]),
            np.array([[0.4, np.inf]]),
            np.array([[0.5, 0.5]]),  # zero-span diagram
        ],
        ids=["inf_first", "inf_last", "all_finite", "single_inf", "zero_span"],
    )
    def test_limits_are_finite_and_ordered(self, ax, dgm):
        draw_persistence_diagram(ax, dgm, [])
        for lo, hi in (ax.get_xlim(), ax.get_ylim()):
            assert np.isfinite(lo) and np.isfinite(hi)
            assert hi > lo

    def test_x_limits_ignore_the_infinity_band(self, ax):
        """The band raises the y ceiling only; x still tracks the births."""
        draw_persistence_diagram(ax, INF_FIRST, [])
        finite_max = float(
            np.concatenate(
                [INF_FIRST[:, 0], INF_FIRST[np.isfinite(INF_FIRST[:, 1]), 1]]
            ).max()
        )
        assert ax.get_xlim()[1] < line(ax, GID_INF_BAND).get_ydata()[0]
        assert ax.get_xlim()[1] == pytest.approx(finite_max + 0.05 * (finite_max - 0.1))


# --------------------------------------------------------------------------
# Colours (V2, V3)
# --------------------------------------------------------------------------

class TestColours:

    def test_default_colours_come_from_cycle_colors(self, ax):
        features = make_features(0, 1, 2, diagram=INF_FIRST)
        draw_persistence_diagram(ax, INF_FIRST, features)

        drawn = edge_hexes(collection(ax, GID_HIGHLIGHT))
        assert drawn == [to_hex(c) for c in cycle_colors(features)]

    def test_explicit_colours_are_honoured(self, ax):
        custom = ["#123456", "#abcdef"]
        draw_persistence_diagram(
            ax, INF_LAST, make_features(0, 2, diagram=INF_LAST), colors=custom
        )
        assert edge_hexes(collection(ax, GID_HIGHLIGHT)) == [
            to_hex(c) for c in custom
        ]

    def test_colour_of_feature_k_is_independent_of_the_rows_it_points_at(self, ax):
        """V2: cycle *k*'s colour is fixed by its rank, not by its diagram row."""
        a = make_features(0, 1, diagram=INF_FIRST)
        b = make_features(2, 1, diagram=INF_FIRST)

        draw_persistence_diagram(ax, INF_FIRST, a)
        first = edge_hexes(collection(ax, GID_HIGHLIGHT))
        ax.clear()
        draw_persistence_diagram(ax, INF_FIRST, b)
        assert edge_hexes(collection(ax, GID_HIGHLIGHT)) == first

    def test_highlights_are_unfilled_rings(self, ax):
        """Rings must not hide the underlying persistence-coloured point."""
        draw_persistence_diagram(ax, INF_FIRST, make_features(1, diagram=INF_FIRST))
        faces = collection(ax, GID_HIGHLIGHT).get_facecolors()
        assert faces.size == 0 or np.allclose(faces[:, 3], 0.0)

    def test_too_few_colours_is_an_error(self, ax):
        with pytest.raises(ValueError, match="one colour per feature"):
            draw_persistence_diagram(
                ax, INF_FIRST, make_features(0, 1, diagram=INF_FIRST), colors=["#123456"]
            )


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

class TestDegenerateInput:

    @pytest.mark.parametrize(
        "empty",
        [None, np.empty((0, 2)), np.array([])],
        ids=["none", "empty_2d", "empty_1d"],
    )
    def test_empty_diagram_renders_an_empty_state(self, ax, empty):
        draw_persistence_diagram(ax, empty, [])

        assert text(ax, GID_EMPTY) is not None
        assert collection(ax, GID_FINITE) is None
        assert np.isfinite(ax.get_xlim()).all()
        assert np.isfinite(ax.get_ylim()).all()

    def test_single_finite_feature(self, ax):
        dgm = np.array([[0.25, 0.75]])
        draw_persistence_diagram(ax, dgm, make_features(0, diagram=dgm))

        assert offsets(collection(ax, GID_HIGHLIGHT)) == pytest.approx(dgm)
        assert edge_hexes(collection(ax, GID_HIGHLIGHT)) == [
            to_hex(cycle_colors(1)[0])
        ]

    def test_single_essential_feature(self, ax):
        dgm = np.array([[0.25, np.inf]])
        draw_persistence_diagram(ax, dgm, make_features(0, diagram=dgm))

        pts = offsets(collection(ax, GID_HIGHLIGHT))
        assert pts[0, 0] == pytest.approx(0.25)
        assert np.isfinite(pts[0, 1])

    def test_features_none_is_treated_as_none_shown(self, ax):
        draw_persistence_diagram(ax, INF_FIRST, None)
        assert collection(ax, GID_HIGHLIGHT) is None

    def test_titles_and_labels_are_set_even_when_empty(self, ax):
        draw_persistence_diagram(ax, None, [])
        assert "H₁" in ax.get_title()
        assert ax.get_xlabel() == "Birth"
        assert ax.get_ylabel() == "Death"


class TestInputValidation:

    def test_wrong_shape_is_rejected(self, ax):
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            draw_persistence_diagram(ax, np.zeros((3, 3)), [])

    def test_nan_is_rejected_naming_the_row(self, ax):
        dgm = np.array([[0.1, 0.9], [0.2, np.nan]])
        with pytest.raises(ValueError, match="NaN at row 1"):
            draw_persistence_diagram(ax, dgm, [])

    def test_infinite_birth_is_rejected(self, ax):
        dgm = np.array([[np.inf, 0.9]])
        with pytest.raises(ValueError, match="non-finite birth"):
            draw_persistence_diagram(ax, dgm, [])

    def test_negative_infinite_death_is_rejected(self, ax):
        dgm = np.array([[0.1, -np.inf]])
        with pytest.raises(ValueError, match="death -inf"):
            draw_persistence_diagram(ax, dgm, [])

    @pytest.mark.parametrize("bad_index", [3, -1, 99])
    def test_out_of_range_feature_index_is_named(self, ax, bad_index):
        feature = CycleFeature(index=bad_index, birth=0.1, death=0.9, persistence=0.8)
        with pytest.raises(ValueError, match="out of range"):
            draw_persistence_diagram(ax, INF_FIRST, [feature])


# --------------------------------------------------------------------------
# V10 / V11 hygiene
# --------------------------------------------------------------------------

class TestPlottingHygiene:

    def test_no_deprecation_warnings_on_a_full_draw(self, ax):
        """V10: clean under -W error::DeprecationWarning.

        MatplotlibDeprecationWarning subclasses DeprecationWarning, so the
        single filter covers both.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            draw_persistence_diagram(
                ax, INF_FIRST, make_features(0, 1, 2, diagram=INF_FIRST)
            )
            ax.figure.canvas.draw()

    def test_never_calls_plt_show(self, ax, monkeypatch):
        """V11."""
        def boom(*args, **kwargs):
            raise AssertionError("draw_persistence_diagram called plt.show()")

        monkeypatch.setattr(plt, "show", boom)
        draw_persistence_diagram(ax, INF_FIRST, make_features(0, diagram=INF_FIRST))

    def test_returns_none_and_draws_in_place(self, ax):
        assert draw_persistence_diagram(ax, INF_FIRST, []) is None
        assert len(ax.collections) > 0

    def test_module_does_not_import_pyplot(self):
        """Panels own the figure lifecycle; this helper only draws on an Axes."""
        import repcycles.plotting.diagram as mod

        assert not hasattr(mod, "plt")
