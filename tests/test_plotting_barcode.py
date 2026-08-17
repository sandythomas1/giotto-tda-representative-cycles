"""Tests for the H₁ barcode view (spec 001 T11, requirement V6, AC10).

Artifact-structure tests only, per the constitution's testing bar: artist
counts, artist types, colours and data ranges are asserted directly on the
figure.  No image comparison anywhere in this file.

Diagrams are synthetic and features are constructed directly, so every edge
case — an all-essential diagram, a single bar, a zero-persistence bar — is
reachable without depending on `fit()`, which cannot be coerced into
producing most of them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from repcycles.feature import CycleFeature  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.barcode import NEUTRAL_COLOR, plot_barcode  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test creates figures through pyplot; don't leak them."""
    yield
    plt.close("all")


def artists_by_prefix(ax, prefix: str) -> dict:
    """Return ``{diagram row: artist}`` for artists tagged with *prefix*.

    The barcode tags each artist with a ``gid`` of the form
    ``"<kind>-<diagram row>"``, which is what lets these tests talk about
    "the bar for row 3" instead of guessing at draw order.
    """
    out = {}
    for artist in ax.get_children():
        gid = artist.get_gid() or ""
        if gid.startswith(prefix + "-"):
            out[int(gid.rsplit("-", 1)[1])] = artist
    return out


def finite_bars(ax) -> dict:
    return artists_by_prefix(ax, "bar-finite")


def infinite_bars(ax) -> dict:
    return artists_by_prefix(ax, "bar-infinite")


def feature_markers(ax) -> dict:
    return artists_by_prefix(ax, "feature-marker")


def arrow_vertices(arrow: FancyArrowPatch) -> np.ndarray:
    """Real data-space vertices of an arrow patch.

    ``Path`` stores a placeholder vertex for the ``CLOSEPOLY`` code that is
    ignored when rendering; including it would make the arrow look as though
    it extended to arbitrary coordinates.
    """
    from matplotlib.path import Path

    path = arrow.get_path()
    if path.codes is None:
        return path.vertices
    return path.vertices[path.codes != Path.CLOSEPOLY]


def bar_y(ax, row: int) -> float:
    """The y position of the bar for *row*, whatever kind of artist it is."""
    if row in finite_bars(ax):
        return float(finite_bars(ax)[row].get_ydata()[0])
    return float(arrow_vertices(infinite_bars(ax)[row])[0, 1])


def essential_feature(index: int, birth: float) -> CycleFeature:
    return CycleFeature(
        index=index,
        birth=birth,
        death=np.inf,
        persistence=np.inf,
        birth_edge=(0, 1),
        death_edge=(-1, -1),
        cycle_edges=np.array([[0, 1], [1, 2], [2, 0]]),
        is_essential=True,
        is_verified=True,
    )


def finite_feature(index: int, birth: float, death: float) -> CycleFeature:
    return CycleFeature(
        index=index,
        birth=birth,
        death=death,
        persistence=death - birth,
        birth_edge=(0, 1),
        death_edge=(1, 2),
        cycle_edges=np.array([[0, 1], [1, 2], [2, 0]]),
        is_verified=True,
    )


MIXED_DIAGRAM = np.array(
    [
        [0.10, 0.90],   # row 0: finite, persistence 0.80
        [0.50, np.inf],  # row 1: essential
        [0.20, 0.35],   # row 2: finite, persistence 0.15
    ]
)


# ----------------------------------------------------------------------
# AC10 — an essential bar renders an arrow and does not raise
# ----------------------------------------------------------------------


class TestEssentialBars:

    def test_essential_bar_renders_an_arrow_artist(self):
        """AC10.  The old implementation dropped this bar entirely."""
        fig = plot_barcode(MIXED_DIAGRAM, [essential_feature(1, 0.5)])
        ax = fig.axes[0]

        arrows = infinite_bars(ax)
        assert set(arrows) == {1}
        assert isinstance(arrows[1], FancyArrowPatch)

    def test_every_diagram_row_gets_exactly_one_bar(self):
        fig = plot_barcode(MIXED_DIAGRAM)
        ax = fig.axes[0]

        assert set(finite_bars(ax)) == {0, 2}
        assert set(infinite_bars(ax)) == {1}

    def test_arrow_starts_at_birth_and_runs_to_the_right_edge(self):
        fig = plot_barcode(MIXED_DIAGRAM)
        ax = fig.axes[0]

        verts = arrow_vertices(infinite_bars(ax)[1])
        x_left, x_right = verts[:, 0].min(), verts[:, 0].max()
        _, xlim_hi = ax.get_xlim()

        assert x_left == pytest.approx(0.50, abs=1e-9)
        # Past every finite death, and inside the axes so the head is drawn.
        assert x_right > MIXED_DIAGRAM[np.isfinite(MIXED_DIAGRAM[:, 1]), 1].max()
        assert x_right <= xlim_hi + 1e-9

    def test_arrow_is_a_different_artist_type_from_a_finite_bar(self):
        """Visually distinct is asserted structurally: patch vs line."""
        fig = plot_barcode(MIXED_DIAGRAM)
        ax = fig.axes[0]

        assert isinstance(infinite_bars(ax)[1], FancyArrowPatch)
        assert isinstance(finite_bars(ax)[0], Line2D)

    def test_all_infinite_diagram_renders_and_does_not_raise(self):
        """A truncated filtration whose every class is essential is normal
        input, not an empty diagram."""
        dgm = np.array([[0.2, np.inf], [0.7, np.inf], [0.4, np.inf]])

        fig = plot_barcode(dgm)
        ax = fig.axes[0]

        assert len(infinite_bars(ax)) == 3
        assert finite_bars(ax) == {}
        assert all(np.isfinite(ax.get_xlim()))

    def test_single_infinite_bar_renders(self):
        fig = plot_barcode(np.array([[0.3, np.inf]]))
        ax = fig.axes[0]

        assert len(infinite_bars(ax)) == 1
        assert all(np.isfinite(ax.get_xlim()))


# ----------------------------------------------------------------------
# Ordering
# ----------------------------------------------------------------------


class TestOrdering:

    def test_essential_bars_sort_above_every_finite_bar(self):
        fig = plot_barcode(MIXED_DIAGRAM)
        ax = fig.axes[0]

        assert bar_y(ax, 1) > bar_y(ax, 0) > bar_y(ax, 2)

    def test_finite_bars_sort_by_persistence_descending(self):
        dgm = np.array([[0.0, 0.2], [0.0, 1.0], [0.0, 0.5]])
        fig = plot_barcode(dgm)
        ax = fig.axes[0]

        assert bar_y(ax, 1) > bar_y(ax, 2) > bar_y(ax, 0)

    def test_equal_persistence_breaks_ties_by_birth_then_row(self):
        """Without the tiebreakers, several essential bars — all with
        persistence == inf — would have arbitrary relative order."""
        dgm = np.array([[0.9, np.inf], [0.1, np.inf], [0.1, np.inf]])
        fig = plot_barcode(dgm)
        ax = fig.axes[0]

        # birth 0.1 rows first (rows 1 then 2), birth 0.9 last.
        assert bar_y(ax, 1) > bar_y(ax, 2) > bar_y(ax, 0)

    def test_order_is_deterministic_across_calls(self):
        dgm = np.array([[0.0, 1.0], [0.5, np.inf], [0.0, 1.0], [0.2, 0.3]])
        ys = []
        for _ in range(2):
            ax = plot_barcode(dgm).axes[0]
            ys.append([bar_y(ax, r) for r in range(len(dgm))])
        assert ys[0] == ys[1]


# ----------------------------------------------------------------------
# Colours (V2 / V3) — shared palette, no persistence colormap
# ----------------------------------------------------------------------


class TestColors:

    def test_feature_bars_use_the_shared_cycle_colors(self):
        features = [essential_feature(1, 0.5), finite_feature(0, 0.1, 0.9)]
        expected = cycle_colors(features)

        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]

        assert to_hex(infinite_bars(ax)[1].get_edgecolor()) == to_hex(expected[0])
        assert to_hex(finite_bars(ax)[0].get_color()) == to_hex(expected[1])

    def test_non_feature_bars_are_neutral(self):
        features = [finite_feature(0, 0.1, 0.9)]
        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]

        assert to_hex(finite_bars(ax)[2].get_color()) == to_hex(NEUTRAL_COLOR)
        assert to_hex(infinite_bars(ax)[1].get_edgecolor()) == to_hex(NEUTRAL_COLOR)

    def test_explicit_colors_override_the_default_palette(self):
        """The other views pass their own list so all views agree."""
        features = [finite_feature(0, 0.1, 0.9), essential_feature(1, 0.5)]
        ax = plot_barcode(MIXED_DIAGRAM, features, colors=["#123456", "#abcdef"]).axes[0]

        assert to_hex(finite_bars(ax)[0].get_color()) == "#123456"
        assert to_hex(infinite_bars(ax)[1].get_edgecolor()) == "#abcdef"

    def test_colour_of_feature_k_is_stable_as_more_features_appear(self):
        """Cycle 0 must keep its colour between two figures of one fit."""
        one = plot_barcode(MIXED_DIAGRAM, [finite_feature(0, 0.1, 0.9)]).axes[0]
        two = plot_barcode(
            MIXED_DIAGRAM, [finite_feature(0, 0.1, 0.9), essential_feature(1, 0.5)]
        ).axes[0]

        assert to_hex(finite_bars(one)[0].get_color()) == to_hex(
            finite_bars(two)[0].get_color()
        )

    def test_no_bar_colour_depends_on_persistence(self):
        """Two features with very different persistence, same palette slot:
        the old plasma colormap would have coloured these differently."""
        dgm = np.array([[0.0, 10.0], [0.0, 0.01]])
        a = plot_barcode(dgm, [finite_feature(0, 0.0, 10.0)]).axes[0]
        b = plot_barcode(dgm, [finite_feature(1, 0.0, 0.01)]).axes[0]

        assert to_hex(finite_bars(a)[0].get_color()) == to_hex(
            finite_bars(b)[1].get_color()
        )


# ----------------------------------------------------------------------
# Feature marking
# ----------------------------------------------------------------------


class TestFeatureMarking:

    def test_only_feature_rows_are_marked(self):
        features = [essential_feature(1, 0.5), finite_feature(2, 0.2, 0.35)]
        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]

        assert set(feature_markers(ax)) == {1, 2}

    def test_no_features_means_no_markers(self):
        ax = plot_barcode(MIXED_DIAGRAM).axes[0]
        assert feature_markers(ax) == {}

    def test_marker_sits_at_the_bar_it_marks(self):
        features = [essential_feature(1, 0.5)]
        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]
        marker = feature_markers(ax)[1]

        assert float(marker.get_xdata()[0]) == pytest.approx(0.5)
        assert float(marker.get_ydata()[0]) == pytest.approx(bar_y(ax, 1))

    def test_marker_takes_the_features_colour(self):
        features = [essential_feature(1, 0.5)]
        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]

        assert to_hex(feature_markers(ax)[1].get_markerfacecolor()) == to_hex(
            cycle_colors(1)[0]
        )

    def test_feature_bars_are_thicker_than_plain_bars(self):
        ax = plot_barcode(MIXED_DIAGRAM, [finite_feature(0, 0.1, 0.9)]).axes[0]

        assert finite_bars(ax)[0].get_linewidth() > finite_bars(ax)[2].get_linewidth()

    def test_ytick_labels_name_the_feature_rank(self):
        features = [essential_feature(1, 0.5), finite_feature(2, 0.2, 0.35)]
        ax = plot_barcode(MIXED_DIAGRAM, features).axes[0]

        labels = {t.get_text() for t in ax.get_yticklabels()}
        assert labels == {"#0", "#1"}


# ----------------------------------------------------------------------
# Degenerate numerics — the pers.max() normalisation edge cases
# ----------------------------------------------------------------------


class TestDegenerateDiagrams:

    def test_single_finite_bar(self):
        """One bar means pers.max() == pers, the old normalisation's
        boundary case."""
        ax = plot_barcode(np.array([[0.25, 0.75]])).axes[0]

        assert set(finite_bars(ax)) == {0}
        lo, hi = ax.get_xlim()
        assert np.isfinite([lo, hi]).all() and hi > lo

    def test_zero_persistence_bar_does_not_collapse_the_axes(self):
        """birth == death gives a zero span; a span-proportional pad would
        make the axis limits equal."""
        ax = plot_barcode(np.array([[1.0, 1.0]])).axes[0]

        lo, hi = ax.get_xlim()
        assert np.isfinite([lo, hi]).all()
        assert hi > lo

    def test_zero_persistence_bar_is_drawn_at_its_true_length(self):
        ax = plot_barcode(np.array([[1.0, 1.0], [0.0, 2.0]])).axes[0]
        bar = finite_bars(ax)[0]

        assert list(bar.get_xdata()) == pytest.approx([1.0, 1.0])

    def test_zero_persistence_bar_alongside_an_essential_bar(self):
        """Both former division hazards at once: pers.max() would be inf and
        the finite span would be zero."""
        ax = plot_barcode(np.array([[1.0, 1.0], [1.0, np.inf]])).axes[0]

        assert len(finite_bars(ax)) == 1
        assert len(infinite_bars(ax)) == 1
        lo, hi = ax.get_xlim()
        assert np.isfinite([lo, hi]).all() and hi > lo

    @pytest.mark.parametrize(
        "dgm",
        [
            np.array([[0.0, np.inf]]),
            np.array([[0.0, 0.0]]),
            np.array([[0.5, np.inf], [0.5, 0.5]]),
            MIXED_DIAGRAM,
            np.array([[0.0, 1e6], [0.0, np.inf]]),
        ],
    )
    def test_axis_limits_are_always_finite(self, dgm):
        """`inf` must never leak into an axis limit."""
        ax = plot_barcode(dgm).axes[0]

        assert np.isfinite(ax.get_xlim()).all()
        assert np.isfinite(ax.get_ylim()).all()

    def test_finite_bar_spans_exactly_birth_to_death(self):
        ax = plot_barcode(MIXED_DIAGRAM).axes[0]

        assert list(finite_bars(ax)[0].get_xdata()) == pytest.approx([0.10, 0.90])
        assert list(finite_bars(ax)[2].get_xdata()) == pytest.approx([0.20, 0.35])


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


class TestValidation:

    def test_empty_diagram_still_raises(self):
        with pytest.raises(ValueError, match="No H1 features found"):
            plot_barcode(np.empty((0, 2)))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No H1 features found"):
            plot_barcode([])

    def test_all_infinite_diagram_does_not_raise_the_empty_error(self):
        """Regression: `dgm[np.isfinite(dgm[:, 1])]` emptied this input."""
        plot_barcode(np.array([[0.1, np.inf], [0.4, np.inf]]))

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            plot_barcode(np.array([[0.1, 0.2, 0.3]]))

    def test_nan_rejected_naming_the_row(self):
        with pytest.raises(ValueError, match="NaN at row 1"):
            plot_barcode(np.array([[0.1, 0.2], [np.nan, 0.5]]))

    def test_non_finite_birth_rejected(self):
        with pytest.raises(ValueError, match="birth must be finite"):
            plot_barcode(np.array([[np.inf, np.inf]]))

    def test_death_before_birth_rejected(self):
        with pytest.raises(ValueError, match="death must not precede birth"):
            plot_barcode(np.array([[0.9, 0.2]]))

    def test_feature_index_outside_the_diagram_rejected(self):
        """Marking an arbitrary bar is the failure this spec exists to stop."""
        with pytest.raises(ValueError, match=r"features\[0\].index = 7"):
            plot_barcode(MIXED_DIAGRAM, [finite_feature(7, 0.1, 0.9)])

    def test_two_features_claiming_one_row_rejected(self):
        features = [finite_feature(0, 0.1, 0.9), finite_feature(0, 0.1, 0.9)]
        with pytest.raises(ValueError, match="both claim"):
            plot_barcode(MIXED_DIAGRAM, features)

    def test_colors_length_must_match_features(self):
        with pytest.raises(ValueError, match="one-to-one"):
            plot_barcode(MIXED_DIAGRAM, [finite_feature(0, 0.1, 0.9)], colors=["#fff000", "#000fff"])


# ----------------------------------------------------------------------
# Figure contract (V10, V11)
# ----------------------------------------------------------------------


class TestFigureContract:

    def test_returns_a_figure(self):
        fig = plot_barcode(MIXED_DIAGRAM)
        assert isinstance(fig, plt.Figure)

    def test_never_calls_plt_show(self, monkeypatch):
        called = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))

        plot_barcode(MIXED_DIAGRAM, [essential_feature(1, 0.5)])

        assert called == []

    def test_emits_no_deprecation_warnings(self):
        """V10 on the pinned matplotlib."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", PendingDeprecationWarning)
            warnings.simplefilter(
                "error", matplotlib.MatplotlibDeprecationWarning
            )
            plot_barcode(MIXED_DIAGRAM, [essential_feature(1, 0.5)])

    def test_save_path_writes_a_file(self, tmp_path):
        out = tmp_path / "barcode.png"

        plot_barcode(MIXED_DIAGRAM, save_path=str(out))

        assert out.exists() and out.stat().st_size > 0

    def test_figsize_is_honoured(self):
        fig = plot_barcode(MIXED_DIAGRAM, figsize=(11, 3))
        assert tuple(fig.get_size_inches()) == pytest.approx((11.0, 3.0))

    def test_accepts_a_plain_nested_list_diagram(self):
        ax = plot_barcode([[0.1, 0.9], [0.2, float("inf")]]).axes[0]

        assert len(finite_bars(ax)) == 1
        assert len(infinite_bars(ax)) == 1
