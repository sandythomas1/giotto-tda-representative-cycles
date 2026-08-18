"""Tests for the overview and single-cycle figures (spec 001 T13 — V5, V8).

Structural assertions only — artist counts, offsets, colours and annotation
text.  No image comparison anywhere (constitution, Testing Bar).

Two acceptance criteria live here:

- **AC9 (matplotlib half)** — cycle *k* is the same colour in the overlay, the
  diagram, the barcode, the panels figure and the single-cycle figure of one
  fit.  Compared after normalising through ``to_hex``, because matplotlib
  artists carry RGBA tuples that never compare equal to a hex string (V2).
- **AC12 (matplotlib half)** — ``plot_cycle(0)`` returns a figure for 2-D and
  3-D input.

Unlike the panel tests, these run against a real fitted
:class:`~repcycles.core.RepresentativeCycles`: the views under test are
composites whose whole job is to agree with each other about one fit, and a
hand-built feature list would let them agree about nothing real.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from repcycles.core import RepresentativeCycles  # noqa: E402
from repcycles.errors import CycleReconstructionWarning  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.barcode import NEUTRAL_COLOR  # noqa: E402
from repcycles.plotting.diagram import GID_EMPTY, GID_HIGHLIGHT  # noqa: E402
from repcycles.plotting.overview import (  # noqa: E402
    CONTEXT_COLOR,
    GID_CONTEXT,
    GID_EMPTY_BARCODE,
    GID_OVERVIEW_CLOUD,
    GID_OVERVIEW_NOTE,
    plot_cycle,
    plot_overview,
)
from repcycles.plotting.panels import (  # noqa: E402
    GID_CLOUD,
    GID_CYCLE,
    GID_CYCLE_VERTICES,
    plot_matplotlib,
)
from repcycles.projection import project_for_cycle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_torus(n=300, R=2.0, r=0.8, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack(
        [
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
        ]
    )


def make_noisy_circle(n=80, noise=0.04, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    # Jittered on purpose: a perfectly regular polygon puts every
    # nearest-neighbour distance at the same float64 value, and the loop is
    # then traced or not on round-off.
    return X + rng.standard_normal(X.shape) * noise


@pytest.fixture(scope="module")
def torus_fit():
    """3-D fit with many features — the shared-plane case."""
    return RepresentativeCycles(min_persistence=0.2).fit(make_torus())


@pytest.fixture(scope="module")
def circle_fit():
    """2-D fit — the pass-through-projection case."""
    return RepresentativeCycles(min_persistence=0.5).fit(make_noisy_circle())


@pytest.fixture(scope="module")
def essential_fit():
    """A fit whose only feature never dies (``death = inf``).

    ``min_persistence`` keeps the short-lived noise classes out, which is also
    what keeps this fixture free of reconstruction warnings: those classes are
    filtered before anything is traced.
    """
    return RepresentativeCycles(max_edge_length=1.0, min_persistence=0.3).fit(
        make_noisy_circle()
    )


@pytest.fixture
def empty_fit():
    """A fit with H₁ bars in the diagram but no features above threshold."""
    return RepresentativeCycles(min_persistence=99.0).fit(make_torus(n=150))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Artist lookup helpers
# ---------------------------------------------------------------------------


def collections(ax, gid):
    return [c for c in ax.collections if c.get_gid() == gid]


def collection(ax, gid):
    found = collections(ax, gid)
    assert len(found) <= 1, f"expected at most one {gid}, got {len(found)}"
    return found[0] if found else None


def text(ax, gid):
    found = [t for t in ax.texts if t.get_gid() == gid]
    assert len(found) <= 1, f"expected at most one {gid}, got {len(found)}"
    return found[0] if found else None


def offsets(coll):
    return np.asarray(coll.get_offsets(), dtype=float)


def hexes(colors):
    return [to_hex(c) for c in colors]


def overlay_axes(fig):
    """The cloud panel: the only axes carrying the overlay's cloud."""
    found = [ax for ax in fig.axes if collection(ax, GID_OVERVIEW_CLOUD)]
    assert len(found) == 1
    return found[0]


def diagram_axes(fig):
    return next(ax for ax in fig.axes if "Persistence Diagram" in ax.get_title())


def barcode_axes(fig):
    return next(ax for ax in fig.axes if "Barcode" in ax.get_title())


def bar_colors(ax):
    """``{diagram row: hex}`` for every bar drawn on a barcode axes."""
    found = {}
    for line in ax.lines:
        gid = line.get_gid() or ""
        if gid.startswith("bar-finite-"):
            found[int(gid.rsplit("-", 1)[1])] = to_hex(line.get_color())
    for patch in ax.patches:
        gid = patch.get_gid() or ""
        if gid.startswith("bar-infinite-"):
            found[int(gid.rsplit("-", 1)[1])] = to_hex(patch.get_edgecolor())
    return found


# ---------------------------------------------------------------------------
# plot_overview — structure
# ---------------------------------------------------------------------------


class TestOverviewStructure:

    def test_returns_a_figure_with_the_three_views(self, torus_fit):
        fig = plot_overview(torus_fit)

        assert isinstance(fig, Figure)
        # Cloud, diagram, barcode — plus the diagram's colorbar axes.
        assert overlay_axes(fig) is not None
        assert diagram_axes(fig) is not None
        assert barcode_axes(fig) is not None

    def test_every_selected_cycle_is_overlaid_on_one_pair_of_axes(
        self, torus_fit
    ):
        fig = plot_overview(torus_fit, max_cycles=4)
        ax = overlay_axes(fig)

        assert len(collections(ax, GID_CYCLE)) == 4
        assert len(collections(ax, GID_CYCLE_VERTICES)) == 4

    def test_the_whole_cloud_is_drawn_behind_the_loops(self, torus_fit):
        fig = plot_overview(torus_fit)
        cloud = collection(overlay_axes(fig), GID_OVERVIEW_CLOUD)

        # Context is the point of an overview: a loop drawn without the cloud
        # it came from cannot be judged.
        assert len(offsets(cloud)) == len(torus_fit.point_cloud_)

    def test_max_cycles_caps_the_overlay(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=2)
        assert len(collections(overlay_axes(fig), GID_CYCLE)) == 2

    def test_the_diagram_and_barcode_still_show_every_class(self, torus_fit):
        """`max_cycles` limits the *highlighting*, never the evidence."""
        fig = plot_overview(torus_fit, max_cycles=2)

        assert len(bar_colors(barcode_axes(fig))) == len(torus_fit.diagrams_[1])
        highlights = collection(diagram_axes(fig), GID_HIGHLIGHT)
        assert len(offsets(highlights)) == 2

    def test_never_calls_show(self, monkeypatch, torus_fit):
        def fail():  # pragma: no cover - only runs on regression
            raise AssertionError("plot_overview must never call plt.show()")

        monkeypatch.setattr(plt, "show", fail)
        assert plot_overview(torus_fit) is not None

    def test_saving_writes_the_file(self, tmp_path, torus_fit):
        target = tmp_path / "overview.png"
        plot_overview(torus_fit, save_path=str(target))
        assert target.exists() and target.stat().st_size > 0

    def test_bad_save_directory_fails_before_drawing(self, tmp_path, torus_fit):
        missing = tmp_path / "nope" / "overview.png"
        with pytest.raises(ValueError, match="nope"):
            plot_overview(torus_fit, save_path=str(missing))


# ---------------------------------------------------------------------------
# plot_overview — the shared plane
# ---------------------------------------------------------------------------


class TestSharedProjection:
    """One pair of axes can only carry one plane, and the figure says so."""

    def test_the_overlay_uses_the_plane_of_all_drawn_loops(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=3)
        cycles = torus_fit.features_[:3]
        vertices = np.unique(
            np.concatenate([f.cycle_vertices for f in cycles]).astype(int)
        )
        expected = project_for_cycle(
            np.asarray(torus_fit.point_cloud_, dtype=float), vertices
        )

        drawn = offsets(collection(overlay_axes(fig), GID_OVERVIEW_CLOUD))
        assert np.allclose(drawn, expected.coords)

    def test_the_annotation_quotes_the_shared_fraction(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=3)
        note = text(overlay_axes(fig), GID_OVERVIEW_NOTE)

        assert "shared best-fit plane over all 3 loops" in note.get_text()
        assert "plot_cycle()" in note.get_text()

    def test_two_dimensional_input_quotes_no_fraction(self, circle_fit):
        fig = plot_overview(circle_fit)
        note = text(overlay_axes(fig), GID_OVERVIEW_NOTE)

        assert "2-D input" in note.get_text()
        assert "%" not in note.get_text()

    def test_two_dimensional_coordinates_are_untouched(self, circle_fit):
        fig = plot_overview(circle_fit)
        drawn = offsets(collection(overlay_axes(fig), GID_OVERVIEW_CLOUD))
        assert np.allclose(drawn, np.asarray(circle_fit.point_cloud_))


# ---------------------------------------------------------------------------
# AC9 — one cycle, one colour, everywhere
# ---------------------------------------------------------------------------


class TestAcceptanceCriterion9:

    def test_overlay_colours_follow_cycle_colors_in_feature_order(
        self, torus_fit
    ):
        fig = plot_overview(torus_fit, max_cycles=5)
        expected = cycle_colors(torus_fit.features_[:5])

        drawn = [
            to_hex(coll.get_colors()[0])
            for coll in collections(overlay_axes(fig), GID_CYCLE)
        ]
        assert drawn == hexes(expected)

    def test_overlay_matches_the_diagram_highlights(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=5)
        highlights = collection(diagram_axes(fig), GID_HIGHLIGHT)

        overlay = [
            to_hex(coll.get_colors()[0])
            for coll in collections(overlay_axes(fig), GID_CYCLE)
        ]
        assert hexes(highlights.get_edgecolors()) == overlay

    def test_overlay_matches_the_barcode_bars(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=5)
        bars = bar_colors(barcode_axes(fig))
        expected = hexes(cycle_colors(torus_fit.features_[:5]))

        for feature, color in zip(torus_fit.features_[:5], expected):
            assert bars[feature.index] == color

    def test_bars_without_a_drawn_cycle_stay_neutral(self, torus_fit):
        fig = plot_overview(torus_fit, max_cycles=2)
        bars = bar_colors(barcode_axes(fig))
        drawn_rows = {f.index for f in torus_fit.features_[:2]}

        others = [c for row, c in bars.items() if row not in drawn_rows]
        assert others and set(others) == {to_hex(NEUTRAL_COLOR)}

    def test_the_same_colour_appears_in_panels_overview_and_plot_cycle(
        self, torus_fit
    ):
        """The end-to-end half of AC9: three independent figures, one fit."""
        expected = hexes(cycle_colors(torus_fit.features_[:3]))

        overview = plot_overview(torus_fit, max_cycles=3)
        panels = plot_matplotlib(torus_fit, max_cycles=3)

        from_overview = [
            to_hex(coll.get_colors()[0])
            for coll in collections(overlay_axes(overview), GID_CYCLE)
        ]
        from_panels = [
            to_hex(collection(ax, GID_CYCLE).get_colors()[0])
            for ax in panels.axes
            if collection(ax, GID_CYCLE)
        ]
        from_single = [
            to_hex(
                collection(
                    plot_cycle(torus_fit, k).axes[0], GID_CYCLE
                ).get_colors()[0]
            )
            for k in range(3)
        ]

        assert from_overview == expected
        assert from_panels == expected
        assert from_single == expected


# ---------------------------------------------------------------------------
# plot_overview — empty and essential states
# ---------------------------------------------------------------------------


class TestOverviewEdgeStates:

    def test_zero_features_still_returns_a_figure(self, empty_fit):
        fig = plot_overview(empty_fit)

        assert isinstance(fig, Figure)
        assert len(empty_fit.features_) == 0
        assert not collections(overlay_axes(fig), GID_CYCLE)

    def test_zero_features_still_draws_the_cloud_and_the_bars(self, empty_fit):
        fig = plot_overview(empty_fit)

        cloud = collection(overlay_axes(fig), GID_OVERVIEW_CLOUD)
        assert len(offsets(cloud)) == len(empty_fit.point_cloud_)
        # The classes exist; none of them cleared the threshold.  Saying so is
        # the whole value of the figure in this state.
        assert len(bar_colors(barcode_axes(fig))) > 0

    def test_zero_features_annotates_rather_than_quoting_a_fraction(
        self, empty_fit
    ):
        note = text(overlay_axes(plot_overview(empty_fit)), GID_OVERVIEW_NOTE)
        assert "no cycles to project" in note.get_text()

    def test_a_fit_with_no_diagram_rows_renders_both_empty_states(self):
        """Two points cannot make a loop: no features *and* no bars."""
        rc = RepresentativeCycles().fit(np.array([[0.0, 0.0], [1.0, 0.0]]))
        fig = plot_overview(rc)

        assert text(diagram_axes(fig), GID_EMPTY) is not None
        assert text(barcode_axes(fig), GID_EMPTY_BARCODE) is not None

    def test_an_essential_feature_renders_without_raising(self, essential_fit):
        fig = plot_overview(essential_fit)

        assert essential_fit.features_[0].is_essential
        assert len(collections(overlay_axes(fig), GID_CYCLE)) == 1
        # inf must never reach an axis limit.
        for ax in (overlay_axes(fig), diagram_axes(fig), barcode_axes(fig)):
            assert np.isfinite(ax.get_xlim()).all()
            assert np.isfinite(ax.get_ylim()).all()

    def test_unfitted_model_is_rejected(self):
        with pytest.raises(RuntimeError, match="fit"):
            plot_overview(RepresentativeCycles())

    @pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
    def test_bad_max_cycles_is_rejected(self, torus_fit, bad):
        with pytest.raises((TypeError, ValueError)):
            plot_overview(torus_fit, max_cycles=bad)


# ---------------------------------------------------------------------------
# AC12 — plot_cycle
# ---------------------------------------------------------------------------


class TestAcceptanceCriterion12:

    def test_returns_a_figure_for_three_dimensional_input(self, torus_fit):
        fig = plot_cycle(torus_fit, 0)
        assert isinstance(fig, Figure)
        assert collection(fig.axes[0], GID_CYCLE) is not None

    def test_returns_a_figure_for_two_dimensional_input(self, circle_fit):
        fig = plot_cycle(circle_fit, 0)
        assert isinstance(fig, Figure)
        assert collection(fig.axes[0], GID_CYCLE) is not None

    def test_returns_a_figure_for_an_essential_cycle(self, essential_fit):
        fig = plot_cycle(essential_fit, 0)
        assert isinstance(fig, Figure)
        assert "∞" in fig.axes[0].get_title()


class TestSingleCycleContext:
    """V8: the neighbourhood the reconstruction actually searched."""

    def test_context_is_the_points_within_the_birth_radius(self, torus_fit):
        feature = torus_fit.features_[0]
        u, v = feature.birth_edge
        radius = torus_fit._dist_matrix_[u, v]
        vertices = np.unique(np.asarray(feature.cycle_vertices, dtype=int))
        expected = np.flatnonzero(
            (torus_fit._dist_matrix_[vertices] <= radius).any(axis=0)
        )

        context = collection(plot_cycle(torus_fit, 0).axes[0], GID_CONTEXT)
        assert len(offsets(context)) == len(expected)

    def test_context_is_measured_in_the_original_metric(self, torus_fit):
        """Not in the projected plane — a rotation must not recruit
        neighbours that are far away in the data."""
        feature = torus_fit.features_[0]
        u, v = feature.birth_edge
        radius = torus_fit._dist_matrix_[u, v]
        projection = project_for_cycle(
            np.asarray(torus_fit.point_cloud_, dtype=float),
            feature.cycle_vertices,
        )

        drawn = offsets(collection(plot_cycle(torus_fit, 0).axes[0], GID_CONTEXT))
        vertices = np.unique(np.asarray(feature.cycle_vertices, dtype=int))
        expected = projection.coords[
            np.flatnonzero(
                (torus_fit._dist_matrix_[vertices] <= radius).any(axis=0)
            )
        ]
        assert np.allclose(np.sort(drawn, axis=0), np.sort(expected, axis=0))

    def test_context_sits_above_the_cloud_and_below_the_loop(self, torus_fit):
        ax = plot_cycle(torus_fit, 0).axes[0]

        assert (
            collection(ax, GID_CLOUD).get_zorder()
            < collection(ax, GID_CONTEXT).get_zorder()
            < collection(ax, GID_CYCLE).get_zorder()
        )

    def test_context_has_its_own_colour_and_legend_entry(self, torus_fit):
        ax = plot_cycle(torus_fit, 0).axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]

        assert to_hex(collection(ax, GID_CONTEXT).get_facecolor()[0]) == to_hex(
            CONTEXT_COLOR
        )
        # The panel's own keys must survive: the context layer is added to the
        # legend, never substituted for it.
        assert any("birth radius" in label for label in labels)
        assert any("birth edge" in label for label in labels)
        assert any("cycle path" in label for label in labels)

    def test_a_wider_radius_draws_more_context(self, torus_fit):
        narrow = plot_cycle(torus_fit, 0)
        wide = plot_cycle(torus_fit, 0, context_radius=10.0)

        n_narrow = len(offsets(collection(narrow.axes[0], GID_CONTEXT)))
        n_wide = len(offsets(collection(wide.axes[0], GID_CONTEXT)))
        assert n_wide > n_narrow
        assert n_wide == len(torus_fit.point_cloud_)

    def test_a_zero_radius_still_draws_the_cycle(self, torus_fit):
        ax = plot_cycle(torus_fit, 0, context_radius=0.0).axes[0]

        # The cycle's own vertices are at distance 0 from themselves.
        assert collection(ax, GID_CYCLE) is not None
        assert len(offsets(collection(ax, GID_CONTEXT))) == len(
            np.unique(torus_fit.features_[0].cycle_vertices)
        )

    def test_the_cycle_keeps_its_own_best_fit_plane(self, torus_fit):
        """Unlike the overview, nothing is shared here."""
        feature = torus_fit.features_[0]
        expected = project_for_cycle(
            np.asarray(torus_fit.point_cloud_, dtype=float),
            feature.cycle_vertices,
        )

        ax = plot_cycle(torus_fit, 0).axes[0]
        assert np.allclose(
            offsets(collection(ax, GID_CLOUD)), expected.coords
        )


class TestSingleCycleValidation:

    def test_unfitted_model_is_rejected(self):
        with pytest.raises(RuntimeError, match="fit"):
            plot_cycle(RepresentativeCycles(), 0)

    def test_index_past_the_end_names_the_valid_range(self, circle_fit):
        n = len(circle_fit.features_)
        with pytest.raises(IndexError, match=f"0..{n - 1}"):
            plot_cycle(circle_fit, n)

    def test_a_fit_with_no_features_says_so(self, empty_fit):
        with pytest.raises(IndexError, match="no features"):
            plot_cycle(empty_fit, 0)

    def test_negative_index_is_rejected_rather_than_wrapped(self, torus_fit):
        # -1 would silently draw the *least* persistent cycle, the opposite of
        # what a most-persistent-first list leads a reader to expect.
        with pytest.raises(IndexError):
            plot_cycle(torus_fit, -1)

    @pytest.mark.parametrize("bad", [1.5, "0", None, True])
    def test_non_integer_index_is_rejected(self, torus_fit, bad):
        with pytest.raises(TypeError):
            plot_cycle(torus_fit, bad)

    @pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0])
    def test_bad_context_radius_is_rejected(self, torus_fit, bad):
        with pytest.raises(ValueError):
            plot_cycle(torus_fit, 0, context_radius=bad)

    def test_never_calls_show(self, monkeypatch, torus_fit):
        def fail():  # pragma: no cover - only runs on regression
            raise AssertionError("plot_cycle must never call plt.show()")

        monkeypatch.setattr(plt, "show", fail)
        assert plot_cycle(torus_fit, 0) is not None

    def test_saving_writes_the_file(self, tmp_path, torus_fit):
        target = tmp_path / "cycle.png"
        plot_cycle(torus_fit, 0, save_path=str(target))
        assert target.exists() and target.stat().st_size > 0

    def test_bad_save_directory_fails_before_drawing(self, tmp_path, torus_fit):
        with pytest.raises(ValueError, match="nope"):
            plot_cycle(torus_fit, 0, save_path=str(tmp_path / "nope" / "c.png"))


# ---------------------------------------------------------------------------
# AC15 — the deprecation gate
# ---------------------------------------------------------------------------


class TestDeprecationGate:

    def test_no_deprecation_warnings_from_an_overview(self, torus_fit):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter(
                "error", matplotlib.MatplotlibDeprecationWarning
            )
            warnings.simplefilter("ignore", CycleReconstructionWarning)
            plot_overview(torus_fit)

    def test_no_deprecation_warnings_from_a_single_cycle(self, torus_fit):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter(
                "error", matplotlib.MatplotlibDeprecationWarning
            )
            warnings.simplefilter("ignore", CycleReconstructionWarning)
            plot_cycle(torus_fit, 0)

    def test_the_composite_layout_emits_no_layout_warning(self, torus_fit):
        """A colorbar inside a nested gridspec is what `tight_layout` cannot
        handle; the figure uses constrained layout so the user never sees a
        warning they cannot act on."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_overview(torus_fit)

        assert not [
            w for w in caught if "tight_layout" in str(w.message)
        ]


# ---------------------------------------------------------------------------
# The methods on the model
# ---------------------------------------------------------------------------


class TestModelMethods:

    def test_plot_overview_method_delegates(self, torus_fit):
        fig = torus_fit.plot_overview(max_cycles=2)
        assert len(collections(overlay_axes(fig), GID_CYCLE)) == 2

    def test_plot_cycle_method_delegates(self, torus_fit):
        fig = torus_fit.plot_cycle(1)
        assert "Rank 2 cycle" in fig.axes[0].get_title()

    def test_the_methods_import_no_plotting_stack_at_module_scope(self):
        """The constitution's architecture rule, re-checked at the seam these
        two methods added."""
        import subprocess
        import sys

        code = (
            "import sys; import repcycles.core; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib leaked'; "
            "assert 'plotly' not in sys.modules, 'plotly leaked'"
        )
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0
