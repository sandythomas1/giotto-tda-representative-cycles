"""The integration gate for spec 001 (T16).

Every acceptance criterion in ``specs/001-representative-cycle-fidelity/
spec.md`` maps to a test here, named for its number.  The other test modules
own their own lanes and test them in isolation with hand-built inputs; this
module exercises the *composition*, through the public API, on one fit — which
is where things that passed in isolation break.

| AC   | Test |
|------|------|
| 1    | ``TestAC1EssentialCircle`` |
| 2    | ``TestAC2ZhuRectangle`` |
| 3    | ``TestAC3BirthMatchesItsOwnEdge`` |
| 4    | ``TestAC4PairingIsPure`` (a/b/c also in ``test_pairing.py``) |
| 5    | ``TestAC5EveryFixtureVerifies`` |
| 6    | ``TestAC6Determinism`` |
| 7    | ``TestAC7InputValidation`` |
| 8    | ``TestAC8Projection`` |
| 9    | ``TestAC9OneCycleOneColour`` |
| 10   | ``TestAC10EssentialBarcode`` |
| 11   | ``TestAC11EssentialDiagramHighlight`` |
| 12   | ``TestAC12FiguresForTwoAndThreeDimensions`` |
| 13   | ``TestAC13SharedGraph`` here; the timing ratio is
         ``test_reconstruction_perf.py`` under ``-m perf`` |
| 13a  | ``test_reconstruction_perf.py`` |
| 14   | ``TestAC14PublicApi`` (plus ``test_representative_cycles.py``, the
         untouched pre-existing suite) |
| 15   | ``TestAC15NoDeprecatedApis`` |
| 16   | ``TestAC16ScriptsRun`` — marked ``scripts``, deselectable |
"""

from __future__ import annotations

import math
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from repcycles import core  # noqa: E402
from repcycles.errors import CycleReconstructionError  # noqa: E402
from repcycles.pairing import pair_generators  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.diagram import GID_HIGHLIGHT  # noqa: E402
from repcycles.plotting.overview import GID_OVERVIEW_CLOUD  # noqa: E402
from repcycles.plotting.panels import GID_CYCLE  # noqa: E402
from repcycles.projection import project_for_cycle  # noqa: E402
from repcycles.validation import check_size  # noqa: E402
from representative_cycles import (  # noqa: E402
    CycleFeature,
    RepresentativeCycles,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Zhu (2013) §2.3: birth = 2, death = √5, a 4-edge cycle.
RECTANGLE = np.array(
    [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Fixtures — the four shapes AC5 names, plus the truncated circle of AC1
# ---------------------------------------------------------------------------


def make_circle(n=60, noise=0.0, seed=0) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    if noise:
        X = X + np.random.default_rng(seed).standard_normal(X.shape) * noise
    return X


def make_figure_eight(n=120, noise=0.05, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2
    theta = np.linspace(0, 2 * np.pi, half, endpoint=False)
    left = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
    right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
    X = np.vstack([left, right])
    return X + rng.standard_normal(X.shape) * noise


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


def make_annulus(n=200, seed=2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0.5**2, 1.5**2, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


@pytest.fixture(scope="module")
def torus_fit():
    return RepresentativeCycles(min_persistence=0.2).fit(make_torus())


@pytest.fixture(scope="module")
def essential_fit():
    """A truncated filtration: one loop that never dies."""
    return RepresentativeCycles(max_edge_length=1.0, min_persistence=0.3).fit(
        make_circle(n=80, noise=0.04)
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def cycle_hexes(ax):
    return [
        to_hex(coll.get_colors()[0])
        for coll in ax.collections
        if coll.get_gid() == GID_CYCLE
    ]


# ---------------------------------------------------------------------------
# AC1 — the headline change
# ---------------------------------------------------------------------------


class TestAC1EssentialCircle:
    """60-point circle at ``max_edge_length=1.0``: one essential feature.

    Before this spec the same fit returned *zero* features — the loop was
    there, and the library said nothing about it.
    """

    @pytest.fixture(scope="class")
    def fit(self):
        return RepresentativeCycles(max_edge_length=1.0).fit(
            make_circle(n=60, noise=0.02, seed=5)
        )

    def test_exactly_one_feature(self, fit):
        assert len(fit.features_) == 1

    def test_it_is_essential_with_infinite_death(self, fit):
        feature = fit.features_[0]
        assert feature.is_essential
        assert np.isposinf(feature.death)
        assert np.isposinf(feature.persistence)

    def test_it_carries_a_real_loop(self, fit):
        assert fit.features_[0].n_edges >= 3
        assert fit.features_[0].is_verified

    def test_opting_out_reproduces_the_old_empty_result(self):
        rc = RepresentativeCycles(
            max_edge_length=1.0, include_essential=False
        ).fit(make_circle(n=60, noise=0.02, seed=5))
        assert rc.features_ == []

    def test_min_persistence_never_filters_it(self):
        rc = RepresentativeCycles(
            max_edge_length=1.0, min_persistence=10.0
        ).fit(make_circle(n=60, noise=0.02, seed=5))
        assert len(rc.features_) == 1 and rc.features_[0].is_essential


# ---------------------------------------------------------------------------
# AC2 — the exactly-checkable case
# ---------------------------------------------------------------------------


class TestAC2ZhuRectangle:

    @pytest.fixture(scope="class")
    def fit(self):
        return RepresentativeCycles().fit(RECTANGLE)

    def test_one_feature_born_at_two_and_dying_at_root_five(self, fit):
        assert len(fit.features_) == 1
        feature = fit.features_[0]
        assert feature.birth == pytest.approx(2.0, abs=1e-4)
        assert feature.death == pytest.approx(math.sqrt(5.0), abs=1e-4)

    def test_the_cycle_is_the_four_sided_perimeter(self, fit):
        feature = fit.features_[0]
        assert feature.n_edges == 4
        assert feature.cycle_length == pytest.approx(6.0, abs=1e-9)
        assert feature.is_verified


# ---------------------------------------------------------------------------
# AC3 — births belong to their own generators
# ---------------------------------------------------------------------------


class TestAC3BirthMatchesItsOwnEdge:
    """The pairing regression, stated as an invariant.

    Positional pairing put a feature's birth next to a *different* class's
    birth edge; the invariant that catches it is that every feature's reported
    birth is the length of the edge it names.
    """

    @pytest.mark.parametrize(
        "kwargs, cloud",
        [
            ({"min_persistence": 0.2}, make_torus()),
            (
                {"max_edge_length": 1.0, "min_persistence": 0.3},
                make_circle(n=80, noise=0.04),
            ),
        ],
    )
    def test_birth_equals_the_length_of_the_birth_edge(self, kwargs, cloud):
        rc = RepresentativeCycles(**kwargs).fit(cloud)
        assert rc.features_

        for feature in rc.features_:
            u, v = feature.birth_edge
            assert feature.birth == pytest.approx(
                rc._dist_matrix_[u, v], abs=1e-4
            )

    def test_a_fit_mixing_finite_and_essential_bars_holds_too(self):
        rc = RepresentativeCycles(max_edge_length=1.2).fit(
            make_figure_eight(n=100, noise=0.05)
        )
        kinds = {f.is_essential for f in rc.features_}
        assert kinds == {True, False}, "fixture must produce both kinds"

        for feature in rc.features_:
            u, v = feature.birth_edge
            assert feature.birth == pytest.approx(
                rc._dist_matrix_[u, v], abs=1e-4
            )


# ---------------------------------------------------------------------------
# AC4 — pairing, as a pure function
# ---------------------------------------------------------------------------


class TestAC4PairingIsPure:
    """No real dataset can be coerced into producing an unmatched generator,
    so the failure path is tested on synthetic arrays.  The full a/b/c matrix
    lives in ``test_pairing.py``; this is the gate's own check that the error
    still reaches callers."""

    def test_an_unmatchable_generator_raises_naming_it(self):
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        generators = np.array([[0, 1, 0, 1]])
        diagram = np.array([[5.0, 6.0]])  # no row of birth 1.0

        with pytest.raises(CycleReconstructionError) as excinfo:
            pair_generators(
                generators, np.empty((0, 2), dtype=int), diagram, D
            )
        assert "1.0" in str(excinfo.value)

    def test_congruent_holes_each_claim_their_own_row(self):
        """Three identical bars are interchangeable, not an error."""
        D = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )
        generators = np.array([[0, 1, 0, 1], [0, 2, 0, 2], [0, 3, 0, 3]])
        diagram = np.array(
            [[1.0, math.sqrt(3.0)]] * 3, dtype=float
        )

        rows = pair_generators(
            generators, np.empty((0, 2), dtype=int), diagram, D
        )
        assert len({row.diagram_index for row in rows}) == 3


# ---------------------------------------------------------------------------
# AC5 / AC6 — verification and determinism
# ---------------------------------------------------------------------------


CANONICAL_FIXTURES = {
    "circle": (make_circle(n=80, noise=0.04, seed=7), 0.5),
    "figure_eight": (make_figure_eight(), 0.4),
    "torus": (make_torus(), 0.2),
    "annulus": (make_annulus(), 0.4),
}


class TestAC5EveryFixtureVerifies:

    @pytest.mark.parametrize("name", sorted(CANONICAL_FIXTURES))
    def test_every_feature_is_verified(self, name):
        cloud, threshold = CANONICAL_FIXTURES[name]
        rc = RepresentativeCycles(min_persistence=threshold).fit(cloud)

        assert rc.features_, f"{name} produced no features to check"
        assert all(f.is_verified for f in rc.features_)

    @pytest.mark.parametrize("name", sorted(CANONICAL_FIXTURES))
    def test_cycle_length_is_the_sum_of_its_edges(self, name):
        cloud, threshold = CANONICAL_FIXTURES[name]
        rc = RepresentativeCycles(min_persistence=threshold).fit(cloud)

        for feature in rc.features_:
            edges = np.asarray(feature.cycle_edges, dtype=int)
            summed = rc._dist_matrix_[edges[:, 0], edges[:, 1]].sum()
            assert feature.cycle_length == pytest.approx(summed, abs=1e-9)

    @pytest.mark.parametrize("name", sorted(CANONICAL_FIXTURES))
    def test_the_path_closes_on_itself(self, name):
        cloud, threshold = CANONICAL_FIXTURES[name]
        rc = RepresentativeCycles(min_persistence=threshold).fit(cloud)

        for feature in rc.features_:
            path = np.asarray(feature.cycle_path, dtype=int)
            assert path[0] == path[-1]
            assert len(path) - 1 == feature.n_edges


class TestAC6Determinism:

    def test_two_fits_produce_identical_cycle_edges(self):
        X = make_torus()
        first = RepresentativeCycles(min_persistence=0.2).fit(X).features_
        second = RepresentativeCycles(min_persistence=0.2).fit(X).features_

        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert a.index == b.index
            assert np.array_equal(a.cycle_edges, b.cycle_edges)
            assert np.array_equal(a.cycle_path, b.cycle_path)

    def test_refitting_the_same_model_does_not_accumulate_features(self):
        X = make_torus()
        rc = RepresentativeCycles(min_persistence=0.2)
        first = len(rc.fit(X).features_)
        assert len(rc.fit(X).features_) == first


# ---------------------------------------------------------------------------
# AC7 — the untrusted-input boundary, one assertion per bullet
# ---------------------------------------------------------------------------


class TestAC7InputValidation:

    def test_nan_in_coordinates_is_rejected_naming_the_index(self):
        X = make_circle(n=20)
        X[7, 1] = np.nan
        with pytest.raises(ValueError, match="7"):
            RepresentativeCycles().fit(X)

    def test_infinite_coordinates_are_rejected(self):
        X = make_circle(n=20)
        X[3, 0] = np.inf
        with pytest.raises(ValueError):
            RepresentativeCycles().fit(X)

    def test_infinity_in_a_precomputed_matrix_means_no_edge(self):
        D = np.array(
            [
                [0.0, 1.0, 1.0, np.inf],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [np.inf, 1.0, 1.0, 0.0],
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = RepresentativeCycles(metric="precomputed").fit(D)

        assert rc.diagrams_ is not None  # the fit completed

    def test_negative_precomputed_distance_is_rejected(self):
        D = np.array([[0.0, -1.0], [-1.0, 0.0]])
        with pytest.raises(ValueError):
            RepresentativeCycles(metric="precomputed").fit(D)

    def test_above_five_thousand_points_warns_without_failing(self):
        # Asserted on the guard rather than on a real 5001-point fit: the
        # behaviour under test is the warning, and ripser on 5001 points would
        # add minutes to the suite for no extra coverage.
        with pytest.warns(ResourceWarning):
            check_size(5001)

    def test_exceeding_an_explicit_max_points_raises(self):
        with pytest.raises(ValueError, match="max_points"):
            RepresentativeCycles(max_points=10).fit(make_circle(n=40))

    def test_validation_runs_before_the_distance_matrix_is_allocated(
        self, monkeypatch
    ):
        def explode(*_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("cdist ran before validation")

        monkeypatch.setattr(core, "cdist", explode)
        X = make_circle(n=20)
        X[2, 0] = np.nan
        with pytest.raises(ValueError):
            RepresentativeCycles().fit(X)


# ---------------------------------------------------------------------------
# AC8 — projection
# ---------------------------------------------------------------------------


class TestAC8Projection:

    def test_every_cycle_beats_the_projection_it_replaces(self, torus_fit):
        """The SVD-optimality invariant: it cannot flake."""
        X = np.asarray(torus_fit.point_cloud_, dtype=float)

        for feature in torus_fit.features_:
            projection = project_for_cycle(X, feature.cycle_vertices)
            assert (
                projection.variance_retained
                >= projection.baseline_retained - 1e-12
            )

    def test_the_panels_draw_that_projection(self, torus_fit):
        """The invariant is worth nothing if the figure ignores it."""
        from repcycles.plotting.panels import GID_CLOUD

        X = np.asarray(torus_fit.point_cloud_, dtype=float)
        fig = torus_fit.plot_matplotlib(max_cycles=1)
        panel = next(
            ax
            for ax in fig.axes
            if any(c.get_gid() == GID_CYCLE for c in ax.collections)
        )
        drawn = next(
            c for c in panel.collections if c.get_gid() == GID_CLOUD
        ).get_offsets()

        expected = project_for_cycle(X, torus_fit.features_[0].cycle_vertices)
        assert np.allclose(np.asarray(drawn), expected.coords)


# ---------------------------------------------------------------------------
# AC9 — one cycle, one colour, in every view
# ---------------------------------------------------------------------------


class TestAC9OneCycleOneColour:
    """The cross-lane criterion: four views built by four separate tasks have
    to agree about one fit."""

    def test_all_four_views_agree(self, torus_fit):
        n = 4
        expected = [to_hex(c) for c in cycle_colors(torus_fit.features_[:n])]

        panels = torus_fit.plot_matplotlib(max_cycles=n)
        overview = torus_fit.plot_overview(max_cycles=n)
        barcode = torus_fit.plot_barcode()
        plotly_fig = torus_fit.plot_plotly(max_cycles=n)

        from_panels = [
            hexes[0]
            for hexes in (cycle_hexes(ax) for ax in panels.axes)
            if hexes
        ]
        overlay_axes = next(
            ax
            for ax in overview.axes
            if any(c.get_gid() == GID_OVERVIEW_CLOUD for c in ax.collections)
        )
        from_overview = cycle_hexes(overlay_axes)

        bars = {}
        for line in barcode.axes[0].lines:
            gid = line.get_gid() or ""
            if gid.startswith("bar-finite-"):
                bars[int(gid.rsplit("-", 1)[1])] = to_hex(line.get_color())
        from_barcode = [
            bars[f.index] for f in torus_fit.features_[:n]
        ]

        # Plotly groups a cycle's traces under `legendgroup="cycle{rank}"`;
        # the loop itself is the `lines` trace (the black birth edge and the
        # vertex markers are separate traces in the same group).
        from_plotly = []
        for rank in range(n):
            group = [
                t
                for t in plotly_fig.data
                if getattr(t, "legendgroup", None) == f"cycle{rank}"
                and t.mode == "lines"
                and t.name != "birth edge"
            ]
            assert group, f"no loop trace for cycle {rank}"
            from_plotly.append(to_hex(group[0].line.color))

        assert from_panels == expected
        assert from_overview == expected
        assert from_barcode == expected
        assert from_plotly == expected


# ---------------------------------------------------------------------------
# AC10 / AC11 — essential classes survive the views
# ---------------------------------------------------------------------------


class TestAC10EssentialBarcode:

    def test_an_infinite_bar_renders_as_an_arrow(self, essential_fit):
        assert any(f.is_essential for f in essential_fit.features_)
        fig = essential_fit.plot_barcode()

        arrows = [
            p
            for p in fig.axes[0].patches
            if isinstance(p, FancyArrowPatch)
            and (p.get_gid() or "").startswith("bar-infinite-")
        ]
        assert arrows

    def test_axis_limits_stay_finite(self, essential_fit):
        ax = essential_fit.plot_barcode().axes[0]
        assert np.isfinite(ax.get_xlim()).all()


class TestAC11EssentialDiagramHighlight:

    def test_the_correct_point_is_highlighted_without_indexerror(
        self, essential_fit
    ):
        fig = essential_fit.plot_matplotlib()
        diagram_axes = next(
            ax for ax in fig.axes if "Persistence Diagram" in ax.get_title()
        )
        highlights = next(
            c
            for c in diagram_axes.collections
            if c.get_gid() == GID_HIGHLIGHT
        )
        drawn = np.asarray(highlights.get_offsets())

        assert len(drawn) == len(essential_fit.features_)
        births = sorted(f.birth for f in essential_fit.features_)
        assert sorted(drawn[:, 0]) == pytest.approx(births, abs=1e-6)
        assert np.isfinite(drawn).all()


# ---------------------------------------------------------------------------
# AC12 — figures for 2-D and 3-D input
# ---------------------------------------------------------------------------


class TestAC12FiguresForTwoAndThreeDimensions:

    @pytest.fixture(
        scope="class", params=["two_dimensional", "three_dimensional"]
    )
    def fit(self, request):
        if request.param == "two_dimensional":
            return RepresentativeCycles(min_persistence=0.4).fit(
                make_annulus()
            )
        return RepresentativeCycles(min_persistence=0.2).fit(make_torus())

    def test_plot_cycle_returns_a_figure(self, fit):
        assert fit.plot_cycle(0) is not None

    def test_plot_plotly_with_the_skeleton_returns_a_figure(self, fit):
        figure = fit.plot_plotly(show_skeleton=True, max_cycles=2)
        assert figure is not None and len(figure.data) > 0

    def test_plot_overview_returns_a_figure(self, fit):
        assert fit.plot_overview(max_cycles=2) is not None


# ---------------------------------------------------------------------------
# AC13 — the shared graph, wired into fit()
# ---------------------------------------------------------------------------


class TestAC13SharedGraph:
    """The timing ratio and the memory budget live in
    ``test_reconstruction_perf.py``.  What belongs in the gate is that the fast
    path is the one ``fit()`` actually takes, and that it changes nothing."""

    def test_fit_builds_one_graph_for_the_whole_fit(self, monkeypatch):
        built = []
        original = core.RipsGraphCache

        class CountingCache(original):
            def __init__(self, dist_matrix, max_edge_length=np.inf):
                built.append(max_edge_length)
                super().__init__(dist_matrix, max_edge_length)

        monkeypatch.setattr(core, "RipsGraphCache", CountingCache)
        rc = RepresentativeCycles(min_persistence=0.2).fit(make_torus())

        assert len(rc.features_) > 1
        assert len(built) == 1

    def test_the_shared_graph_changes_no_result(self, monkeypatch):
        X = make_torus()
        shared = RepresentativeCycles(min_persistence=0.2).fit(X).features_

        monkeypatch.setattr(core, "_MIN_FEATURES_FOR_GRAPH_CACHE", 10**9)
        per_feature = RepresentativeCycles(min_persistence=0.2).fit(X).features_

        assert len(shared) == len(per_feature)
        for a, b in zip(shared, per_feature):
            assert a.cycle_length == pytest.approx(b.cycle_length, abs=1e-9)
            assert a.is_verified == b.is_verified


# ---------------------------------------------------------------------------
# AC14 — the public API is still the public API
# ---------------------------------------------------------------------------


class TestAC14PublicApi:

    def test_the_documented_import_path_still_works(self):
        rc = RepresentativeCycles(min_persistence=0.4).fit(make_annulus())
        assert isinstance(rc.features_[0], CycleFeature)

    def test_the_shim_and_the_package_are_the_same_classes(self):
        import repcycles

        assert RepresentativeCycles is repcycles.RepresentativeCycles
        assert CycleFeature is repcycles.CycleFeature

    def test_positional_construction_through_death_edge_still_works(self):
        feature = CycleFeature(3, 0.5, 1.5, 1.0, (1, 2), (3, 4))

        assert feature.birth_edge == (1, 2)
        assert (feature.cycle_length, feature.is_essential) == (0.0, False)

    def test_computation_imports_no_plotting_stack(self):
        code = (
            "import sys, repcycles.core; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'plotly' not in sys.modules"
        )
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---------------------------------------------------------------------------
# AC15 — no deprecated matplotlib APIs
# ---------------------------------------------------------------------------


class TestAC15NoDeprecatedApis:

    def test_the_whole_matplotlib_surface_is_clean(self, torus_fit):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter(
                "error", matplotlib.MatplotlibDeprecationWarning
            )

            torus_fit.plot_matplotlib(max_cycles=3)
            torus_fit.plot_overview(max_cycles=3)
            torus_fit.plot_cycle(0)
            torus_fit.plot_barcode()

    def test_no_plot_method_shows_a_figure(self, monkeypatch, torus_fit):
        def fail():  # pragma: no cover - only runs on regression
            raise AssertionError("a plot method called plt.show()")

        monkeypatch.setattr(plt, "show", fail)
        torus_fit.plot_matplotlib(max_cycles=2)
        torus_fit.plot_overview(max_cycles=2)
        torus_fit.plot_cycle(0)
        torus_fit.plot_barcode()


# ---------------------------------------------------------------------------
# AC16 — the shipped scripts
# ---------------------------------------------------------------------------


@pytest.mark.scripts
class TestAC16ScriptsRun:
    """Both example scripts run to completion, and Zhu reports five passes.

    Marked ``scripts`` because between them they fit a dozen point clouds and
    write every figure in ``output/`` — a minute of work that has no business
    running on every edit.  Deselect with ``-m "not scripts"``.
    """

    @staticmethod
    def _run(script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-X", "utf8", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_examples_runs_to_completion(self):
        result = self._run("examples.py")
        assert result.returncode == 0, result.stderr[-2000:]
        assert "All outputs saved to" in result.stdout

    def test_zhu_replication_reports_five_passes(self):
        result = self._run("zhu_paper_replication.py")
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.count("[OK]") == 5
