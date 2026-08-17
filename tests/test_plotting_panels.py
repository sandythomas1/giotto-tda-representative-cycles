"""Tests for the cycle panels (spec 001 T12 — V2, V3, V4, V10, V11; AC8, AC15).

Structural assertions only — artist counts, offsets, colours, line styles and
annotation text.  No image comparison anywhere (constitution, Testing Bar).

The central regression is AC8.  Every panel used to draw ``X[:, :2]``, which
on the torus fixture keeps as little as 0.46 of a tube-loop's variance; the
panel must now draw the cycle's own best-fit plane and *say* what that plane
retained.  The proof that the panel really used the projection is not the
annotation text but the drawn data: the context cloud's offsets are compared
against :func:`project_for_cycle` coordinates point for point.

Fixtures are deliberately self-contained (ripser + scipy only).  ``repcycles
.core`` and ``repcycles.reconstruction`` are mid-change in neighbouring tasks,
and these tests must not go red for someone else's edit — except in the one
end-to-end smoke test, which is explicitly there to catch wiring drift.
"""

from __future__ import annotations

import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from gph import ripser_parallel  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from repcycles.feature import CycleFeature  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.diagram import GID_HIGHLIGHT  # noqa: E402
from repcycles.plotting.panels import (  # noqa: E402
    GID_BIRTH_EDGE,
    GID_BIRTH_VERTICES,
    GID_CLOUD,
    GID_CYCLE,
    GID_CYCLE_VERTICES,
    GID_PROJECTION_NOTE,
    GID_UNVERIFIED,
    draw_cycle_panel,
    plot_matplotlib,
)
from repcycles.projection import project_for_cycle  # noqa: E402

#: AC8, absolute half.
MIN_RETAINED = 0.90

#: Slack for the SVD-optimality invariant (different arithmetic, same value).
OPTIMALITY_SLACK = 1e-12


# ---------------------------------------------------------------------------
# Artist lookup helpers
# ---------------------------------------------------------------------------


def collection(ax, gid):
    """The single collection carrying *gid*, or ``None``."""
    found = [c for c in ax.collections if c.get_gid() == gid]
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


def percentages(message: str) -> list:
    """Every percentage quoted in an annotation, as fractions."""
    return [float(m) / 100.0 for m in re.findall(r"(\d+\.\d+)%", message)]


# ---------------------------------------------------------------------------
# Point clouds and features, built without repcycles.core
# ---------------------------------------------------------------------------


def make_torus(n=600, R=2.0, r=0.5, noise=0.05, seed=8) -> np.ndarray:
    """Random sample from a torus in R³ — the AC8 benchmark cloud."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X = np.column_stack([x, y, z])
    return X + rng.standard_normal(X.shape) * noise


def make_tilted_circle(n=40, seed=3) -> np.ndarray:
    """A planar circle in R³ on a plane tilted 55° out of the x-y plane.

    Exactly planar, so its best-fit plane retains ~1.0 while ``X[:, :2]`` sees
    it foreshortened — the V4 failure in its simplest reproducible form.

    Angles are jittered because a *perfectly* regular polygon puts every
    nearest-neighbour distance at the same float64 value: the Rips graph at
    the birth radius then keeps or drops those edges on round-off, the loop
    cannot be traced, and the fixture would be testing that knife edge
    instead of the projection.
    """
    rng = np.random.default_rng(seed)
    theta = np.sort(
        np.linspace(0, 2 * np.pi, n, endpoint=False)
        + rng.uniform(-0.02, 0.02, n)
    )
    flat = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    c, s = np.cos(np.deg2rad(55)), np.sin(np.deg2rad(55))
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return flat @ tilt.T


def loop_through(D: np.ndarray, u: int, v: int) -> list:
    """Vertices of the shortest u→v path at radius ``D[u, v]``, birth edge
    excluded, closed by that edge — the library's own method, reimplemented
    here so these tests depend only on scipy."""
    adjacency = D <= D[u, v]
    adjacency[u, v] = adjacency[v, u] = False
    np.fill_diagonal(adjacency, False)

    rows, cols = np.nonzero(adjacency)
    graph = csr_matrix((D[rows, cols], (rows, cols)), shape=D.shape)
    distances, predecessors = dijkstra(
        graph, directed=False, indices=u, return_predecessors=True
    )
    if np.isinf(distances[v]):
        return [u, v, u]

    path, current = [], v
    while current != u and current >= 0:
        path.append(int(current))
        current = int(predecessors[current])
    path.append(int(u))
    path.reverse()
    path.append(int(u))  # close the loop through the birth edge
    return path


def feature_from_path(
    D: np.ndarray,
    path: list,
    index: int = 0,
    birth: float = 1.0,
    death: float = 2.0,
    is_verified: bool = True,
    is_essential: bool = False,
) -> CycleFeature:
    """A ``CycleFeature`` for a closed vertex path (first vertex repeated)."""
    edges = np.array([[path[i], path[i + 1]] for i in range(len(path) - 1)])
    length = float(sum(D[a, b] for a, b in edges)) if len(edges) else 0.0
    persistence = np.inf if is_essential else death - birth
    return CycleFeature(
        index=index,
        birth=birth,
        death=np.inf if is_essential else death,
        persistence=persistence,
        birth_edge=(int(path[-2]), int(path[-1])),
        death_edge=(-1, -1) if is_essential else (0, 0),
        cycle_edges=edges,
        cycle_vertices=np.unique(np.asarray(path[:-1], dtype=int)),
        cycle_length=length,
        cycle_path=np.asarray(path, dtype=int),
        is_essential=is_essential,
        is_verified=is_verified,
    )


def top_h1_features(X: np.ndarray, k: int):
    """The *k* most persistent H₁ classes of *X*, as (diagram, features)."""
    D = cdist(X, X)
    result = ripser_parallel(
        X.astype(np.float32), maxdim=1, return_generators=True
    )
    diagram = result["dgms"][1]
    generators = result["gens"][1][0]

    persistence = diagram[:, 1] - diagram[:, 0]
    ranked = np.argsort(-persistence)[:k]

    features = []
    for row in ranked:
        u, v = int(generators[row][0]), int(generators[row][1])
        features.append(
            feature_from_path(
                D,
                loop_through(D, u, v),
                index=int(row),
                birth=float(diagram[row, 0]),
                death=float(diagram[row, 1]),
            )
        )
    return diagram, features


class FakeFit:
    """The read-only surface ``plot_matplotlib`` consumes from a fitted model."""

    def __init__(self, X, features, diagram):
        self.point_cloud_ = np.asarray(X, dtype=float)
        self.features_ = list(features)
        self.diagrams_ = [np.empty((0, 2)), np.asarray(diagram, dtype=float)]


@pytest.fixture(scope="module")
def torus():
    X = make_torus()
    diagram, features = top_h1_features(X, 6)
    return X, diagram, features


@pytest.fixture
def ax():
    fig, axes = plt.subplots()
    yield axes
    plt.close(fig)


@pytest.fixture
def square_cloud():
    """A 2-D unit square with a hole in the middle of it."""
    ring = np.array(
        [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 12, False)]
    )
    return np.vstack([ring, [[0.0, 3.0], [3.0, 0.0]]])


@pytest.fixture
def square_feature(square_cloud):
    D = cdist(square_cloud, square_cloud)
    return feature_from_path(D, list(range(12)) + [0])


# ===========================================================================
# AC8 — 3-D panels draw the cycle's own plane
# ===========================================================================


class TestTorusProjection:
    """AC8 on the fixture the requirement was measured on."""

    def test_panel_draws_the_best_fit_plane_not_the_first_two_columns(
        self, torus, ax
    ):
        """The proof the panel used the projection: its cloud offsets *are*
        the projected coordinates, and they are not ``X[:, :2]``."""
        X, _, features = torus
        feature = features[0]
        expected = project_for_cycle(X, feature.cycle_vertices)

        draw_cycle_panel(ax, X, feature, cycle_colors(1)[0], rank=1)

        drawn = offsets(collection(ax, GID_CLOUD))
        np.testing.assert_allclose(drawn, expected.coords, atol=1e-12)
        assert not np.allclose(drawn, X[:, :2])

    def test_top_six_panels_each_retain_at_least_ninety_percent(self, torus):
        """AC8, absolute half — measured through the projection each panel
        draws, one panel per cycle."""
        X, _, features = torus
        retained = []
        for rank, feature in enumerate(features[:6], start=1):
            fig, axes = plt.subplots()
            draw_cycle_panel(axes, X, feature, cycle_colors(6)[rank - 1], rank)
            projection = project_for_cycle(X, feature.cycle_vertices)
            np.testing.assert_allclose(
                offsets(collection(axes, GID_CLOUD)),
                projection.coords,
                atol=1e-12,
            )
            retained.append(projection.variance_retained)
            plt.close(fig)

        assert min(retained) >= MIN_RETAINED, (
            "AC8 absolute: per-panel retention = "
            f"{[round(v, 4) for v in retained]}"
        )

    def test_every_panel_beats_the_projection_it_replaces(self, torus):
        """AC8, relative half — the SVD-optimality invariant, which cannot
        flake, plus a guard that the fixture still exhibits the failure."""
        X, _, features = torus
        projections = [
            project_for_cycle(X, f.cycle_vertices) for f in features[:6]
        ]
        for p in projections:
            assert p.variance_retained >= p.baseline_retained - OPTIMALITY_SLACK

        worst_baseline = min(p.baseline_retained for p in projections)
        assert worst_baseline < 0.70, (
            "fixture no longer exhibits the failure V4 exists to fix: worst "
            f"X[:, :2] retention = {worst_baseline:.4f}"
        )

    def test_panel_annotates_both_the_fitted_and_the_naive_fraction(
        self, torus, ax
    ):
        X, _, features = torus
        feature = features[0]
        projection = project_for_cycle(X, feature.cycle_vertices)

        draw_cycle_panel(ax, X, feature, cycle_colors(1)[0], rank=1)

        note = text(ax, GID_PROJECTION_NOTE)
        assert note is not None
        quoted = percentages(note.get_text())
        assert quoted == [
            pytest.approx(round(projection.variance_retained, 3), abs=5e-3),
            pytest.approx(round(projection.baseline_retained, 3), abs=5e-3),
        ]
        assert "best-fit plane" in note.get_text()

    def test_whole_cloud_is_drawn_so_the_loop_keeps_its_context(
        self, torus, ax
    ):
        X, _, features = torus
        draw_cycle_panel(ax, X, features[0], cycle_colors(1)[0], rank=1)

        cloud = offsets(collection(ax, GID_CLOUD))
        cycle_points = offsets(collection(ax, GID_CYCLE_VERTICES))
        assert len(cloud) == len(X)
        assert len(cycle_points) < len(X)


# ===========================================================================
# 2-D pass-through and degenerate cycles
# ===========================================================================


class TestTwoDimensionalPassthrough:

    def test_coordinates_are_untouched(self, ax, square_cloud, square_feature):
        draw_cycle_panel(ax, square_cloud, square_feature, "#0072B2", rank=1)
        np.testing.assert_allclose(
            offsets(collection(ax, GID_CLOUD)), square_cloud
        )

    def test_annotation_quotes_no_fraction(
        self, ax, square_cloud, square_feature
    ):
        """Nothing was projected, so there is no retained fraction to quote;
        a cheerful '100%' here would be noise dressed as information."""
        draw_cycle_panel(ax, square_cloud, square_feature, "#0072B2", rank=1)

        message = text(ax, GID_PROJECTION_NOTE).get_text()
        assert "2-D input" in message
        assert percentages(message) == []


class TestDegenerateCycle:
    """A cycle with no well-defined plane is reported as such, never as a
    good fit (constitution: no silent degradation)."""

    @staticmethod
    def collinear_cloud():
        t = np.linspace(-1.0, 1.0, 9)
        line = np.column_stack([t, t, t])
        return np.vstack([line, [[0.5, -0.5, 0.25], [-0.3, 0.8, -0.1]]])

    def test_annotation_says_degenerate_and_quotes_the_fallback(self, ax):
        X = self.collinear_cloud()
        D = cdist(X, X)
        feature = feature_from_path(D, list(range(9)) + [0])
        projection = project_for_cycle(X, feature.cycle_vertices)
        assert projection.is_degenerate  # fixture guard

        draw_cycle_panel(ax, X, feature, "#0072B2", rank=1)

        message = text(ax, GID_PROJECTION_NOTE).get_text()
        assert "degenerate" in message
        assert "best-fit plane" not in message
        # The line x = y = z keeps exactly 2/3 under X[:, :2]; the panel says
        # so rather than implying a good fit.
        assert percentages(message) == [pytest.approx(2 / 3, abs=5e-3)]

    def test_still_draws_a_usable_panel(self, ax):
        X = self.collinear_cloud()
        D = cdist(X, X)
        feature = feature_from_path(D, list(range(9)) + [0])

        draw_cycle_panel(ax, X, feature, "#0072B2", rank=1)

        assert len(offsets(collection(ax, GID_CLOUD))) == len(X)
        assert collection(ax, GID_CYCLE) is not None


# ===========================================================================
# V2 / V3 — shared colours
# ===========================================================================


class TestSharedColours:

    def test_panel_artists_take_the_supplied_hex_colour(
        self, ax, square_cloud, square_feature
    ):
        color = cycle_colors(3)[2]
        draw_cycle_panel(ax, square_cloud, square_feature, color, rank=3)

        assert hexes(collection(ax, GID_CYCLE).get_colors()) == [to_hex(color)]
        assert hexes(
            collection(ax, GID_CYCLE_VERTICES).get_facecolors()
        ) == [to_hex(color)]

    def test_figure_colours_follow_cycle_colors_in_feature_order(self, torus):
        X, diagram, features = torus
        fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=4)
        expected = [to_hex(c) for c in cycle_colors(4)]

        drawn = [
            hexes(collection(panel, GID_CYCLE).get_colors())[0]
            for panel in fig.axes[1:5]
        ]
        assert drawn == expected
        plt.close(fig)

    def test_panel_colour_matches_its_diagram_highlight(self, torus):
        """V2 across two views of the same fit: the ring in the persistence
        diagram is the same colour as the loop it points at."""
        X, diagram, features = torus
        fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=3)

        highlight = collection(fig.axes[0], GID_HIGHLIGHT)
        assert hexes(highlight.get_edgecolors()) == [
            hexes(collection(panel, GID_CYCLE).get_colors())[0]
            for panel in fig.axes[1:4]
        ]
        plt.close(fig)

    def test_no_colour_is_sampled_from_a_continuous_colormap(self):
        """V3: `cm.tab10(np.linspace(...))` and friends are gone for good.

        Asserted over the parsed syntax tree rather than the raw text, so the
        module docstring may keep *describing* the misuse it removed.
        """
        import ast
        import inspect

        from repcycles.plotting import panels

        tree = ast.parse(inspect.getsource(panels))
        referenced = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        assert not referenced & {"cm", "tab10", "linspace", "get_cmap"}
        assert not hasattr(panels, "cm")


# ===========================================================================
# Essential features: ∞, never nan
# ===========================================================================


class TestEssentialFeature:

    @staticmethod
    def essential(square_cloud):
        D = cdist(square_cloud, square_cloud)
        return feature_from_path(
            D, list(range(12)) + [0], birth=0.5, is_essential=True
        )

    def test_title_renders_infinity_for_death_and_persistence(
        self, ax, square_cloud
    ):
        feature = self.essential(square_cloud)
        assert np.isposinf(feature.death) and np.isposinf(feature.persistence)

        draw_cycle_panel(ax, square_cloud, feature, "#0072B2", rank=1)

        title = ax.get_title()
        assert title.count("∞") == 2
        assert "inf" not in title.lower()

    def test_no_nan_reaches_the_panel_text(self, ax, square_cloud):
        feature = self.essential(square_cloud)
        draw_cycle_panel(ax, square_cloud, feature, "#0072B2", rank=1)

        rendered = ax.get_title() + text(ax, GID_PROJECTION_NOTE).get_text()
        assert "nan" not in rendered.lower()

    def test_figure_with_an_essential_bar_renders(self, square_cloud):
        feature = self.essential(square_cloud)
        diagram = np.array([[0.5, np.inf], [0.2, 0.4]])
        fig = plot_matplotlib(FakeFit(square_cloud, [feature], diagram))

        assert isinstance(fig, Figure)
        assert np.isfinite(fig.axes[0].get_ylim()).all()
        plt.close(fig)


# ===========================================================================
# Verification is visible
# ===========================================================================


class TestVerificationIsVisible:

    def test_unverified_loop_is_badged_and_drawn_differently(
        self, square_cloud
    ):
        D = cdist(square_cloud, square_cloud)
        path = list(range(12)) + [0]
        verified = feature_from_path(D, path, is_verified=True)
        unverified = feature_from_path(D, path, is_verified=False)

        fig, (left, right) = plt.subplots(1, 2)
        draw_cycle_panel(left, square_cloud, verified, "#0072B2", rank=1)
        draw_cycle_panel(right, square_cloud, unverified, "#0072B2", rank=2)

        assert text(left, GID_UNVERIFIED) is None
        assert text(right, GID_UNVERIFIED) is not None
        assert (
            collection(left, GID_CYCLE).get_linestyle()
            != collection(right, GID_CYCLE).get_linestyle()
        )
        plt.close(fig)

    def test_unverified_loop_is_named_in_the_legend(self, ax, square_cloud):
        D = cdist(square_cloud, square_cloud)
        feature = feature_from_path(D, list(range(12)) + [0], is_verified=False)

        draw_cycle_panel(ax, square_cloud, feature, "#0072B2", rank=1)

        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("unverified" in label for label in labels)


# ===========================================================================
# Structure of the composite figure (V11)
# ===========================================================================


class TestFigureContract:

    def test_returns_a_figure_with_one_panel_per_cycle_plus_the_diagram(
        self, torus
    ):
        X, diagram, features = torus
        fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=3)

        assert isinstance(fig, Figure)
        # 1 diagram + 3 panels + the diagram's colorbar axes
        assert len([a for a in fig.axes if a.get_gid() != "colorbar"]) >= 4
        for panel in fig.axes[1:4]:
            assert collection(panel, GID_CLOUD) is not None
            assert text(panel, GID_PROJECTION_NOTE) is not None
        plt.close(fig)

    def test_zero_features_still_returns_a_figure(self, square_cloud):
        diagram = np.array([[0.2, 0.9]])
        fig = plot_matplotlib(FakeFit(square_cloud, [], diagram))

        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_empty_diagram_and_no_features_still_returns_a_figure(
        self, square_cloud
    ):
        fig = plot_matplotlib(FakeFit(square_cloud, [], np.empty((0, 2))))

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_max_cycles_caps_the_panels(self, torus):
        X, diagram, features = torus
        fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=2)

        panels_drawn = [
            a for a in fig.axes if collection(a, GID_CLOUD) is not None
        ]
        assert len(panels_drawn) == 2
        plt.close(fig)

    def test_never_calls_show(self, monkeypatch, torus):
        def explode(*args, **kwargs):
            raise AssertionError("plot_matplotlib must never call plt.show()")

        monkeypatch.setattr(plt, "show", explode)
        X, diagram, features = torus
        fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=2)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_saving_writes_the_file_and_prints_nothing(
        self, tmp_path, capsys, square_cloud, square_feature
    ):
        """The constitution bans `print` in library code outside summary()."""
        target = tmp_path / "panels.png"
        diagram = np.array([[1.0, 2.0]])
        fig = plot_matplotlib(
            FakeFit(square_cloud, [square_feature], diagram),
            save_path=str(target),
        )

        assert target.exists() and target.stat().st_size > 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        plt.close(fig)

    def test_bad_save_directory_fails_before_drawing(
        self, tmp_path, square_cloud, square_feature
    ):
        missing = tmp_path / "no_such_dir" / "panels.png"
        before = plt.get_fignums()

        with pytest.raises(ValueError, match="does not exist"):
            plot_matplotlib(
                FakeFit(square_cloud, [square_feature], np.array([[1.0, 2.0]])),
                save_path=str(missing),
            )

        assert plt.get_fignums() == before  # no orphaned figure


class TestInputValidation:

    def test_unfitted_model_is_rejected(self):
        class Unfitted:
            point_cloud_ = None
            features_ = []
            diagrams_ = None

        with pytest.raises(RuntimeError, match="fit()"):
            plot_matplotlib(Unfitted())

    def test_negative_max_cycles_is_rejected(self, square_cloud):
        """A negative slice would silently draw a *different* set of cycles."""
        fit = FakeFit(square_cloud, [], np.array([[1.0, 2.0]]))
        with pytest.raises(ValueError, match="non-negative"):
            plot_matplotlib(fit, max_cycles=-1)

    def test_non_integer_max_cycles_is_rejected(self, square_cloud):
        fit = FakeFit(square_cloud, [], np.array([[1.0, 2.0]]))
        with pytest.raises(TypeError, match="integer"):
            plot_matplotlib(fit, max_cycles=2.5)

    def test_mismatched_projection_is_rejected(
        self, ax, square_cloud, square_feature
    ):
        foreign = project_for_cycle(square_cloud[:8], np.array([0, 1, 2]))
        with pytest.raises(ValueError, match="covers 8 points"):
            draw_cycle_panel(
                ax,
                square_cloud,
                square_feature,
                "#0072B2",
                rank=1,
                projection=foreign,
            )

    def test_supplied_projection_is_used_as_given(self, ax, square_cloud):
        """T13 fits the plane once and shares it; the panel must honour it."""
        D = cdist(square_cloud, square_cloud)
        feature = feature_from_path(D, list(range(12)) + [0])
        shared = project_for_cycle(square_cloud, feature.cycle_vertices)

        draw_cycle_panel(
            ax, square_cloud, feature, "#0072B2", rank=1, projection=shared
        )
        np.testing.assert_allclose(
            offsets(collection(ax, GID_CLOUD)), shared.coords
        )

    def test_feature_pointing_outside_the_cloud_is_rejected(self, ax):
        X = np.zeros((5, 3))
        feature = CycleFeature(
            index=0,
            birth=1.0,
            death=2.0,
            persistence=1.0,
            birth_edge=(0, 99),
            cycle_vertices=np.array([0, 1, 2]),
        )
        with pytest.raises(IndexError, match="99"):
            draw_cycle_panel(ax, X, feature, "#0072B2", rank=1)


class TestBirthEdgeIsDistinguished:

    def test_birth_edge_is_drawn_separately_and_starred(
        self, ax, square_cloud, square_feature
    ):
        draw_cycle_panel(ax, square_cloud, square_feature, "#0072B2", rank=1)

        birth = collection(ax, GID_BIRTH_EDGE)
        assert birth is not None
        assert len(birth.get_segments()) == 1
        assert hexes(birth.get_colors()) == ["#000000"]
        assert len(offsets(collection(ax, GID_BIRTH_VERTICES))) == 2

    def test_loop_edges_exclude_the_birth_edge(
        self, ax, square_cloud, square_feature
    ):
        draw_cycle_panel(ax, square_cloud, square_feature, "#0072B2", rank=1)

        loop = collection(ax, GID_CYCLE)
        assert len(loop.get_segments()) == square_feature.n_edges - 1


# ===========================================================================
# AC15 — the deprecation gate
# ===========================================================================


class TestDeprecationGate:

    def test_no_deprecation_warnings_from_a_full_figure(self, torus):
        X, diagram, features = torus
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", PendingDeprecationWarning)
            fig = plot_matplotlib(FakeFit(X, features, diagram), max_cycles=3)
        plt.close(fig)

    def test_no_deprecation_warnings_from_a_single_panel(
        self, ax, square_cloud, square_feature
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", PendingDeprecationWarning)
            draw_cycle_panel(ax, square_cloud, square_feature, "#0072B2", 1)


# ===========================================================================
# End-to-end: the real model, so wiring drift is caught here and not in a demo
# ===========================================================================


class TestAgainstTheRealModel:

    def test_fitted_three_dimensional_cloud_renders_projected_panels(self):
        from repcycles import RepresentativeCycles

        X = make_tilted_circle()
        rc = RepresentativeCycles().fit(X)
        assert rc.features_ and rc.features_[0].n_edges >= 3  # fixture guard

        fig = plot_matplotlib(rc, max_cycles=2)
        panel = fig.axes[1]

        expected = project_for_cycle(X, rc.features_[0].cycle_vertices)
        np.testing.assert_allclose(
            offsets(collection(panel, GID_CLOUD)), expected.coords, atol=1e-12
        )
        message = text(panel, GID_PROJECTION_NOTE).get_text()
        assert "best-fit plane" in message
        # The circle is exactly planar, so the fitted plane recovers it whole
        # while the naive view sees it foreshortened — V4 in one assertion.
        fitted, naive = percentages(message)
        assert fitted == pytest.approx(1.0, abs=1e-3)
        assert naive < 0.90
        plt.close(fig)
