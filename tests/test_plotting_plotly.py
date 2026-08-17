"""Tests for the interactive Plotly view (spec 001 T14 — V7, V9, V2/V3).

Plotly figures are inspectable data structures, so nothing here renders an
image: every assertion reads ``fig.data`` traces and ``fig.layout``.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import to_hex

matplotlib.use("Agg")

from repcycles import RepresentativeCycles  # noqa: E402
from repcycles.feature import CycleFeature  # noqa: E402
from repcycles.palette import cycle_colors  # noqa: E402
from repcycles.plotting.interactive import (  # noqa: E402
    SKELETON_TRACE_NAME,
    plot_plotly,
)
from repcycles.reconstruction import reconstruct_cycle  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------


def circle(n: int = 40, radius: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def torus(n_major: int = 12, n_minor: int = 10, R: float = 3.0, r: float = 1.0):
    """A 3-D torus on a regular grid — deterministic, 120 points by default."""
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    return np.column_stack(
        (
            (R + r * np.cos(vv)) * np.cos(uu),
            (R + r * np.cos(vv)) * np.sin(uu),
            r * np.sin(vv),
        )
    )


def fit_with_cycles(X: np.ndarray, **kwargs) -> RepresentativeCycles:
    """Fit for real, then attach T6's reconstructed cycles to each feature.

    ``core.fit()`` does not yet unpack T6's ``CycleResult`` into
    ``CycleFeature`` — that rewire belongs to ``repcycles/core.py``, which T14
    does not own.  Composing the two steps here keeps these tests running
    against real ripser output and the real reconstruction, without depending
    on the pending integration.
    """
    rc = RepresentativeCycles(reconstruct_cycles=False, **kwargs)
    rc.fit(X)
    for feature in rc.features_:
        result = reconstruct_cycle(
            rc._dist_matrix_, feature.birth_edge, feature.birth
        )
        feature.cycle_edges = result.edges
        feature.cycle_vertices = result.vertices
        feature.cycle_path = result.path
        feature.cycle_length = result.length
        feature.is_verified = result.is_verified
    return rc


def stub_rc(X: np.ndarray, features, dist_matrix=None) -> RepresentativeCycles:
    """A real ``RepresentativeCycles`` with its fitted state set by hand.

    Used where a test needs exact control over the feature records (an
    essential bar, a specific feature count) rather than whatever ripser
    happens to produce.
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=float)
    rc = RepresentativeCycles()
    rc.point_cloud_ = X
    rc.features_ = list(features)
    rc._dist_matrix_ = cdist(X, X) if dist_matrix is None else dist_matrix
    return rc


def square_feature(index: int = 0, **overrides) -> CycleFeature:
    """A 4-edge loop on vertices 0-3, with plausible metadata."""
    kwargs = dict(
        index=index,
        birth=1.0,
        death=1.5,
        persistence=0.5,
        birth_edge=(0, 1),
        death_edge=(1, 2),
        cycle_edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
        cycle_vertices=np.array([0, 1, 2, 3]),
        cycle_length=4.0,
        cycle_path=np.array([0, 1, 2, 3, 0]),
        is_verified=True,
    )
    kwargs.update(overrides)
    return CycleFeature(**kwargs)


def cycle_traces(fig, rank: int):
    return [t for t in fig.data if getattr(t, "legendgroup", None) == f"cycle{rank}"]


def skeleton_traces(fig):
    return [
        t
        for t in fig.data
        if (t.name or "").startswith(SKELETON_TRACE_NAME)
    ]


def segment_lengths(trace, n_axes: int) -> np.ndarray:
    """Recover the length of each drawn line segment from a plotly trace."""
    axes = [np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float)]
    if n_axes == 3:
        axes.append(np.asarray(trace.z, dtype=float))
    deltas = np.column_stack([a[0::3] - a[1::3] for a in axes])
    return np.linalg.norm(deltas, axis=1)


@pytest.fixture(scope="module")
def rc_circle():
    return fit_with_cycles(circle(40))


@pytest.fixture(scope="module")
def rc_torus():
    return fit_with_cycles(torus())


# ----------------------------------------------------------------------
# AC12 (plotly half): a figure comes back for 2-D and 3-D, with the skeleton
# ----------------------------------------------------------------------


class TestReturnsFigure:

    def test_2d_with_skeleton_returns_figure(self, rc_circle):
        import plotly.graph_objects as go

        fig = plot_plotly(rc_circle, show_skeleton=True)
        assert isinstance(fig, go.Figure)
        assert len(skeleton_traces(fig)) == 1

    def test_3d_with_skeleton_returns_figure(self, rc_torus):
        import plotly.graph_objects as go

        fig = plot_plotly(rc_torus, show_skeleton=True)
        assert isinstance(fig, go.Figure)
        (skeleton,) = skeleton_traces(fig)
        assert isinstance(skeleton, go.Scatter3d)
        assert np.asarray(skeleton.z).size > 0

    def test_2d_uses_scatter_not_scatter3d(self, rc_circle):
        import plotly.graph_objects as go

        fig = plot_plotly(rc_circle, show_skeleton=True)
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_unfitted_model_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="fit"):
            plot_plotly(RepresentativeCycles())


# ----------------------------------------------------------------------
# V7 — hover metadata
# ----------------------------------------------------------------------


HOVER_FIELDS = (
    "H₁ feature #",
    "birth:",
    "death:",
    "persistence:",
    "cycle length:",
    "edges:",
    "verified:",
)


class TestHoverMetadata:

    def test_every_cycle_trace_carries_all_seven_fields(self, rc_circle):
        fig = plot_plotly(rc_circle)
        traces = [t for t in fig.data if getattr(t, "legendgroup", None)]
        assert traces, "expected at least one cycle trace"
        for trace in traces:
            hover = trace.hovertext
            assert isinstance(hover, str)
            for field in HOVER_FIELDS:
                assert field in hover, f"{field!r} missing from {hover!r}"

    def test_hover_reports_the_feature_index_and_values(self):
        feature = square_feature(index=7, birth=0.25, death=1.75, persistence=1.5)
        fig = plot_plotly(stub_rc(circle(4), [feature]))
        hover = cycle_traces(fig, 0)[0].hovertext
        assert "H₁ feature #7" in hover
        assert "birth: 0.2500" in hover
        assert "death: 1.7500" in hover
        assert "persistence: 1.5000" in hover
        assert "cycle length: 4.0000" in hover
        assert "edges: 4" in hover
        assert "verified: yes" in hover

    def test_essential_feature_renders_infinity(self):
        feature = square_feature(
            death=np.inf, persistence=np.inf, is_essential=True
        )
        fig = plot_plotly(stub_rc(circle(4), [feature]))
        hover = cycle_traces(fig, 0)[0].hovertext
        assert "death: ∞" in hover
        assert "persistence: ∞" in hover

    def test_unverified_feature_says_so(self):
        feature = square_feature(is_verified=False)
        fig = plot_plotly(stub_rc(circle(4), [feature]))
        assert "verified: no" in cycle_traces(fig, 0)[0].hovertext

    def test_3d_cycle_traces_carry_hover(self, rc_torus):
        fig = plot_plotly(rc_torus, max_cycles=2)
        traces = [t for t in fig.data if getattr(t, "legendgroup", None)]
        assert traces
        assert all("cycle length:" in t.hovertext for t in traces)


# ----------------------------------------------------------------------
# AC9 (plotly half) — shared colours
# ----------------------------------------------------------------------


class TestSharedColours:

    def test_trace_colours_match_cycle_colors(self):
        features = [square_feature(index=k) for k in range(3)]
        fig = plot_plotly(stub_rc(circle(4), features))
        expected = [to_hex(c) for c in cycle_colors(len(features))]

        for k, want in enumerate(expected):
            traces = cycle_traces(fig, k)
            assert traces
            for trace in traces:
                if trace.name == "birth edge":
                    assert to_hex(trace.line.color) == to_hex("black")
                elif trace.mode == "lines":
                    assert to_hex(trace.line.color) == want
                else:
                    assert to_hex(trace.marker.color) == want

    def test_palette_repeats_beyond_its_length(self):
        n = 8  # Okabe-Ito has 6 entries
        features = [square_feature(index=k) for k in range(n)]
        fig = plot_plotly(stub_rc(circle(4), features), max_cycles=n)
        expected = [to_hex(c) for c in cycle_colors(n)]
        got = [
            to_hex(cycle_traces(fig, k)[-1].marker.color) for k in range(n)
        ]
        assert got == expected

    def test_does_not_use_the_plotly_qualitative_palette(self):
        from plotly.colors import qualitative

        fig = plot_plotly(stub_rc(circle(4), [square_feature()]))
        marker = to_hex(cycle_traces(fig, 0)[-1].marker.color)
        assert marker != to_hex(qualitative.Plotly[0])
        assert marker == to_hex(cycle_colors(1)[0])


# ----------------------------------------------------------------------
# V9 — Rips 1-skeleton underlay and its edge budget
# ----------------------------------------------------------------------


class TestSkeleton:

    def test_off_by_default(self, rc_circle):
        fig = plot_plotly(rc_circle)
        assert skeleton_traces(fig) == []

    def test_skeleton_is_drawn_beneath_everything_else(self, rc_circle):
        fig = plot_plotly(rc_circle, show_skeleton=True)
        assert (fig.data[0].name or "").startswith(SKELETON_TRACE_NAME)

    def test_skeleton_uses_the_first_cycles_birth_radius(self, rc_circle):
        radius = rc_circle.features_[0].birth
        fig = plot_plotly(rc_circle, show_skeleton=True)
        (skeleton,) = skeleton_traces(fig)
        assert f"r={radius:.3f}" in skeleton.name
        assert np.all(segment_lengths(skeleton, 2) <= radius + 1e-9)

    def test_untruncated_skeleton_draws_every_edge_and_adds_no_note(
        self, rc_circle
    ):
        radius = rc_circle.features_[0].birth
        D = rc_circle._dist_matrix_
        rows, cols = np.triu_indices(D.shape[0], k=1)
        n_total = int(np.count_nonzero(D[rows, cols] <= radius))

        fig = plot_plotly(rc_circle, show_skeleton=True)
        (skeleton,) = skeleton_traces(fig)
        assert np.asarray(skeleton.x).size == 3 * n_total
        assert "truncated" not in fig.layout.title.text

    def test_truncation_draws_exactly_the_shortest_n_edges(self, rc_circle):
        budget = 10
        radius = rc_circle.features_[0].birth
        D = rc_circle._dist_matrix_
        rows, cols = np.triu_indices(D.shape[0], k=1)
        lengths = D[rows, cols]
        present = np.sort(lengths[lengths <= radius])
        assert present.size > budget, "fixture must exceed the budget"

        fig = plot_plotly(
            rc_circle, show_skeleton=True, skeleton_max_edges=budget
        )
        (skeleton,) = skeleton_traces(fig)
        drawn = np.sort(segment_lengths(skeleton, 2))
        assert drawn.size == budget
        assert np.allclose(drawn, present[:budget], atol=1e-9)

    def test_truncation_annotates_the_title(self, rc_circle):
        budget = 10
        radius = rc_circle.features_[0].birth
        D = rc_circle._dist_matrix_
        rows, cols = np.triu_indices(D.shape[0], k=1)
        n_total = int(np.count_nonzero(D[rows, cols] <= radius))

        fig = plot_plotly(
            rc_circle, show_skeleton=True, skeleton_max_edges=budget
        )
        assert (
            f"skeleton truncated: {budget} / {n_total} edges"
            in fig.layout.title.text
        )

    def test_truncation_is_deterministic(self, rc_circle):
        kwargs = dict(show_skeleton=True, skeleton_max_edges=17)
        a = skeleton_traces(plot_plotly(rc_circle, **kwargs))[0]
        b = skeleton_traces(plot_plotly(rc_circle, **kwargs))[0]
        assert np.array_equal(
            np.asarray(a.x, dtype=float), np.asarray(b.x, dtype=float),
            equal_nan=True,
        )

    def test_truncation_in_3d(self, rc_torus):
        fig = plot_plotly(rc_torus, show_skeleton=True, skeleton_max_edges=5)
        (skeleton,) = skeleton_traces(fig)
        assert np.asarray(skeleton.x).size == 15
        assert "skeleton truncated: 5 /" in fig.layout.title.text

    def test_infinite_distances_are_treated_as_absent_edges(self):
        """`+inf` in a precomputed matrix means 'no edge' (F10)."""
        X = circle(4)
        D = np.array(
            [
                [0.0, 1.0, 1.0, np.inf],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [np.inf, 1.0, 1.0, 0.0],
            ]
        )
        feature = square_feature(birth=2.0)
        fig = plot_plotly(
            stub_rc(X, [feature], dist_matrix=D), show_skeleton=True
        )
        (skeleton,) = skeleton_traces(fig)
        # 6 pairs, one of them infinite -> 5 edges, 3 coordinates each.
        assert np.asarray(skeleton.x).size == 15

    def test_zero_budget_is_rejected(self, rc_circle):
        with pytest.raises(ValueError, match="skeleton_max_edges"):
            plot_plotly(rc_circle, skeleton_max_edges=0)

    def test_missing_distance_matrix_raises(self):
        rc = stub_rc(circle(4), [square_feature()])
        rc._dist_matrix_ = None
        with pytest.raises(RuntimeError, match="_dist_matrix_"):
            plot_plotly(rc, show_skeleton=True)


# ----------------------------------------------------------------------
# Empty and edge-case fits
# ----------------------------------------------------------------------


class TestEmptyState:

    def test_zero_features_returns_a_figure(self):
        import plotly.graph_objects as go

        fig = plot_plotly(stub_rc(circle(6), []))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # the point cloud alone

    def test_zero_features_with_skeleton_says_why_none_is_drawn(self):
        """No focused cycle means no birth radius; that is stated, not hidden."""
        fig = plot_plotly(stub_rc(circle(6), []), show_skeleton=True)
        assert skeleton_traces(fig) == []
        assert "skeleton not drawn" in fig.layout.title.text

    def test_feature_without_reconstructed_edges_still_plots(self):
        feature = CycleFeature(
            index=0,
            birth=1.0,
            death=2.0,
            persistence=1.0,
            birth_edge=(0, 1),
            cycle_vertices=np.array([0, 1]),
        )
        fig = plot_plotly(stub_rc(circle(4), [feature]))
        traces = cycle_traces(fig, 0)
        assert len(traces) == 1  # vertices only, no edge traces
        assert "edges: 0" in traces[0].hovertext

    def test_max_cycles_limits_the_legend_groups(self, rc_torus):
        fig = plot_plotly(rc_torus, max_cycles=2)
        groups = {t.legendgroup for t in fig.data if t.legendgroup}
        assert groups == {"cycle0", "cycle1"}


# ----------------------------------------------------------------------
# Output and library hygiene
# ----------------------------------------------------------------------


class TestOutput:

    def test_save_html_writes_the_file_without_printing(
        self, rc_circle, tmp_path, capsys
    ):
        out = tmp_path / "cycles.html"
        plot_plotly(rc_circle, save_html=str(out))
        assert out.exists() and out.stat().st_size > 0
        assert capsys.readouterr().out == ""

    def test_title_is_preserved_when_nothing_is_truncated(self, rc_circle):
        fig = plot_plotly(rc_circle, title="My Loops")
        assert fig.layout.title.text == "My Loops"

    def test_birth_edge_is_drawn_black_and_dashed(self, rc_circle):
        fig = plot_plotly(rc_circle, max_cycles=1)
        birth = [t for t in cycle_traces(fig, 0) if t.name == "birth edge"]
        assert len(birth) == 1
        assert birth[0].line.dash == "dash"
        assert to_hex(birth[0].line.color) == to_hex("black")
