"""Tests for cycle verification and metadata (spec 001 T6).

Covers F5 (``length`` / ``path``), F6 (``is_verified``), F7 (loud degradation)
and F8 (determinism), and the acceptance criteria AC5 and AC6.

These tests drive :func:`repcycles.reconstruction.reconstruct_cycle` directly
against generators read straight from ripser, deliberately bypassing
``repcycles.core`` — the point is to test reconstruction, not the assembly
around it.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import numpy as np
import pytest
from gph import ripser_parallel
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from repcycles.errors import CycleReconstructionWarning
from repcycles.reconstruction import CycleResult, reconstruct_cycle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

#: Zhu (2013) §2.3 rectangle: birth = 2, death = √5, perimeter = 6.
RECT_POINTS = np.array(
    [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
)
RECT_PERIMETER = 6.0


def _make_circle(n: int = 60, noise: float = 0.05) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    return X + RNG.standard_normal(X.shape) * noise


def _make_figure_eight(n: int = 120, noise: float = 0.05) -> np.ndarray:
    half = n // 2
    theta = np.linspace(0, 2 * np.pi, half, endpoint=False)
    left = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
    right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
    X = np.vstack([left, right])
    return X + RNG.standard_normal(X.shape) * noise


def _make_torus(n: int = 300, R: float = 2.0, r: float = 0.8) -> np.ndarray:
    theta = RNG.uniform(0, 2 * np.pi, n)
    phi = RNG.uniform(0, 2 * np.pi, n)
    return np.column_stack(
        [
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
        ]
    )


def _make_annulus(n: int = 150, r_in: float = 1.0, r_out: float = 2.0) -> np.ndarray:
    theta = RNG.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(RNG.uniform(r_in**2, r_out**2, n))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _two_far_clusters() -> np.ndarray:
    """Two tight triangles ~1000 units apart.

    Vertices 1 and 3 are the closest cross-cluster pair (999 apart); every
    other cross pair is strictly farther. So at radius 999 the edge (1, 3) is
    the *only* link between the clusters — remove it and there is no path,
    which is exactly the disconnection F7 has to report.
    """
    left = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.9]])
    right = left + np.array([1000.0, 0.0])
    return np.vstack([left, right])


#: (birth edge, birth radius) for the unique bridge of ``_two_far_clusters``.
BRIDGE_EDGE = (1, 3)
BRIDGE_RADIUS = 999.0


def _h1_generators(
    X: np.ndarray, thresh: float = np.inf
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], float]]]:
    """Return ``(dist_matrix, [(birth_edge, birth_radius), ...])`` from ripser.

    ``birth_radius`` is taken as the float64 length of the birth edge rather
    than the float32 diagram value — see the note in
    ``reconstruction._reference_radius``.
    """
    D = cdist(X, X)
    result = ripser_parallel(
        X.astype(np.float32),
        maxdim=1,
        thresh=thresh,
        return_generators=True,
    )
    gens = result["gens"]
    rows = []
    if len(gens[1]) > 0:
        rows.extend((int(g[0]), int(g[1])) for g in gens[1][0])
    if len(gens[3]) > 0:
        rows.extend((int(g[0]), int(g[1])) for g in gens[3][0])
    return D, [((u, v), float(D[u, v])) for u, v in rows]


FIXTURES = {
    "circle": (_make_circle(), np.inf),
    "figure_eight": (_make_figure_eight(), np.inf),
    "torus": (_make_torus(), np.inf),
    "annulus": (_make_annulus(), np.inf),
    "circle_truncated": (_make_circle(), 1.0),  # produces essential classes
}


# ---------------------------------------------------------------------------
# AC5 — every feature verified, length == summed edge lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURES))
class TestAcceptanceCriterion5:
    """Circle / figure-eight / torus / annulus: all features must verify."""

    def test_every_feature_is_verified(self, name):
        X, thresh = FIXTURES[name]
        D, generators = _h1_generators(X, thresh)
        assert generators, f"{name} produced no H1 generators to test"

        unverified = [
            edge
            for edge, radius in generators
            if not reconstruct_cycle(D, edge, radius).is_verified
        ]
        assert not unverified, (
            f"{name}: {len(unverified)} of {len(generators)} features failed "
            f"verification, e.g. birth edge {unverified[0]}"
        )

    def test_length_equals_summed_edge_lengths(self, name):
        X, thresh = FIXTURES[name]
        D, generators = _h1_generators(X, thresh)

        for edge, radius in generators:
            result = reconstruct_cycle(D, edge, radius)
            expected = D[result.edges[:, 0], result.edges[:, 1]].sum()
            assert result.length == pytest.approx(expected, abs=1e-9), (
                f"{name}: length {result.length} != summed edges {expected} "
                f"for birth edge {edge}"
            )

    def test_every_edge_is_within_the_birth_radius(self, name):
        """The claim `is_verified` makes, re-measured independently."""
        X, thresh = FIXTURES[name]
        D, generators = _h1_generators(X, thresh)

        for edge, radius in generators:
            result = reconstruct_cycle(D, edge, radius)
            lengths = D[result.edges[:, 0], result.edges[:, 1]]
            assert lengths.max() <= radius * (1 + 1e-9)

    def test_cycles_have_at_least_three_edges(self, name):
        X, thresh = FIXTURES[name]
        D, generators = _h1_generators(X, thresh)

        for edge, radius in generators:
            result = reconstruct_cycle(D, edge, radius)
            assert len(result.edges) >= 3, (
                f"{name}: birth edge {edge} gave a degenerate "
                f"{len(result.edges)}-edge 'cycle'"
            )


# ---------------------------------------------------------------------------
# AC6 — determinism (F8)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """F8: self-consistency, not cross-version identity."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_two_calls_return_byte_identical_edges(self, name):
        X, thresh = FIXTURES[name]
        D, generators = _h1_generators(X, thresh)

        for edge, radius in generators:
            first = reconstruct_cycle(D, edge, radius)
            second = reconstruct_cycle(D, edge, radius)
            assert np.array_equal(first.edges, second.edges)
            assert np.array_equal(first.path, second.path)
            assert first.length == second.length
            assert first.is_verified == second.is_verified

    def test_scipy_dijkstra_predecessors_are_stable_under_ties(self):
        """A graph deliberately full of equal-cost paths still resolves the
        same way every call, so no extra tie-breaking machinery is needed.

        Every edge of this 6-cycle has weight 1, so both arcs between the
        birth-edge endpoints cost exactly the same; only a deterministic
        implementation can return the same one 20 times running.
        """
        n = 6
        D = np.ones((n, n))
        np.fill_diagonal(D, 0.0)

        results = [reconstruct_cycle(D, (0, 3), 1.0) for _ in range(20)]
        for result in results[1:]:
            assert np.array_equal(result.edges, results[0].edges)


# ---------------------------------------------------------------------------
# Zhu §2.3 rectangle — analytically known geometry
# ---------------------------------------------------------------------------


class TestZhuRectangle:
    """Regression test, not a demo (see specs/constitution.md)."""

    @pytest.fixture(scope="class")
    def result(self):
        D, generators = _h1_generators(RECT_POINTS)
        assert len(generators) == 1
        edge, radius = generators[0]
        assert radius == pytest.approx(2.0, abs=1e-9), "birth radius is 2"
        return reconstruct_cycle(D, edge, radius)

    def test_cycle_has_four_edges(self, result):
        assert len(result.edges) == 4

    def test_cycle_length_is_the_perimeter(self, result):
        assert result.length == pytest.approx(RECT_PERIMETER, abs=1e-9)

    def test_cycle_is_verified(self, result):
        assert result.is_verified is True

    def test_path_visits_all_four_corners(self, result):
        assert sorted(result.path[:-1].tolist()) == [0, 1, 2, 3]

    def test_length_matches_hand_computed_edge_sum(self, result):
        """2 + 1 + 2 + 1, independent of which corner the traversal starts at."""
        D = cdist(RECT_POINTS, RECT_POINTS)
        by_hand = sum(
            D[a, b] for a, b in zip(result.path[:-1], result.path[1:])
        )
        assert by_hand == pytest.approx(RECT_PERIMETER, abs=1e-9)


# ---------------------------------------------------------------------------
# cycle_path contract (F5)
# ---------------------------------------------------------------------------


class TestCyclePath:

    @pytest.fixture(scope="class")
    def result(self):
        D, generators = _h1_generators(_make_circle())
        edge, radius = max(generators, key=lambda g: g[1])
        return reconstruct_cycle(D, edge, radius)

    def test_path_closes_on_itself(self, result):
        assert result.path[0] == result.path[-1]

    def test_path_interior_has_no_repeats(self, result):
        interior = result.path[:-1]
        assert len(np.unique(interior)) == len(interior)

    def test_consecutive_path_pairs_are_exactly_the_edges(self, result):
        pairs = np.column_stack([result.path[:-1], result.path[1:]])
        assert np.array_equal(pairs, result.edges)

    def test_path_is_one_longer_than_edges(self, result):
        assert len(result.path) == len(result.edges) + 1

    def test_every_vertex_has_even_degree(self, result):
        _, counts = np.unique(result.edges, return_counts=True)
        assert np.all(counts % 2 == 0)

    def test_vertices_property_matches_unique_edges(self, result):
        assert np.array_equal(result.vertices, np.unique(result.edges))


# ---------------------------------------------------------------------------
# F7 — degraded reconstruction is loud, never silent
# ---------------------------------------------------------------------------


class TestDegradedReconstruction:

    @staticmethod
    def _disconnected_case():
        X = _two_far_clusters()
        D = cdist(X, X)
        assert D[BRIDGE_EDGE] == pytest.approx(BRIDGE_RADIUS, abs=1e-9)
        return D, BRIDGE_EDGE, BRIDGE_RADIUS

    def test_warns_with_the_dedicated_category(self):
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning):
            reconstruct_cycle(D, edge, radius)

    def test_warning_message_explains_the_disconnection(self):
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning) as record:
            reconstruct_cycle(D, edge, radius)
        message = str(record[0].message)
        assert "connected components" in message
        assert "is_verified=False" in message
        assert "vertices 1 and 3" in message

    def test_result_is_flagged_unverified(self):
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning):
            result = reconstruct_cycle(D, edge, radius)
        assert result.is_verified is False

    def test_fallback_is_the_birth_edge_alone(self):
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning):
            result = reconstruct_cycle(D, edge, radius)
        assert np.array_equal(result.edges, np.array([[1, 3]]))
        assert result.length == pytest.approx(BRIDGE_RADIUS, abs=1e-9)

    def test_fallback_path_still_closes_for_plotting(self):
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning):
            result = reconstruct_cycle(D, edge, radius)
        assert np.array_equal(result.path, np.array([1, 3, 1]))
        assert result.path[0] == result.path[-1]

    def test_birth_edge_is_the_only_edge_in_the_graph(self):
        """Two points and nothing else: removing the birth edge empties the
        graph entirely, the most degenerate possible input."""
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.warns(CycleReconstructionWarning):
            result = reconstruct_cycle(D, (0, 1), 1.0)
        assert np.array_equal(result.edges, np.array([[0, 1]]))
        assert result.length == pytest.approx(1.0, abs=1e-9)
        assert result.is_verified is False

    def test_no_print_in_the_degraded_path(self, capsys):
        """The constitution forbids reporting degradation via stdout."""
        D, edge, radius = self._disconnected_case()
        with pytest.warns(CycleReconstructionWarning):
            reconstruct_cycle(D, edge, radius)
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_healthy_reconstruction_does_not_warn(self):
        """The warning must be a signal, not background noise."""
        D, generators = _h1_generators(RECT_POINTS)
        edge, radius = generators[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reconstruct_cycle(D, edge, radius)
        assert not [
            w
            for w in caught
            if issubclass(w.category, CycleReconstructionWarning)
        ]


# ---------------------------------------------------------------------------
# graph_cache slot (kept working for T7)
# ---------------------------------------------------------------------------


class _StubGraphCache:
    """Minimal stand-in for T7's ``RipsGraphCache``.

    Returns the FULL Rips graph at a radius, birth edge included — a cache is
    shared across features, so excluding a per-feature edge is the callee's
    job.  This test pins that contract before T7 implements against it.
    """

    def __init__(self, dist_matrix: np.ndarray):
        self._D = dist_matrix

    def graph_at(self, radius: float) -> csr_matrix:
        mask = self._D <= radius
        np.fill_diagonal(mask, False)
        rows, cols = np.nonzero(mask)
        return csr_matrix(
            (self._D[rows, cols], (rows, cols)), shape=self._D.shape
        )


class TestGraphCacheSlot:

    def test_default_none_builds_the_graph_per_call(self):
        D, generators = _h1_generators(RECT_POINTS)
        edge, radius = generators[0]
        assert reconstruct_cycle(D, edge, radius).is_verified is True

    def test_cached_graph_gives_the_same_geodesic_length(self):
        X = _make_circle()
        D, generators = _h1_generators(X)
        cache = _StubGraphCache(D)

        for edge, radius in generators:
            plain = reconstruct_cycle(D, edge, radius)
            cached = reconstruct_cycle(D, edge, radius, graph_cache=cache)
            assert cached.length == pytest.approx(plain.length, abs=1e-9)
            assert cached.is_verified == plain.is_verified

    def test_cached_graph_is_left_untouched_afterwards(self):
        """Masking the birth edge must not corrupt a graph shared by every
        other feature — the whole point of caching it."""
        D, generators = _h1_generators(RECT_POINTS)
        cache = _StubGraphCache(D)
        edge, radius = generators[0]

        graph = cache.graph_at(radius)
        before = graph.data.copy()

        class _FixedCache:
            def graph_at(self, _radius):
                return graph

        reconstruct_cycle(D, edge, radius, graph_cache=_FixedCache())
        assert np.array_equal(graph.data, before)

    def test_cached_graph_reports_disconnection_too(self):
        X = _two_far_clusters()
        D = cdist(X, X)
        cache = _StubGraphCache(D)
        with pytest.warns(CycleReconstructionWarning):
            result = reconstruct_cycle(D, BRIDGE_EDGE, BRIDGE_RADIUS, cache)
        assert result.is_verified is False


# ---------------------------------------------------------------------------
# Boundary guards
# ---------------------------------------------------------------------------


class TestBirthEdgeValidation:

    def test_self_loop_is_rejected(self):
        D = cdist(RECT_POINTS, RECT_POINTS)
        with pytest.raises(ValueError, match="distinct vertices"):
            reconstruct_cycle(D, (2, 2), 1.0)

    def test_out_of_range_vertex_is_rejected(self):
        D = cdist(RECT_POINTS, RECT_POINTS)
        with pytest.raises(ValueError, match="out of range"):
            reconstruct_cycle(D, (0, 99), 1.0)

    def test_negative_vertex_is_rejected(self):
        D = cdist(RECT_POINTS, RECT_POINTS)
        with pytest.raises(ValueError, match="out of range"):
            reconstruct_cycle(D, (-1, 2), 1.0)


class TestReturnType:

    def test_returns_a_cycle_result(self):
        D, generators = _h1_generators(RECT_POINTS)
        result = reconstruct_cycle(D, *generators[0])
        assert isinstance(result, CycleResult)
        assert result.edges.dtype.kind == "i"
        assert result.path.dtype.kind == "i"
        assert isinstance(result.length, float)
        assert isinstance(result.is_verified, bool)

    def test_edges_are_two_columns(self):
        D, generators = _h1_generators(RECT_POINTS)
        result = reconstruct_cycle(D, *generators[0])
        assert result.edges.ndim == 2 and result.edges.shape[1] == 2

    def test_length_is_positive_for_a_real_cycle(self):
        D, generators = _h1_generators(RECT_POINTS)
        assert reconstruct_cycle(D, *generators[0]).length > 0.0
        assert math.isfinite(reconstruct_cycle(D, *generators[0]).length)
