"""Tests for the best-fit-plane projection helper (spec 001 T9 — V4 / AC8).

Two properties are under test and they are of different kinds:

* **Relative (an invariant).**  ``variance_retained >= baseline_retained`` for
  every cycle on every ≥3-D cloud.  This follows from the optimality of the
  truncated SVD, so it cannot flake and is asserted broadly — on the torus, on
  the sphere, and on randomly generated clouds and vertex sets.
* **Absolute (a benchmark).**  The top 6 cycles of the torus fixture each
  retain ≥ 0.90.  That is an empirical property of *this* fixture, so it is
  asserted only there.

The fixtures and the H₁ cycle extractor below are deliberately self-contained.
``repcycles.core`` and ``repcycles.reconstruction`` are owned by other tasks
in this wave and are mid-change; the projection contract must be testable
without them, and these tests must not go red for someone else's edit.
"""

from __future__ import annotations

import numpy as np
import pytest
from gph import ripser_parallel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from repcycles.projection import (
    DEGENERACY_RATIO,
    Projection,
    project_for_cycle,
)

# Tolerance for the SVD-optimality invariant.  The two fractions are computed
# by different arithmetic, so an exactly-planar cycle can put them a few ulps
# apart; anything larger than this would be a real violation.
OPTIMALITY_SLACK = 1e-12


# ---------------------------------------------------------------------------
# Fixtures: point clouds (generators mirror examples.py, seeded for repeatability)
# ---------------------------------------------------------------------------


def make_torus(
    n: int = 600,
    R: float = 2.0,
    r: float = 0.5,
    noise: float = 0.05,
    seed: int = 8,
) -> np.ndarray:
    """Random sample from a torus in R³ — the AC8 benchmark cloud."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X = np.column_stack([x, y, z])
    return X + rng.standard_normal(X.shape) * noise


def make_sphere(n: int = 300, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """Random sample from S² in R³.  Its H₁ loops are genuinely 3-D — they
    wrap a curved surface — so they stress the invariant harder than the
    torus, whose tube-loops are nearly planar."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts + rng.standard_normal(pts.shape) * noise


def make_tilted_circle(n: int = 60, seed: int = 3) -> np.ndarray:
    """A planar circle embedded in R³ on a tilted plane.

    Exactly planar, so the best-fit plane must retain ~1.0 while ``X[:, :2]``
    (which sees the circle edge-on-ish) retains noticeably less.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    flat = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    tilt = _rotation(np.deg2rad(55), axis=0) @ _rotation(np.deg2rad(35), axis=2)
    circle = flat @ tilt.T
    context = rng.standard_normal((20, 3)) * 0.3
    return np.vstack([circle, context])


def _rotation(angle: float, axis: int) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i] = R[j, j] = c
    R[i, j], R[j, i] = -s, s
    return R


# ---------------------------------------------------------------------------
# Fixture: representative H1 cycles, extracted without repcycles.core
# ---------------------------------------------------------------------------


def top_h1_cycle_vertices(X: np.ndarray, k: int) -> list:
    """Vertex sets of the *k* most persistent H₁ cycles of *X*.

    Uses the same method the library does — ripser's birth edge, then the
    shortest path around the loop in the Rips graph at the birth radius — but
    implemented here so these tests depend only on ripser and scipy.
    """
    D = cdist(X, X)
    result = ripser_parallel(
        X.astype(np.float32), maxdim=1, return_generators=True
    )
    diagram = result["dgms"][1]
    generators = result["gens"][1][0]

    persistence = diagram[:, 1] - diagram[:, 0]
    ranked = np.argsort(-persistence)[:k]

    cycles = []
    for row in ranked:
        u, v = int(generators[row][0]), int(generators[row][1])
        cycles.append(_loop_around(D, u, v))
    return cycles


def _loop_around(D: np.ndarray, u: int, v: int) -> np.ndarray:
    """Vertices of the shortest u→v path at radius ``D[u, v]``, birth edge
    excluded, closed by that edge."""
    adjacency = D <= D[u, v]
    adjacency[u, v] = adjacency[v, u] = False
    np.fill_diagonal(adjacency, False)

    rows, cols = np.nonzero(adjacency)
    graph = csr_matrix((D[rows, cols], (rows, cols)), shape=D.shape)
    distances, predecessors = dijkstra(
        graph, directed=False, indices=u, return_predecessors=True
    )
    if np.isinf(distances[v]):
        return np.array([u, v])

    path, current = [], v
    while current != u and current >= 0:
        path.append(current)
        current = int(predecessors[current])
    path.append(u)
    return np.unique(path)


@pytest.fixture(scope="module")
def torus():
    X = make_torus()
    return X, top_h1_cycle_vertices(X, 12)


@pytest.fixture(scope="module")
def sphere():
    X = make_sphere()
    return X, top_h1_cycle_vertices(X, 12)


# ---------------------------------------------------------------------------
# AC8 absolute: the torus benchmark
# ---------------------------------------------------------------------------


class TestTorusBenchmark:
    """AC8, absolute half — a benchmark on this specific fixture."""

    def test_top_six_cycles_each_retain_at_least_90_percent(self, torus):
        X, cycles = torus
        retained = [
            project_for_cycle(X, c).variance_retained for c in cycles[:6]
        ]
        assert min(retained) >= 0.90, (
            "AC8 absolute: best-fit plane retention per cycle = "
            f"{[round(v, 4) for v in retained]}"
        )

    def test_beats_the_naive_projection_it_replaces(self, torus):
        """The point of V4: X[:, :2] mangles tube-loops, the fitted plane
        does not.  Asserted as a margin, not just an ordering."""
        X, cycles = torus
        projections = [project_for_cycle(X, c) for c in cycles[:6]]
        worst_baseline = min(p.baseline_retained for p in projections)

        assert worst_baseline < 0.70, (
            "fixture no longer exhibits the failure it is meant to guard: "
            f"worst X[:, :2] retention = {worst_baseline:.4f}"
        )
        assert all(
            p.variance_retained > p.baseline_retained + 0.10
            for p in projections
            if p.baseline_retained < 0.90
        )

    def test_no_top_cycle_falls_back(self, torus):
        X, cycles = torus
        assert not any(
            project_for_cycle(X, c).is_degenerate for c in cycles[:6]
        )


# ---------------------------------------------------------------------------
# AC8 relative: the SVD-optimality invariant
# ---------------------------------------------------------------------------


class TestOptimalityInvariant:
    """AC8, relative half.  Guaranteed by Eckart-Young, so it holds for every
    cycle of every ≥3-D cloud — including degenerate ones, where both figures
    collapse to the same fallback value."""

    def test_holds_for_every_torus_cycle(self, torus):
        X, cycles = torus
        for c in cycles:
            p = project_for_cycle(X, c)
            assert p.variance_retained >= p.baseline_retained - OPTIMALITY_SLACK

    def test_holds_for_every_sphere_cycle(self, sphere):
        X, cycles = sphere
        for c in cycles:
            p = project_for_cycle(X, c)
            assert p.variance_retained >= p.baseline_retained - OPTIMALITY_SLACK

    def test_sphere_loops_are_genuinely_three_dimensional(self, sphere):
        """Guards the fixture itself: if the sphere's loops were near-planar
        in the first two coordinates, the test above would prove nothing."""
        X, cycles = sphere
        baselines = [project_for_cycle(X, c).baseline_retained for c in cycles]
        assert min(baselines) < 0.70

    @pytest.mark.parametrize("seed", range(12))
    def test_holds_for_random_clouds_and_random_vertex_sets(self, seed):
        rng = np.random.default_rng(seed)
        n_dims = int(rng.integers(3, 6))
        X = rng.standard_normal((40, n_dims)) * rng.uniform(0.1, 5.0, n_dims)
        size = int(rng.integers(1, 15))
        vertices = rng.choice(len(X), size=size, replace=False)

        p = project_for_cycle(X, vertices)
        assert p.variance_retained >= p.baseline_retained - OPTIMALITY_SLACK
        assert 0.0 <= p.variance_retained <= 1.0
        assert 0.0 <= p.baseline_retained <= 1.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestPlanarCycle:

    def test_tilted_planar_circle_is_recovered_almost_exactly(self):
        X = make_tilted_circle()
        p = project_for_cycle(X, np.arange(60))

        assert p.variance_retained == pytest.approx(1.0, abs=1e-9)
        assert not p.is_degenerate
        assert p.baseline_retained < 0.95  # the naive view sees it tilted

    def test_projected_circle_is_still_a_circle(self):
        """The plane is an isometry on points lying in it, so the tilted
        circle must come back round, not elliptical."""
        X = make_tilted_circle()
        coords = project_for_cycle(X, np.arange(60)).coords[:60]
        radii = np.linalg.norm(coords, axis=1)
        assert radii.max() - radii.min() < 1e-9

    def test_three_vertex_cycle_spans_its_plane_exactly(self):
        X = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        p = project_for_cycle(X, np.array([0, 1, 2]))

        assert not p.is_degenerate
        assert p.variance_retained == pytest.approx(1.0, abs=1e-9)
        assert len(p.coords) == 4


class TestContextIsPreserved:
    """The plane is fitted on the cycle but applied to the whole cloud —
    that is what keeps the loop in context instead of floating alone."""

    def test_coords_has_one_row_per_point_of_the_original_cloud(self, torus):
        X, cycles = torus
        p = project_for_cycle(X, cycles[0])
        assert p.coords.shape == (len(X), 2)
        assert len(cycles[0]) < len(X)  # the cycle really is a small subset

    def test_row_order_is_the_clouds_own(self):
        X = make_tilted_circle()
        cycle = np.array([0, 5, 10, 15, 20])
        p = project_for_cycle(X, cycle)
        reordered = project_for_cycle(X[::-1], len(X) - 1 - cycle)
        np.testing.assert_allclose(p.coords, reordered.coords[::-1], atol=1e-9)

    def test_distances_within_the_plane_survive_the_projection(self, torus):
        """A projection can only shorten distances, never lengthen them."""
        X, cycles = torus
        p = project_for_cycle(X, cycles[0])
        original = cdist(X[:80], X[:80])
        projected = cdist(p.coords[:80], p.coords[:80])
        assert (projected <= original + 1e-9).all()


# ---------------------------------------------------------------------------
# Degenerate inputs: never raise, never overstate
# ---------------------------------------------------------------------------


class TestDegenerateCycles:

    @staticmethod
    def _collinear_cloud() -> np.ndarray:
        """Points on the line x = y = z, plus off-line context."""
        t = np.linspace(-1.0, 1.0, 9)
        line = np.column_stack([t, t, t])
        return np.vstack([line, [[0.5, -0.5, 0.25], [-0.3, 0.8, -0.1]]])

    def test_collinear_cycle_falls_back_without_raising(self):
        X = self._collinear_cloud()
        p = project_for_cycle(X, np.arange(9))

        assert p.is_degenerate
        assert p.coords.shape == (len(X), 2)
        np.testing.assert_allclose(p.coords, X[:, :2])

    def test_collinear_cycle_reports_variance_honestly(self):
        """The line x=y=z keeps exactly 2/3 of its variance under X[:, :2].
        Claiming 1.0 here would overstate the result — forbidden."""
        X = self._collinear_cloud()
        p = project_for_cycle(X, np.arange(9))

        assert p.variance_retained == pytest.approx(2.0 / 3.0, rel=1e-9)
        assert p.variance_retained == pytest.approx(p.baseline_retained)
        assert p.variance_retained < 1.0

    def test_near_collinear_cycle_is_also_treated_as_degenerate(self):
        X = self._collinear_cloud()
        X[:9, 2] += np.linspace(0, DEGENERACY_RATIO * 1e-3, 9)
        assert project_for_cycle(X, np.arange(9)).is_degenerate

    @pytest.mark.parametrize("n_vertices", [0, 1, 2])
    def test_too_few_vertices_falls_back(self, n_vertices):
        X = make_tilted_circle()
        p = project_for_cycle(X, np.arange(n_vertices))

        assert p.is_degenerate
        assert p.coords.shape == (len(X), 2)
        assert np.isfinite(p.coords).all()
        assert np.isfinite(p.variance_retained)

    def test_identical_vertices_have_no_variance_to_lose(self):
        X = np.array([[1.0, 2.0, 3.0]] * 4 + [[0.0, 0.0, 0.0]])
        p = project_for_cycle(X, np.arange(4))

        assert p.is_degenerate
        assert p.variance_retained == 1.0
        assert p.baseline_retained == 1.0

    def test_two_vertex_cycle_reports_its_true_loss(self):
        """Two points on the z-axis: X[:, :2] collapses them to one dot and
        keeps none of their variance.  The report must say so."""
        X = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        p = project_for_cycle(X, np.array([0, 1]))

        assert p.is_degenerate
        assert p.variance_retained == pytest.approx(0.0, abs=1e-12)


class TestLowDimensionalInput:

    def test_two_dimensional_input_passes_through(self):
        X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        p = project_for_cycle(X, np.array([0, 1, 2]))

        np.testing.assert_array_equal(p.coords, X)
        assert p.variance_retained == 1.0
        assert p.baseline_retained == 1.0
        assert not p.is_degenerate

    def test_two_dimensional_pass_through_copies_rather_than_aliases(self):
        X = np.zeros((4, 2))
        p = project_for_cycle(X, np.array([0, 1, 2]))
        p.coords[0, 0] = 99.0
        assert X[0, 0] == 0.0

    def test_one_dimensional_input_is_padded_and_flagged(self):
        X = np.array([[0.0], [1.0], [2.0]])
        p = project_for_cycle(X, np.array([0, 1, 2]))

        assert p.coords.shape == (3, 2)
        np.testing.assert_array_equal(p.coords[:, 0], X[:, 0])
        np.testing.assert_array_equal(p.coords[:, 1], np.zeros(3))
        assert p.is_degenerate


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_repeated_calls_are_bit_identical(self, torus):
        X, cycles = torus
        first = project_for_cycle(X, cycles[0])
        second = project_for_cycle(X, cycles[0])

        assert np.array_equal(first.coords, second.coords)
        assert first.variance_retained == second.variance_retained
        assert first.baseline_retained == second.baseline_retained

    def test_vertex_order_and_duplicates_do_not_change_the_plane(self, torus):
        """A caller passing a closed ``cycle_path`` (first vertex repeated)
        must get the same view as one passing ``cycle_vertices``."""
        X, cycles = torus
        cycle = cycles[0]
        shuffled = np.concatenate([cycle[::-1], cycle[:1]])

        assert np.array_equal(
            project_for_cycle(X, cycle).coords,
            project_for_cycle(X, shuffled).coords,
        )

    def test_basis_signs_follow_the_documented_convention(self):
        """An ellipse in the x-z plane has unambiguous principal directions
        (x then z).  The sign convention — largest component positive — pins
        the arbitrary SVD sign, so the output must be (x, z), not (-x, -z).
        """
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        X = np.column_stack(
            [2.0 * np.cos(theta), np.zeros(40), 1.0 * np.sin(theta)]
        )
        coords = project_for_cycle(X, np.arange(40)).coords

        np.testing.assert_allclose(coords[:, 0], X[:, 0], atol=1e-9)
        np.testing.assert_allclose(coords[:, 1], X[:, 2], atol=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:

    def test_nan_coordinate_is_rejected_with_its_position(self):
        X = make_tilted_circle()
        X[7, 2] = np.nan
        with pytest.raises(ValueError, match="row 7, column 2"):
            project_for_cycle(X, np.arange(10))

    def test_infinite_coordinate_is_rejected(self):
        X = make_tilted_circle()
        X[3, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            project_for_cycle(X, np.arange(10))

    def test_non_2d_cloud_is_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            project_for_cycle(np.zeros((4, 3, 2)), np.array([0, 1, 2]))

    def test_zero_width_cloud_is_rejected(self):
        with pytest.raises(ValueError, match="at least one column"):
            project_for_cycle(np.zeros((4, 0)), np.array([0, 1, 2]))

    def test_out_of_range_vertex_is_rejected_by_name(self):
        X = make_tilted_circle()
        with pytest.raises(IndexError, match="999"):
            project_for_cycle(X, np.array([0, 1, 999]))

    def test_negative_vertex_is_rejected(self):
        X = make_tilted_circle()
        with pytest.raises(IndexError, match="-1"):
            project_for_cycle(X, np.array([-1, 1, 2]))

    @pytest.mark.parametrize(
        "vertices",
        [np.array([0.0, 1.0, 2.0]), np.array([True, False, True])],
    )
    def test_non_integer_vertices_are_rejected(self, vertices):
        """A bool array would otherwise be silently read as indices [0, 1]
        rather than as a mask."""
        X = make_tilted_circle()
        with pytest.raises(ValueError, match="integer array"):
            project_for_cycle(X, vertices)

    def test_python_list_of_indices_is_accepted(self):
        X = make_tilted_circle()
        assert project_for_cycle(X, [0, 1, 2, 3]).coords.shape == (len(X), 2)


class TestProjectionDataclass:

    def test_is_frozen(self, torus):
        X, cycles = torus
        p = project_for_cycle(X, cycles[0])
        with pytest.raises(Exception):
            p.variance_retained = 0.0

    def test_fields_have_the_contracted_types(self, torus):
        X, cycles = torus
        p = project_for_cycle(X, cycles[0])

        assert isinstance(p, Projection)
        assert p.coords.dtype == np.float64
        assert isinstance(p.variance_retained, float)
        assert isinstance(p.baseline_retained, float)
        assert isinstance(p.is_degenerate, bool)
