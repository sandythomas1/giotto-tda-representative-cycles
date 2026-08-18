"""The shared Rips graph: correctness, memory, and speed (spec 001 T7).

Reconstruction needs the Rips 1-skeleton at every feature's birth radius.  The
legacy path rebuilt a dense ``n × n`` mask per feature; :class:`RipsGraphCache`
builds one filtration-sorted edge list per fit and slices it.

The optimisation is only worth anything if it is *invisible* — same cycles,
less time — so most of this module tests equality against the path it
replaces, and only the two acceptance criteria measure cost:

- **AC13** — an in-process ratio on the 1500-point torus, never a wall-clock
  number, because a wall-clock target hard-codes one machine into the suite.
  Marked ``perf``; deselect with ``-m "not perf"``.  Its fixture costs ~5 s of
  ripser, which is why nothing else in this file uses it.
- **AC13a** — the sorted-edge structure's dtypes and total ``nbytes``.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pytest
from gph import ripser_parallel
from scipy.spatial.distance import cdist

from repcycles import core
from repcycles.core import RepresentativeCycles
from repcycles.reconstruction import RipsGraphCache, reconstruct_cycle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Ratio AC13 demands of the reused-graph path against the dense-mask path.
REQUIRED_SPEEDUP = 2.0

#: Points in the benchmark torus named by AC13.
BENCHMARK_POINTS = 1500

#: Points in the everyday torus.  Big enough to produce a few hundred H₁
#: classes and exercise every cache branch, small enough that ripser costs
#: ~0.1 s instead of the benchmark's ~5 s.
WORKING_POINTS = 400


#: Zhu (2013) §2.3 rectangle: one H₁ class, born at 2.0, dying at √5.
RECTANGLE = np.array(
    [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
)


def _torus(n: int, seed: int = 7, R: float = 2.0, r: float = 0.8) -> np.ndarray:
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


def _h1_generators(
    X: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], float]]]:
    """``(D, [(birth_edge, birth_radius), ...])`` straight from ripser.

    The radius is the exact float64 edge length, matching what
    :meth:`RepresentativeCycles.fit` cuts the graph at.
    """
    D = cdist(X, X)
    result = ripser_parallel(
        X.astype(np.float32),
        maxdim=1,
        return_generators=True,
        n_threads=1,
    )
    rows = result["gens"][1][0]
    return D, [
        ((int(u), int(v)), float(D[int(u), int(v)])) for u, v, _, _ in rows
    ]


def _reconstruct_all_legacy(D, generators):
    """One dense ``n × n`` mask per feature — the path being replaced."""
    return [reconstruct_cycle(D, edge, radius) for edge, radius in generators]


def _reconstruct_all_cached(D, generators):
    """One sorted edge list per fit, visited in ascending birth radius.

    Returned in the caller's original order so the two paths line up
    feature-for-feature; the *visiting* order is the ascending one, which is
    what lets the cache reuse its prefix.
    """
    order = sorted(range(len(generators)), key=lambda i: generators[i][1])
    cache = RipsGraphCache(
        D, max_edge_length=max(radius for _, radius in generators)
    )
    results = [None] * len(generators)
    for i in order:
        edge, radius = generators[i]
        results[i] = reconstruct_cycle(D, edge, radius, graph_cache=cache)
    return results


@pytest.fixture(scope="module")
def working_torus():
    """400-point torus with its generators — the cheap workhorse fixture."""
    return _h1_generators(_torus(WORKING_POINTS))


@pytest.fixture(scope="module")
def benchmark_torus():
    """The 1500-point torus AC13 names.  ~5 s of ripser: perf tests only."""
    return _h1_generators(_torus(BENCHMARK_POINTS))


def _dense_mask_nbytes(n: int) -> int:
    """Bytes of the boolean ``n × n`` mask the edge list replaces."""
    return n * n * np.dtype(bool).itemsize


# ---------------------------------------------------------------------------
# The graph itself
# ---------------------------------------------------------------------------


class TestGraphAtMatchesTheDenseMask:
    """The cache is a *representation* change, not a topology change."""

    def test_every_radius_reproduces_the_dense_mask_exactly(self):
        D = cdist(_torus(60), _torus(60))
        cache = RipsGraphCache(D)

        for radius in np.linspace(0.05, D.max(), 25):
            expected = np.triu(D <= radius, k=1)
            actual = cache.graph_at(radius).toarray() != 0
            # A genuine zero-length edge would read as absent here; the torus
            # has none, and the (i, j) sets are what this asserts.
            assert np.array_equal(np.triu(actual, k=1), expected), radius

    def test_edges_are_stored_once_in_upper_triangular_form(self):
        D = cdist(_torus(40), _torus(40))
        graph = RipsGraphCache(D).graph_at(1.0)
        lower = graph.toarray()[np.tril_indices(40)]

        # csgraph is always called with directed=False, which reads an edge in
        # both directions, so storing the mirror would double memory for
        # nothing.
        assert not lower.any()

    def test_weights_are_the_original_float64_distances(self):
        D = cdist(_torus(40), _torus(40))
        graph = RipsGraphCache(D).graph_at(1.0)
        rows, cols = graph.nonzero()

        assert graph.dtype == np.float64
        assert np.array_equal(np.asarray(graph[rows, cols]).ravel(), D[rows, cols])

    def test_infinite_distances_never_become_edges(self):
        # `+inf` in a precomputed matrix means "no edge" (F10), so it must not
        # be stored as a very long one.
        D = np.array(
            [
                [0.0, 1.0, np.inf],
                [1.0, 0.0, 1.0],
                [np.inf, 1.0, 0.0],
            ]
        )
        cache = RipsGraphCache(D)

        assert cache.n_edges == 2
        assert cache.graph_at(np.inf).nnz == 2


class TestFloat32KeysDoNotChangeTheComplex:
    """The cheap sort key buys memory, not approximation."""

    #: 1 + 2⁻²⁴ is not representable in float32: it rounds *down* to 1.0.
    UNREPRESENTABLE = 1.0 + 2.0**-24

    def _matrix(self) -> np.ndarray:
        d = self.UNREPRESENTABLE
        return np.array([[0.0, d, 2.0], [d, 0.0, 2.0], [2.0, 2.0, 0.0]])

    def test_the_fixture_really_straddles_a_float32_boundary(self):
        assert np.float32(self.UNREPRESENTABLE) < self.UNREPRESENTABLE

    def test_edge_is_present_at_its_own_exact_length(self):
        cache = RipsGraphCache(self._matrix())
        assert cache.graph_at(self.UNREPRESENTABLE).nnz == 1

    def test_edge_is_absent_just_below_its_length(self):
        # Rounding the key *up* is what prevents a too-long edge sneaking in;
        # the exact float64 re-test is what keeps a real one from being lost.
        cache = RipsGraphCache(self._matrix())
        assert cache.graph_at(1.0).nnz == 0


class TestGraphReuse:
    """Rebuild only when the prefix actually changes."""

    def test_the_same_radius_returns_the_same_object(self):
        cache = RipsGraphCache(cdist(_torus(40), _torus(40)))
        assert cache.graph_at(1.0) is cache.graph_at(1.0)

    def test_a_radius_that_admits_no_new_edge_reuses_the_graph(self):
        D = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 3.0], [3.0, 3.0, 0.0]])
        cache = RipsGraphCache(D)
        # Nothing lies between 1.0 and 2.0, so the prefix is unchanged.
        assert cache.graph_at(1.0) is cache.graph_at(2.0)

    def test_crossing_an_edge_length_rebuilds(self):
        D = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 3.0], [3.0, 3.0, 0.0]])
        cache = RipsGraphCache(D)
        first = cache.graph_at(1.0)
        second = cache.graph_at(3.0)

        assert first is not second
        assert (first.nnz, second.nnz) == (1, 3)

    def test_descending_radii_stay_correct(self):
        """Out-of-order calls cost rebuilds, but must never lie."""
        D = cdist(_torus(50), _torus(50))
        cache = RipsGraphCache(D)

        for radius in [1.5, 0.4, 1.2, 0.2, 2.0]:
            expected = int(np.triu(D <= radius, k=1).sum())
            assert cache.graph_at(radius).nnz == expected, radius


class TestGraphAtRefusesWhatItCannotAnswer:
    """A truncated complex returned silently is exactly what the constitution
    forbids; the cache raises instead."""

    def test_radius_beyond_the_limit_is_refused(self):
        cache = RipsGraphCache(cdist(_torus(30), _torus(30)), max_edge_length=1.0)
        with pytest.raises(ValueError, match="max_edge_length"):
            cache.graph_at(1.5)

    def test_radius_at_the_limit_is_answered(self):
        D = cdist(_torus(30), _torus(30))
        cache = RipsGraphCache(D, max_edge_length=1.0)
        assert cache.graph_at(1.0).nnz == int(np.triu(D <= 1.0, k=1).sum())

    def test_nan_radius_is_refused(self):
        cache = RipsGraphCache(cdist(_torus(30), _torus(30)))
        with pytest.raises(ValueError, match="NaN"):
            cache.graph_at(np.nan)

    def test_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            RipsGraphCache(np.zeros((3, 4)))

    def test_nan_limit_is_refused(self):
        with pytest.raises(ValueError, match="NaN"):
            RipsGraphCache(np.zeros((3, 3)), max_edge_length=np.nan)


# ---------------------------------------------------------------------------
# AC13a — memory
# ---------------------------------------------------------------------------


class TestAcceptanceCriterion13a:
    """float32 keys and int32 indices, within 2× the dense mask."""

    def test_dtypes_are_the_ones_the_memory_budget_assumes(self, working_torus):
        D, generators = working_torus
        cache = RipsGraphCache(
            D, max_edge_length=max(r for _, r in generators)
        )

        assert cache._key.dtype == np.float32
        assert cache._rows.dtype == np.int32
        assert cache._cols.dtype == np.int32

    def test_structure_stays_within_twice_the_dense_mask(self, working_torus):
        D, generators = working_torus
        cache = RipsGraphCache(
            D, max_edge_length=max(r for _, r in generators)
        )

        # Bounding the edge list at the largest birth radius any feature asks
        # for is what makes this hold with room to spare: the mask is n²
        # regardless, while the edge list only holds edges short enough to
        # matter.
        assert cache.nbytes <= 2 * _dense_mask_nbytes(WORKING_POINTS)

    def test_the_limit_is_what_keeps_it_small(self, working_torus):
        D, generators = working_torus
        limit = max(r for _, r in generators)

        bounded = RipsGraphCache(D, max_edge_length=limit)
        unbounded = RipsGraphCache(D)

        # Without a limit the structure is every pair — ~6× the mask, and the
        # reason `fit` always passes one.
        assert bounded.nbytes < unbounded.nbytes
        assert unbounded.n_edges == WORKING_POINTS * (WORKING_POINTS - 1) // 2


# ---------------------------------------------------------------------------
# Equivalence — the cheap half of AC13
# ---------------------------------------------------------------------------


class TestCachedPathMatchesLegacyPath:
    """Same cycles, feature for feature.

    Compared on total geodesic length and verification status rather than on
    edge arrays: equal-cost shortest paths are not unique, so two correct
    graph constructions may legitimately trace different edges (F8/B5).
    """

    def test_every_feature_agrees_on_length_and_verification(
        self, working_torus
    ):
        D, generators = working_torus
        assert len(generators) > 50, "fixture should exercise many features"

        legacy = _reconstruct_all_legacy(D, generators)
        cached = _reconstruct_all_cached(D, generators)

        for i, (want, got) in enumerate(zip(legacy, cached)):
            assert got.length == pytest.approx(want.length, abs=1e-9), i
            assert got.is_verified == want.is_verified, i

    def test_cached_cycles_are_closed_loops_within_their_radius(
        self, working_torus
    ):
        D, generators = working_torus
        cached = _reconstruct_all_cached(D, generators)

        for (edge, radius), result in zip(generators, cached):
            if not result.is_verified:
                continue
            lengths = D[result.edges[:, 0], result.edges[:, 1]]
            assert lengths.max() <= max(radius, D[edge]) * (1 + 1e-9)
            assert result.path[0] == result.path[-1]

    def test_the_shared_graph_survives_every_feature(self, working_torus):
        """Each feature masks its own birth edge in place; a leak would
        corrupt every later feature rather than fail loudly."""
        D, generators = working_torus
        cache = RipsGraphCache(
            D, max_edge_length=max(r for _, r in generators)
        )
        radius = max(r for _, r in generators)
        expected = int(np.triu(D <= radius, k=1).sum())

        for edge, r in sorted(generators, key=lambda g: g[1]):
            reconstruct_cycle(D, edge, r, graph_cache=cache)

        graph = cache.graph_at(radius)
        assert np.isfinite(graph.data).all()
        assert graph.nnz == expected


# ---------------------------------------------------------------------------
# fit() wiring
# ---------------------------------------------------------------------------


class TestFitBuildsOneGraphPerFit:

    @staticmethod
    def _counting_cache(monkeypatch):
        built = []

        class CountingCache(RipsGraphCache):
            def __init__(self, dist_matrix, max_edge_length=np.inf):
                built.append(float(max_edge_length))
                super().__init__(dist_matrix, max_edge_length)

        monkeypatch.setattr(core, "RipsGraphCache", CountingCache)
        return built

    def test_a_multi_feature_fit_builds_exactly_one(self, monkeypatch):
        built = self._counting_cache(monkeypatch)
        rc = RepresentativeCycles().fit(_torus(120))

        assert len(rc.features_) > 1
        assert len(built) == 1

    def test_the_limit_is_the_largest_birth_radius_asked_for(
        self, monkeypatch
    ):
        built = self._counting_cache(monkeypatch)
        rc = RepresentativeCycles().fit(_torus(120))

        largest = max(
            rc._dist_matrix_[f.birth_edge[0], f.birth_edge[1]]
            for f in rc.features_
        )
        assert built[0] == pytest.approx(largest)

    def test_a_single_feature_fit_skips_the_cache(self, monkeypatch):
        built = self._counting_cache(monkeypatch)
        rc = RepresentativeCycles().fit(RECTANGLE)

        # One sort over every edge costs more than the one dense mask it would
        # replace, so there is nothing to amortise.
        assert len(rc.features_) == 1
        assert built == []

    def test_reconstruction_off_builds_nothing(self, monkeypatch):
        built = self._counting_cache(monkeypatch)
        RepresentativeCycles(reconstruct_cycles=False).fit(_torus(120))
        assert built == []

    def test_features_are_identical_with_and_without_the_cache(
        self, monkeypatch
    ):
        X = _torus(150)
        with_cache = RepresentativeCycles().fit(X).features_

        # Raising the threshold above the feature count forces the per-feature
        # dense-mask path through the very same code.
        monkeypatch.setattr(core, "_MIN_FEATURES_FOR_GRAPH_CACHE", 10**9)
        without_cache = RepresentativeCycles().fit(X).features_

        assert len(with_cache) == len(without_cache)
        for cached, plain in zip(with_cache, without_cache):
            assert cached.index == plain.index
            assert cached.birth_edge == plain.birth_edge
            assert cached.cycle_length == pytest.approx(
                plain.cycle_length, abs=1e-9
            )
            assert cached.is_verified == plain.is_verified


# ---------------------------------------------------------------------------
# AC13 — the ratio
# ---------------------------------------------------------------------------


@pytest.mark.perf
class TestAcceptanceCriterion13:
    """≥ 2× on the 1500-point torus, measured in-process.

    Both paths run in the same interpreter on the same distance matrix, so the
    comparison is between two algorithms rather than between two machines.  The
    measured margin is ~11×, which is why a single timed run of each is enough
    to keep this from flaking.
    """

    @staticmethod
    def _timed(fn, D, generators):
        start = time.perf_counter()
        results = fn(D, generators)
        return time.perf_counter() - start, results

    def test_reused_graph_is_at_least_twice_as_fast(self, benchmark_torus):
        D, generators = benchmark_torus
        assert len(generators) > 100

        legacy_seconds, legacy = self._timed(
            _reconstruct_all_legacy, D, generators
        )
        cached_seconds, cached = self._timed(
            _reconstruct_all_cached, D, generators
        )

        speedup = legacy_seconds / cached_seconds
        assert speedup >= REQUIRED_SPEEDUP, (
            f"reused-graph path was only {speedup:.2f}× the dense-mask path "
            f"({cached_seconds:.3f}s vs {legacy_seconds:.3f}s over "
            f"{len(generators)} features)"
        )

        # Identical cycles, or the speed-up is meaningless.
        for i, (want, got) in enumerate(zip(legacy, cached)):
            assert got.length == pytest.approx(want.length, abs=1e-9), i
            assert got.is_verified == want.is_verified, i

    def test_benchmark_structure_stays_within_twice_the_dense_mask(
        self, benchmark_torus
    ):
        D, generators = benchmark_torus
        cache = RipsGraphCache(
            D, max_edge_length=max(r for _, r in generators)
        )
        assert cache.nbytes <= 2 * _dense_mask_nbytes(BENCHMARK_POINTS)
