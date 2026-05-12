"""
Tests for representative_cycles.py
====================================
Three categories:

1. **Exact** — the Zhu (2013) §2.3 rectangle whose birth/death are
   computable to floating-point precision.
2. **Topological** — circle (β₁=1) and figure-eight (β₁=2) verified by
   persistence spectral-gap checks.
3. **API contracts** — precomputed metric, min_persistence filter,
   reconstruct_cycles=False, fit returns self, input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from representative_cycles import CycleFeature, RepresentativeCycles

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

RECT_POINTS = np.array(
    [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]], dtype=np.float64
)
RECT_BIRTH = 2.0
RECT_DEATH = math.sqrt(5)          # ≈ 2.236068
RECT_PERSISTENCE = RECT_DEATH - RECT_BIRTH


def _make_circle(n: int = 80, noise: float = 0.05) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += RNG.standard_normal(X.shape) * noise
    return X


def _make_figure_eight(n: int = 120, noise: float = 0.05) -> np.ndarray:
    half = n // 2
    theta = np.linspace(0, 2 * np.pi, half, endpoint=False)
    left = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
    right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
    X = np.vstack([left, right])
    X += RNG.standard_normal(X.shape) * noise
    return X


# ---------------------------------------------------------------------------
# 1. Exact example — Zhu §2.3 rectangle
# ---------------------------------------------------------------------------

class TestRectangleExact:
    """4-point rectangle with analytically known birth/death values."""

    @pytest.fixture(scope="class")
    def rc(self):
        r = RepresentativeCycles(min_persistence=0.0)
        r.fit(RECT_POINTS)
        return r

    def test_exactly_one_h1_feature(self, rc):
        assert len(rc.features_) == 1, (
            f"Expected 1 H1 feature, got {len(rc.features_)}"
        )

    def test_birth_equals_two(self, rc):
        assert abs(rc.features_[0].birth - RECT_BIRTH) < 1e-4, (
            f"birth = {rc.features_[0].birth:.6f}, expected {RECT_BIRTH}"
        )

    def test_death_equals_sqrt5(self, rc):
        assert abs(rc.features_[0].death - RECT_DEATH) < 1e-4, (
            f"death = {rc.features_[0].death:.6f}, expected {RECT_DEATH:.6f}"
        )

    def test_persistence_equals_sqrt5_minus_2(self, rc):
        assert abs(rc.features_[0].persistence - RECT_PERSISTENCE) < 1e-4

    def test_representative_cycle_has_four_edges(self, rc):
        """The rectangle boundary has exactly 4 edges."""
        assert len(rc.features_[0].cycle_edges) == 4, (
            f"Expected 4 cycle edges, got {len(rc.features_[0].cycle_edges)}"
        )

    def test_cycle_is_closed(self, rc):
        """Every vertex in the cycle must appear an even number of times
        (closed loop over Z/2Z)."""
        edges = rc.features_[0].cycle_edges
        vertices, counts = np.unique(edges, return_counts=True)
        assert np.all(counts % 2 == 0), (
            f"Cycle is not closed: vertex counts = {dict(zip(vertices, counts))}"
        )

    def test_cycle_vertices_are_subset_of_points(self, rc):
        n = RECT_POINTS.shape[0]
        assert np.all(rc.features_[0].cycle_vertices < n)
        assert np.all(rc.features_[0].cycle_vertices >= 0)

    def test_diagrams_shape(self, rc):
        assert rc.diagrams_ is not None
        assert len(rc.diagrams_) >= 2       # H0, H1
        assert rc.diagrams_[1].shape[1] == 2


# ---------------------------------------------------------------------------
# 2. Circle — β₁ = 1
# ---------------------------------------------------------------------------

class TestCircle:
    """80-point noisy circle should produce one dominant H₁ bar."""

    @pytest.fixture(scope="class")
    def rc(self):
        X = _make_circle(n=80, noise=0.05)
        r = RepresentativeCycles(min_persistence=0.0)
        r.fit(X)
        return r

    def test_has_at_least_one_feature(self, rc):
        assert len(rc.features_) >= 1

    def test_top_feature_has_high_persistence(self, rc):
        assert rc.features_[0].persistence > 1.0, (
            f"Top bar persistence = {rc.features_[0].persistence:.4f}, "
            "expected > 1.0 for a circle"
        )

    def test_large_spectral_gap(self, rc):
        """Persistence of the dominant bar should dwarf the second bar."""
        if len(rc.features_) < 2:
            pytest.skip("Only one feature — trivially a single loop.")
        gap = rc.features_[0].persistence - rc.features_[1].persistence
        assert gap > 0.5, (
            f"Spectral gap = {gap:.4f}, expected > 0.5"
        )

    def test_representative_cycle_is_nontrivial(self, rc):
        assert len(rc.features_[0].cycle_edges) >= 3

    def test_point_cloud_stored(self, rc):
        assert rc.point_cloud_ is not None
        assert rc.point_cloud_.shape[1] == 2


# ---------------------------------------------------------------------------
# 3. Figure-eight — β₁ = 2
# ---------------------------------------------------------------------------

class TestFigureEight:
    """Two tangent circles; should produce two dominant H₁ bars."""

    @pytest.fixture(scope="class")
    def rc(self):
        X = _make_figure_eight(n=120, noise=0.05)
        r = RepresentativeCycles(min_persistence=0.0)
        r.fit(X)
        return r

    def test_has_at_least_two_features(self, rc):
        assert len(rc.features_) >= 2, (
            f"Expected >= 2 features, got {len(rc.features_)}"
        )

    def test_both_dominant_bars_exceed_one(self, rc):
        assert rc.features_[0].persistence > 1.0
        assert rc.features_[1].persistence > 1.0

    def test_two_bars_have_comparable_persistence(self, rc):
        """Both loops are the same size, so bars should be nearly equal."""
        ratio = rc.features_[1].persistence / rc.features_[0].persistence
        assert ratio > 0.7, (
            f"Persistence ratio = {ratio:.4f}, expected > 0.7"
        )

    def test_spectral_gap_after_second_bar(self, rc):
        """Third bar (noise) should be far below the two dominant bars."""
        if len(rc.features_) < 3:
            pytest.skip("Fewer than 3 features.")
        third = rc.features_[2].persistence
        assert third < 0.5, (
            f"Third-bar persistence = {third:.4f}, expected < 0.5"
        )

    def test_cycle_edges_nontrivial_both_features(self, rc):
        assert len(rc.features_[0].cycle_edges) >= 3
        assert len(rc.features_[1].cycle_edges) >= 3


# ---------------------------------------------------------------------------
# 4. Precomputed distance matrix
# ---------------------------------------------------------------------------

class TestPrecomputedMetric:
    """metric='precomputed' should give identical diagrams to 'euclidean'."""

    @pytest.fixture(scope="class")
    def both(self):
        X = _make_circle(n=60, noise=0.05)
        from scipy.spatial.distance import cdist

        D = cdist(X, X)

        rc_euc = RepresentativeCycles(min_persistence=0.0, metric="euclidean")
        rc_euc.fit(X)

        rc_pre = RepresentativeCycles(min_persistence=0.0, metric="precomputed")
        rc_pre.fit(D)

        return rc_euc, rc_pre

    def test_same_number_of_features(self, both):
        rc_euc, rc_pre = both
        assert len(rc_euc.features_) == len(rc_pre.features_)

    def test_same_persistence_values(self, both):
        rc_euc, rc_pre = both
        for f_e, f_p in zip(rc_euc.features_, rc_pre.features_):
            assert abs(f_e.birth - f_p.birth) < 1e-3, (
                f"birth mismatch: euclidean={f_e.birth:.6f} "
                f"precomputed={f_p.birth:.6f}"
            )
            assert abs(f_e.persistence - f_p.persistence) < 1e-3

    def test_precomputed_has_point_cloud_for_visualisation(self, both):
        """MDS embedding should be set so that plot methods work."""
        _, rc_pre = both
        assert rc_pre.point_cloud_ is not None
        assert rc_pre.point_cloud_.ndim == 2
        assert rc_pre.point_cloud_.shape[1] == 2

    def test_precomputed_rejects_non_square(self):
        rc = RepresentativeCycles(metric="precomputed")
        with pytest.raises(ValueError, match="square"):
            rc.fit(np.zeros((4, 3)))

    def test_precomputed_rejects_asymmetric(self):
        D = np.array([[0.0, 1.0], [2.0, 0.0]])   # asymmetric
        rc = RepresentativeCycles(metric="precomputed")
        with pytest.raises(ValueError, match="symmetric"):
            rc.fit(D)

    def test_precomputed_rejects_nonzero_diagonal(self):
        D = np.array([[1.0, 2.0], [2.0, 1.0]])   # non-zero diagonal
        rc = RepresentativeCycles(metric="precomputed")
        with pytest.raises(ValueError, match="diagonal"):
            rc.fit(D)


# ---------------------------------------------------------------------------
# 5. API contracts
# ---------------------------------------------------------------------------

class TestApiContracts:

    def test_fit_returns_self(self):
        X = _make_circle(n=40, noise=0.05)
        rc = RepresentativeCycles()
        result = rc.fit(X)
        assert result is rc

    def test_min_persistence_filters_features(self):
        """Setting a high min_persistence should reduce the feature count."""
        X = _make_circle(n=80, noise=0.05)

        rc_all = RepresentativeCycles(min_persistence=0.0)
        rc_all.fit(X)

        rc_filtered = RepresentativeCycles(min_persistence=0.5)
        rc_filtered.fit(X)

        assert len(rc_filtered.features_) < len(rc_all.features_), (
            "min_persistence=0.5 should discard some features"
        )

    def test_reconstruct_cycles_false_leaves_empty_edges(self):
        X = _make_circle(n=40, noise=0.05)
        rc = RepresentativeCycles(min_persistence=0.0, reconstruct_cycles=False)
        rc.fit(X)
        for f in rc.features_:
            assert len(f.cycle_edges) == 0, (
                "reconstruct_cycles=False should leave cycle_edges empty"
            )

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            RepresentativeCycles(metric="manhattan")

    def test_features_sorted_by_persistence_descending(self):
        X = _make_figure_eight(n=80, noise=0.05)
        rc = RepresentativeCycles(min_persistence=0.0)
        rc.fit(X)
        pers = [f.persistence for f in rc.features_]
        assert pers == sorted(pers, reverse=True)

    def test_diagrams_accessible_before_features(self):
        """diagrams_ should always be set after fit(), even if no features
        survive the min_persistence filter."""
        X = _make_circle(n=20, noise=0.0)
        rc = RepresentativeCycles(min_persistence=999.0)  # filter everything
        rc.fit(X)
        assert rc.diagrams_ is not None
        assert len(rc.features_) == 0

    def test_point_cloud_preserved_float64(self):
        """Coordinates passed to fit() should be stored as float64, not
        downgraded to float32 (which ripser_parallel requires internally)."""
        X = np.array([[0.1234567890123456, 0.0],
                      [1.0, 0.0],
                      [1.0, 1.0],
                      [0.0, 1.0]], dtype=np.float64)
        rc = RepresentativeCycles(min_persistence=0.0)
        rc.fit(X)
        assert rc.point_cloud_.dtype == np.float64

    def test_summary_runs_without_error(self, capsys):
        X = _make_circle(n=40, noise=0.05)
        rc = RepresentativeCycles(min_persistence=0.0)
        rc.fit(X)
        rc.summary()   # should not raise
        captured = capsys.readouterr()
        assert "Birth" in captured.out

    def test_two_point_cloud(self):
        """Two points — no H1 cycles are possible."""
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        rc = RepresentativeCycles(min_persistence=0.0)
        rc.fit(X)
        assert len(rc.features_) == 0

    def test_1d_point_cloud_raises(self):
        rc = RepresentativeCycles(metric="euclidean")
        with pytest.raises(ValueError):
            rc.fit(np.array([1.0, 2.0, 3.0]))  # 1-D, not 2-D

    def test_cycle_feature_dataclass_fields(self):
        """Ensure CycleFeature has all documented fields."""
        f = CycleFeature(
            index=0, birth=0.5, death=1.5, persistence=1.0,
            birth_edge=(0, 1), death_edge=(1, 2),
            cycle_edges=np.array([[0, 1], [1, 2], [2, 0]]),
            cycle_vertices=np.array([0, 1, 2]),
        )
        assert f.persistence == pytest.approx(1.0)
        assert f.birth_edge == (0, 1)
        assert len(f.cycle_edges) == 3
