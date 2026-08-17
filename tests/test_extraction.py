"""Extraction-layer tests: essential classes, ordering, export (spec 001 T5, T8).

These cover the part of the pipeline that turns ripser's output into
``features_`` — specifically the classes that the previous implementation
dropped on the floor.
"""

from __future__ import annotations

import builtins
import math

import numpy as np
import pytest

from repcycles import RepresentativeCycles
from repcycles.core import _feature_sort_key

RNG = np.random.default_rng(42)


def _circle(n: int = 60, noise: float = 0.03) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([np.cos(t), np.sin(t)])
    return X + RNG.standard_normal(X.shape) * noise


def _two_circles(noise: float = 0.02) -> np.ndarray:
    """Two well-separated circles of different radius.

    Truncating the filtration between their two death radii leaves one finite
    bar and one essential bar — the mixed case that broke positional pairing.
    """
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    small = np.column_stack([np.cos(t), np.sin(t)]) * 0.5
    big = np.column_stack([np.cos(t), np.sin(t)]) * 3.0 + np.array([12.0, 0.0])
    X = np.vstack([small, big])
    return X + RNG.standard_normal(X.shape) * noise


# ---------------------------------------------------------------------------
# AC1 — essential classes are no longer dropped
# ---------------------------------------------------------------------------

class TestEssentialClasses:
    """The headline fix: a truncated filtration keeps its dominant loop."""

    @pytest.fixture(scope="class")
    def truncated(self):
        return RepresentativeCycles(
            min_persistence=0.0, max_edge_length=1.0
        ).fit(_circle())

    def test_ac1_exactly_one_essential_feature(self, truncated):
        """Before this spec, this returned zero features."""
        assert len(truncated.features_) == 1
        assert truncated.features_[0].is_essential

    def test_ac1_death_and_persistence_are_infinite(self, truncated):
        f = truncated.features_[0]
        assert np.isinf(f.death)
        assert np.isinf(f.persistence)

    def test_ac1_cycle_is_reconstructed_and_nontrivial(self, truncated):
        f = truncated.features_[0]
        assert f.n_edges >= 3
        assert f.is_verified
        assert f.cycle_length > 0

    def test_essential_death_edge_is_the_sentinel(self, truncated):
        """Nothing kills an essential class, so there is no death simplex."""
        assert truncated.features_[0].death_edge == (-1, -1)

    def test_include_essential_false_reproduces_old_behaviour(self):
        rc = RepresentativeCycles(
            min_persistence=0.0, max_edge_length=1.0, include_essential=False
        ).fit(_circle())
        assert rc.features_ == []

    def test_min_persistence_never_filters_an_essential_feature(self):
        """F3: infinite persistence clears any threshold."""
        rc = RepresentativeCycles(
            min_persistence=1e6, max_edge_length=1.0
        ).fit(_circle())
        assert len(rc.features_) == 1
        assert rc.features_[0].is_essential

    def test_default_filtration_produces_no_essential_features(self):
        """B1 blast radius: with thresh=inf the complex becomes a full
        simplex, every class dies, so this change cannot affect callers who
        do not truncate."""
        rc = RepresentativeCycles(min_persistence=0.0).fit(_circle())
        assert all(not f.is_essential for f in rc.features_)
        assert all(np.isfinite(f.death) for f in rc.features_)


class TestMixedFiniteAndEssential:
    """AC3 — the case where positional pairing goes wrong."""

    @pytest.fixture(scope="class")
    def mixed(self):
        rc = RepresentativeCycles(
            min_persistence=0.0, max_edge_length=1.5
        ).fit(_two_circles())
        return rc

    def test_has_both_kinds(self, mixed):
        kinds = {f.is_essential for f in mixed.features_}
        assert kinds == {True, False}, (
            f"expected one finite and one essential bar, got {kinds}"
        )

    def test_ac3_every_birth_matches_its_own_birth_edge(self, mixed):
        """Each feature's birth must be the length of ITS birth edge — the
        property that positional pairing silently violates."""
        D = mixed._dist_matrix_
        for f in mixed.features_:
            u, v = f.birth_edge
            assert abs(f.birth - D[u, v]) < 1e-4, (
                f"feature {f.index}: birth={f.birth:.6f} but "
                f"|birth edge|={D[u, v]:.6f}"
            )

    def test_essential_sorts_first(self, mixed):
        assert mixed.features_[0].is_essential

    def test_all_features_verified(self, mixed):
        assert all(f.is_verified for f in mixed.features_)


# ---------------------------------------------------------------------------
# F8a — total, deterministic ordering
# ---------------------------------------------------------------------------

class TestOrdering:

    def test_features_sorted_by_persistence_descending(self):
        rc = RepresentativeCycles(min_persistence=0.0).fit(_circle(80))
        pers = [f.persistence for f in rc.features_]
        assert pers == sorted(pers, reverse=True)

    def test_sort_key_is_total_for_multiple_essential_features(self):
        """Several essential features all have persistence == inf; without
        the birth/index tie-breaks their relative order would be whatever
        ripser happened to emit."""
        from repcycles.feature import CycleFeature

        a = CycleFeature(index=5, birth=0.9, death=np.inf,
                         persistence=np.inf, is_essential=True)
        b = CycleFeature(index=2, birth=0.3, death=np.inf,
                         persistence=np.inf, is_essential=True)
        c = CycleFeature(index=9, birth=0.3, death=np.inf,
                         persistence=np.inf, is_essential=True)
        ordered = sorted([a, b, c], key=_feature_sort_key)
        assert [f.index for f in ordered] == [2, 9, 5], (
            "essential features must order by birth, then diagram index"
        )

    def test_essential_outranks_every_finite_feature(self):
        from repcycles.feature import CycleFeature

        essential = CycleFeature(index=1, birth=0.9, death=np.inf,
                                 persistence=np.inf, is_essential=True)
        finite = CycleFeature(index=0, birth=0.1, death=99.0,
                              persistence=98.9)
        ordered = sorted([finite, essential], key=_feature_sort_key)
        assert ordered[0] is essential

    def test_repeated_fits_give_identical_ordering(self):
        X = _two_circles()
        a = RepresentativeCycles(max_edge_length=1.5).fit(X)
        b = RepresentativeCycles(max_edge_length=1.5).fit(X)
        assert [f.index for f in a.features_] == [f.index for f in b.features_]


# ---------------------------------------------------------------------------
# T8 — summary() and to_dataframe()
# ---------------------------------------------------------------------------

class TestSummary:

    def test_shows_verification_column(self, capsys):
        RepresentativeCycles(min_persistence=0.3).fit(_circle()).summary()
        out = capsys.readouterr().out
        assert "Ok" in out
        assert "yes" in out

    def test_renders_infinite_death_as_symbol(self, capsys):
        RepresentativeCycles(max_edge_length=1.0).fit(_circle()).summary()
        out = capsys.readouterr().out
        assert "∞" in out
        assert "inf" not in out.lower().replace("infinite", "")

    def test_empty_features_message(self, capsys):
        RepresentativeCycles(min_persistence=1e9).fit(_circle()).summary()
        assert "No H1 features" in capsys.readouterr().out

    def test_reports_failed_verification_count(self, capsys):
        """A degraded cycle must be called out, not just flagged in a column."""
        rc = RepresentativeCycles(min_persistence=0.0).fit(_circle(20))
        rc.features_[0].is_verified = False
        rc.summary()
        assert "failed verification" in capsys.readouterr().out


class TestToDataFrame:

    @pytest.fixture(scope="class")
    def df(self):
        pd = pytest.importorskip("pandas")
        return RepresentativeCycles(min_persistence=0.3).fit(_circle()).to_dataframe()

    def test_one_row_per_feature(self, df):
        rc = RepresentativeCycles(min_persistence=0.3).fit(_circle())
        assert len(df) == len(rc.features_)

    def test_columns_cover_every_scalar_field(self, df):
        expected = {
            "index", "birth", "death", "persistence",
            "birth_u", "birth_v", "death_u", "death_v",
            "n_edges", "cycle_length", "is_essential", "is_verified",
        }
        assert set(df.columns) == expected

    def test_essential_feature_keeps_inf(self):
        pytest.importorskip("pandas")
        df = RepresentativeCycles(max_edge_length=1.0).fit(_circle()).to_dataframe()
        assert np.isinf(df["death"]).all()
        assert bool(df["is_essential"].all())

    def test_row_order_matches_features(self, df):
        rc = RepresentativeCycles(min_persistence=0.3).fit(_circle())
        assert list(df["index"]) == [f.index for f in rc.features_]

    def test_empty_features_gives_empty_frame_with_columns(self):
        pytest.importorskip("pandas")
        df = RepresentativeCycles(min_persistence=1e9).fit(_circle()).to_dataframe()
        assert len(df) == 0
        assert "cycle_length" in df.columns

    def test_missing_pandas_raises_clear_import_error(self, monkeypatch):
        """pandas is optional; its absence must name the install command."""
        real_import = builtins.__import__

        def _no_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pandas)
        rc = RepresentativeCycles(min_persistence=0.3).fit(_circle())
        with pytest.raises(ImportError, match="pip install pandas"):
            rc.to_dataframe()
