"""Tests for the shared contract: CycleFeature, errors, palette (spec 001 T2)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import to_hex

from repcycles.errors import (
    CycleReconstructionError,
    CycleReconstructionWarning,
    RepCyclesError,
)
from repcycles.feature import CycleFeature
from repcycles.palette import OKABE_ITO, cycle_colors


class TestCycleFeatureContract:
    """B4: new fields must not break existing positional construction."""

    def test_positional_construction_through_death_edge_still_works(self):
        f = CycleFeature(0, 0.5, 1.5, 1.0, (0, 1), (1, 2))
        assert f.birth_edge == (0, 1)
        assert f.death_edge == (1, 2)

    def test_new_fields_default_to_empty_and_false(self):
        f = CycleFeature(index=0, birth=0.0, death=1.0, persistence=1.0)
        assert f.cycle_length == 0.0
        assert len(f.cycle_path) == 0
        assert f.is_essential is False
        assert f.is_verified is False

    def test_n_edges_matches_cycle_edges(self):
        f = CycleFeature(
            index=0, birth=0.0, death=1.0, persistence=1.0,
            cycle_edges=np.array([[0, 1], [1, 2], [2, 0]]),
        )
        assert f.n_edges == 3

    def test_essential_feature_carries_infinite_death(self):
        f = CycleFeature(
            index=0, birth=0.2, death=np.inf, persistence=np.inf,
            is_essential=True,
        )
        assert np.isinf(f.death) and np.isinf(f.persistence)
        assert f.is_essential


class TestErrors:

    def test_reconstruction_error_is_a_package_error(self):
        assert issubclass(CycleReconstructionError, RepCyclesError)
        assert issubclass(RepCyclesError, Exception)

    def test_reconstruction_warning_is_a_user_warning(self):
        """Must be a UserWarning subclass so `warnings.simplefilter` and
        `pytest.warns` treat it conventionally."""
        assert issubclass(CycleReconstructionWarning, UserWarning)

    def test_warning_can_be_caught_by_category(self):
        import warnings

        with pytest.warns(CycleReconstructionWarning):
            warnings.warn("degraded", CycleReconstructionWarning)


class TestPalette:

    def test_returns_hex_strings(self):
        colors = cycle_colors(4)
        assert all(isinstance(c, str) and c.startswith("#") for c in colors)
        assert all(len(c) == 7 for c in colors)

    def test_length_matches_request(self):
        for n in (0, 1, 3, 6, 20):
            assert len(cycle_colors(n)) == n

    def test_accepts_a_feature_sequence(self):
        features = [CycleFeature(i, 0.0, 1.0, 1.0) for i in range(3)]
        assert cycle_colors(features) == cycle_colors(3)

    def test_deterministic(self):
        assert cycle_colors(5) == cycle_colors(5)

    def test_prefix_stable_as_count_grows(self):
        """The colour of cycle k must not change when more cycles appear —
        this is what makes colours comparable across figures from one fit."""
        assert cycle_colors(10)[:3] == cycle_colors(3)

    def test_repeats_rather_than_interpolates_beyond_palette(self):
        colors = cycle_colors(len(OKABE_ITO) + 2)
        assert colors[len(OKABE_ITO)] == colors[0]
        assert set(colors) <= set(OKABE_ITO)

    def test_colours_survive_matplotlib_normalisation(self):
        """V2: renderers convert from hex; comparison happens after to_hex."""
        for c in cycle_colors(6):
            assert to_hex(c) == c.lower()

    def test_palette_entries_are_distinct(self):
        assert len(set(OKABE_ITO)) == len(OKABE_ITO)

    def test_black_excluded_because_birth_edges_are_black(self):
        assert "#000000" not in {c.lower() for c in OKABE_ITO}

    def test_negative_count_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            cycle_colors(-1)
