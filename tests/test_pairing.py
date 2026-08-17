"""Tests for diagram <-> generator pairing (spec 001 T4, F4/F4a/F4b, AC4).

AC4 requires this be exercised as a **pure function on synthetic arrays**: no
real dataset can be coerced into producing a generator that matches no diagram
row, so the failure path is only reachable by constructing the arrays by hand.
``pair_generators`` reads nothing from ``dist_matrix`` except ``D[u, v]`` for
birth edges, so the synthetic matrices below need not be valid metrics.

One real fit is included at the end for realism: it checks that the pairing a
live ``ripser_parallel`` result produces agrees with the birth-edge lengths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from repcycles.errors import CycleReconstructionError
from repcycles.pairing import (
    BIRTH_ATOL,
    BIRTH_RTOL,
    ESSENTIAL_DEATH_EDGE,
    PairedRow,
    pair_generators,
)

NO_EDGE = 99.0  # filler for pairs no test cares about


def dist_matrix_with(n_points: int, lengths: dict) -> np.ndarray:
    """Symmetric zero-diagonal matrix with the given ``(u, v) -> length`` pairs."""
    D = np.full((n_points, n_points), NO_EDGE, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for (u, v), length in lengths.items():
        D[u, v] = D[v, u] = length
    return D


def finite_gens(*rows) -> np.ndarray:
    """``(k, 4)`` array of ``[b_v0, b_v1, d_v0, d_v1]``."""
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def essential_gens(*rows) -> np.ndarray:
    """``(m, 2)`` array of ``[b_v0, b_v1]``."""
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


class TestUnmatchedGenerator:
    """AC4a — a generator matching no diagram row is a hard error."""

    def test_no_matching_row_raises(self):
        D = dist_matrix_with(4, {(0, 1): 2.0})
        diagram = np.array([[1.0, 3.0]])

        with pytest.raises(CycleReconstructionError):
            pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)

    def test_error_message_names_generator_length_and_candidates(self):
        D = dist_matrix_with(4, {(0, 1): 2.0})
        diagram = np.array([[1.0, 3.0], [4.0, 5.0]])

        with pytest.raises(CycleReconstructionError) as excinfo:
            pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)

        message = str(excinfo.value)
        assert "generator 0" in message.lower()
        assert "(0, 1)" in message  # the birth edge
        assert "2.0" in message  # the birth-edge length
        assert "1.0" in message and "3.0" in message  # candidate row 0
        assert "4.0" in message and "5.0" in message  # candidate row 1

    def test_no_feature_is_emitted_when_one_generator_fails(self):
        """F4b: a partial, possibly-mispaired result is never returned."""
        D = dist_matrix_with(6, {(0, 1): 1.0, (2, 3): 7.0})
        diagram = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(CycleReconstructionError):
            pair_generators(
                finite_gens([0, 1, 4, 5], [2, 3, 4, 5]),
                essential_gens(),
                diagram,
                D,
            )

    def test_more_generators_than_diagram_rows_raises(self):
        D = dist_matrix_with(8, {(0, 1): 1.0, (2, 3): 1.0, (4, 5): 1.0})
        diagram = np.array([[1.0, 2.0], [1.0, 2.0]])

        with pytest.raises(CycleReconstructionError) as excinfo:
            pair_generators(
                finite_gens([0, 1, 6, 7], [2, 3, 6, 7], [4, 5, 6, 7]),
                essential_gens(),
                diagram,
                D,
            )

        assert "generator 2" in str(excinfo.value).lower()

    def test_error_reports_that_every_eligible_row_was_claimed(self):
        D = dist_matrix_with(6, {(0, 1): 1.0, (2, 3): 1.0})
        diagram = np.array([[1.0, 2.0]])

        with pytest.raises(CycleReconstructionError) as excinfo:
            pair_generators(
                finite_gens([0, 1, 4, 5], [2, 3, 4, 5]),
                essential_gens(),
                diagram,
                D,
            )

        assert "0 of 1" in str(excinfo.value)


class TestCongruentFeatures:
    """AC4b — the measured trap: three congruent hexagonal holes give three
    diagram rows identical in *both* birth and death, so death-value
    disambiguation disambiguates nothing.  Such rows are interchangeable and
    this must not be an error."""

    CONGRUENT_DIAGRAM = np.array(
        [
            [1.0, 1.7320508],
            [1.0, 1.7320508],
            [1.0, 1.7320508],
        ]
    )

    def _three_holes(self):
        D = dist_matrix_with(12, {(0, 1): 1.0, (2, 3): 1.0, (4, 5): 1.0})
        gens = finite_gens([0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11])
        return gens, D

    def test_three_congruent_holes_pair_without_error(self):
        gens, D = self._three_holes()
        paired = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        assert len(paired) == 3

    def test_each_congruent_hole_claims_a_distinct_row(self):
        gens, D = self._three_holes()
        paired = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        assert sorted(p.diagram_index for p in paired) == [0, 1, 2]

    def test_lowest_unclaimed_row_is_taken_in_generator_order(self):
        """F4a fixes the assignment so repeated runs agree, even though any
        assignment among identical rows is equally correct."""
        gens, D = self._three_holes()
        paired = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        assert [p.diagram_index for p in paired] == [0, 1, 2]

    def test_every_hole_keeps_its_own_birth_edge(self):
        gens, D = self._three_holes()
        paired = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        assert [p.birth_edge for p in paired] == [(0, 1), (2, 3), (4, 5)]
        assert [p.death_edge for p in paired] == [(6, 7), (8, 9), (10, 11)]

    def test_pairing_is_deterministic_across_calls(self):
        gens, D = self._three_holes()
        first = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        second = pair_generators(gens, essential_gens(), self.CONGRUENT_DIAGRAM, D)
        assert first == second


class TestOutOfOrderGenerators:
    """AC4c — the whole point of by-value pairing: generator row i need not be
    diagram row i."""

    DIAGRAM = np.array([[0.5, 0.8], [1.0, 1.5], [2.0, 3.0]])

    def _reversed_gens(self):
        D = dist_matrix_with(8, {(0, 1): 2.0, (2, 3): 0.5, (4, 5): 1.0})
        gens = finite_gens([0, 1, 6, 7], [2, 3, 6, 7], [4, 5, 6, 7])
        return gens, D

    def test_each_generator_pairs_to_its_own_row(self):
        gens, D = self._reversed_gens()
        paired = pair_generators(gens, essential_gens(), self.DIAGRAM, D)
        assert [p.diagram_index for p in paired] == [2, 0, 1]

    def test_birth_and_death_come_from_the_claimed_row(self):
        gens, D = self._reversed_gens()
        paired = pair_generators(gens, essential_gens(), self.DIAGRAM, D)
        assert [p.birth for p in paired] == pytest.approx([2.0, 0.5, 1.0])
        assert [p.death for p in paired] == pytest.approx([3.0, 0.8, 1.5])

    def test_generator_index_is_the_input_row_not_the_diagram_row(self):
        gens, D = self._reversed_gens()
        paired = pair_generators(gens, essential_gens(), self.DIAGRAM, D)
        assert [p.generator_index for p in paired] == [0, 1, 2]

    def test_positional_pairing_would_have_been_wrong(self):
        """Guards the regression this task exists to prevent: reading
        ``diagram[i]`` for generator ``i`` mislabels every feature here."""
        gens, D = self._reversed_gens()
        paired = pair_generators(gens, essential_gens(), self.DIAGRAM, D)
        positional = [float(self.DIAGRAM[i, 0]) for i in range(len(gens))]
        assert [p.birth for p in paired] != pytest.approx(positional)


class TestEssentialGenerators:
    """Essential classes carry a birth edge only and pair with infinite-death
    rows."""

    DIAGRAM = np.array([[1.0, 2.0], [0.5, np.inf], [1.5, 3.0]])

    def _mixed(self):
        D = dist_matrix_with(8, {(0, 1): 1.5, (2, 3): 1.0, (4, 5): 0.5})
        return finite_gens([0, 1, 6, 7], [2, 3, 6, 7]), essential_gens([4, 5]), D

    def test_finite_and_essential_generators_pair_together(self):
        fin, ess, D = self._mixed()
        paired = pair_generators(fin, ess, self.DIAGRAM, D)
        assert [p.diagram_index for p in paired] == [2, 0, 1]

    def test_essential_rows_are_returned_after_finite_rows(self):
        fin, ess, D = self._mixed()
        paired = pair_generators(fin, ess, self.DIAGRAM, D)
        assert [p.is_essential for p in paired] == [False, False, True]

    def test_essential_row_carries_infinite_death_and_sentinel_edge(self):
        fin, ess, D = self._mixed()
        essential = pair_generators(fin, ess, self.DIAGRAM, D)[-1]
        assert np.isinf(essential.death)
        assert essential.death_edge == ESSENTIAL_DEATH_EDGE == (-1, -1)
        assert essential.birth_edge == (4, 5)
        assert essential.birth == pytest.approx(0.5)

    def test_finite_generator_cannot_claim_an_infinite_row(self):
        """A finite generator has a death simplex; an infinite bar has no
        death.  Matching on birth alone would happily confuse them."""
        D = dist_matrix_with(4, {(0, 1): 1.0})
        diagram = np.array([[1.0, np.inf]])

        with pytest.raises(CycleReconstructionError):
            pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)

    def test_essential_generator_cannot_claim_a_finite_row(self):
        D = dist_matrix_with(4, {(0, 1): 1.0})
        diagram = np.array([[1.0, 2.0]])

        with pytest.raises(CycleReconstructionError):
            pair_generators(finite_gens(), essential_gens([0, 1]), diagram, D)

    def test_two_essential_generators_claim_distinct_rows(self):
        D = dist_matrix_with(4, {(0, 1): 1.0, (2, 3): 1.0})
        diagram = np.array([[1.0, np.inf], [1.0, np.inf]])
        paired = pair_generators(
            finite_gens(), essential_gens([0, 1], [2, 3]), diagram, D
        )
        assert sorted(p.diagram_index for p in paired) == [0, 1]
        assert all(p.is_essential for p in paired)


class TestFloat32Tolerance:
    """ripser computes the filtration in float32; our distance matrix is
    float64.  The diagram's birth value is therefore a float32-rounded copy of
    ``D[u, v]`` and exact equality never holds.  ``rtol=1e-5`` is float32
    epsilon (~1.19e-7) with headroom; ``atol=1e-9`` covers births near zero."""

    def test_float32_rounded_birth_still_matches(self):
        exact = math.sqrt(2.0)  # 1.4142135623730951 in float64
        as_float32 = float(np.float32(exact))  # 1.4142135381698608

        assert as_float32 != exact, "fixture assumes float32 loses precision"

        D = dist_matrix_with(4, {(0, 1): exact})
        diagram = np.array([[as_float32, 3.0]])

        paired = pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)
        assert paired[0].diagram_index == 0

    def test_exact_equality_would_have_rejected_the_same_row(self):
        """Documents *why* a tolerance is required rather than ``==``."""
        exact = math.sqrt(2.0)
        assert float(np.float32(exact)) != exact

    def test_float32_epsilon_is_comfortably_inside_the_tolerance(self):
        exact = math.sqrt(2.0)
        relative_error = abs(float(np.float32(exact)) - exact) / exact
        assert relative_error < BIRTH_RTOL

    def test_zero_birth_is_covered_by_the_absolute_tolerance(self):
        D = dist_matrix_with(4, {(0, 1): 0.0})
        diagram = np.array([[BIRTH_ATOL / 2, 1.0]])
        paired = pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)
        assert paired[0].diagram_index == 0

    def test_tolerance_is_still_tight_enough_to_reject_a_real_mismatch(self):
        """A genuinely different feature is ~1e-3 away, not ~1e-7."""
        D = dist_matrix_with(4, {(0, 1): 1.0})
        diagram = np.array([[1.001, 2.0]])

        with pytest.raises(CycleReconstructionError):
            pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)

    def test_tolerances_are_configurable(self):
        D = dist_matrix_with(4, {(0, 1): 1.0})
        diagram = np.array([[1.001, 2.0]])
        paired = pair_generators(
            finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D, rtol=1e-2
        )
        assert paired[0].diagram_index == 0


class TestEmptyAndDegenerateInputs:

    def test_no_generators_and_no_diagram(self):
        assert pair_generators(
            finite_gens(), essential_gens(), np.empty((0, 2)), np.zeros((3, 3))
        ) == []

    def test_no_generators_with_a_populated_diagram(self):
        diagram = np.array([[1.0, 2.0], [0.5, np.inf]])
        assert pair_generators(
            finite_gens(), essential_gens(), diagram, np.zeros((3, 3))
        ) == []

    def test_unclaimed_diagram_rows_are_simply_absent(self):
        D = dist_matrix_with(4, {(0, 1): 1.0})
        diagram = np.array([[1.0, 2.0], [5.0, 6.0]])
        paired = pair_generators(finite_gens([0, 1, 2, 3]), essential_gens(), diagram, D)
        assert [p.diagram_index for p in paired] == [0]

    def test_ripser_style_empty_arrays_are_accepted(self):
        """``gens[1]`` can be an empty list, yielding an array of shape (0,)."""
        assert pair_generators(
            np.array([]), np.array([]), np.empty((0, 2)), np.zeros((2, 2))
        ) == []

    def test_none_generators_are_treated_as_empty(self):
        assert pair_generators(None, None, np.empty((0, 2)), np.zeros((2, 2))) == []


class TestInputValidation:

    def test_wrong_finite_generator_width_rejected(self):
        with pytest.raises(ValueError, match=r"finite_gens must have shape"):
            pair_generators(
                np.array([[0, 1, 2]]), essential_gens(), np.array([[1.0, 2.0]]),
                np.zeros((4, 4)),
            )

    def test_wrong_essential_generator_width_rejected(self):
        with pytest.raises(ValueError, match=r"essential_gens must have shape"):
            pair_generators(
                finite_gens(), np.array([[0, 1, 2, 3]]),
                np.array([[1.0, np.inf]]), np.zeros((4, 4)),
            )

    def test_wrong_diagram_width_rejected(self):
        with pytest.raises(ValueError, match="diagram must have shape"):
            pair_generators(
                finite_gens(), essential_gens(),
                np.array([[1.0, 2.0, 3.0]]), np.zeros((4, 4)),
            )

    def test_non_square_distance_matrix_rejected(self):
        with pytest.raises(ValueError, match="square"):
            pair_generators(
                finite_gens(), essential_gens(),
                np.array([[1.0, 2.0]]), np.zeros((4, 5)),
            )

    def test_vertex_index_outside_the_distance_matrix_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            pair_generators(
                finite_gens([0, 9, 1, 2]), essential_gens(),
                np.array([[1.0, 2.0]]), np.zeros((4, 4)),
            )

    def test_nan_in_the_diagram_rejected_by_row_index(self):
        with pytest.raises(ValueError, match="NaN at row 1"):
            pair_generators(
                finite_gens(), essential_gens(),
                np.array([[1.0, 2.0], [np.nan, 3.0]]), np.zeros((4, 4)),
            )

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            pair_generators(
                finite_gens(), essential_gens(),
                np.array([[1.0, 2.0]]), np.zeros((4, 4)), rtol=-1.0,
            )


class TestPairedRowContract:

    def test_is_frozen(self):
        row = PairedRow(0, 0, 1.0, 2.0, (0, 1), (2, 3), False)
        with pytest.raises(Exception):
            row.birth = 5.0  # type: ignore[misc]

    def test_edges_are_plain_python_int_tuples(self):
        """Downstream code puts these in dict keys and error messages, so
        numpy scalars leaking through would be a nuisance."""
        D = dist_matrix_with(4, {(0, 1): 1.0})
        paired = pair_generators(
            finite_gens([0, 1, 2, 3]), essential_gens(),
            np.array([[1.0, 2.0]]), D,
        )[0]
        assert all(type(i) is int for i in paired.birth_edge + paired.death_edge)
        assert type(paired.birth) is float and type(paired.death) is float


class TestRealFit:
    """Realism check against a live ripser result.  The synthetic tests above
    are the ones that satisfy AC4; this one guards against the pure function
    disagreeing with the arrays ripser actually emits."""

    @staticmethod
    def _figure_eight(n_per_loop: int = 40) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_per_loop, endpoint=False)
        left = np.column_stack([np.cos(theta) - 1.0, np.sin(theta)])
        right = np.column_stack([np.cos(theta) + 1.0, np.sin(theta)])
        return np.vstack([left, right])

    @staticmethod
    def _circle(n_points: int = 60) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    @staticmethod
    def _fit(X: np.ndarray, thresh: float = np.inf):
        from gph import ripser_parallel
        from scipy.spatial.distance import cdist

        result = ripser_parallel(
            X.astype(np.float32),
            maxdim=1,
            thresh=thresh,
            coeff=2,
            return_generators=True,
        )
        gens = result["gens"]
        fin = gens[1][0] if len(gens[1]) > 0 else np.empty((0, 4), dtype=int)
        ess = gens[3][0] if len(gens[3]) > 0 else np.empty((0, 2), dtype=int)
        return result["dgms"][1], fin, ess, cdist(X, X)

    def test_every_paired_birth_equals_its_own_birth_edge_length(self):
        diagram, fin, ess, D = self._fit(self._figure_eight())
        assert len(fin) > 0, "fixture should produce finite H1 classes"

        for row in pair_generators(fin, ess, diagram, D):
            u, v = row.birth_edge
            assert row.birth == pytest.approx(D[u, v], rel=BIRTH_RTOL, abs=1e-6)

    def test_figure_eight_pairs_two_prominent_loops(self):
        diagram, fin, ess, D = self._fit(self._figure_eight())
        paired = pair_generators(fin, ess, diagram, D)

        persistences = sorted((p.death - p.birth for p in paired), reverse=True)
        assert len(persistences) >= 2
        assert persistences[1] > 0.5, "both loops of the figure-eight survive"

    def test_all_diagram_rows_are_claimed_exactly_once(self):
        diagram, fin, ess, D = self._fit(self._figure_eight())
        paired = pair_generators(fin, ess, diagram, D)

        claimed = [p.diagram_index for p in paired]
        assert len(claimed) == len(set(claimed))
        assert len(claimed) == len(diagram)

    def test_thresholded_circle_pairs_an_essential_class(self):
        """The AC1 fixture: 60-point circle at max_edge_length=1.0, whose only
        loop lives in ``gens[3]``."""
        diagram, fin, ess, D = self._fit(self._circle(), thresh=1.0)
        assert len(ess) == 1, "fixture should produce one essential H1 class"

        paired = pair_generators(fin, ess, diagram, D)
        essential = [p for p in paired if p.is_essential]
        assert len(essential) == 1
        u, v = essential[0].birth_edge
        assert essential[0].birth == pytest.approx(D[u, v], rel=BIRTH_RTOL, abs=1e-6)
        assert np.isinf(essential[0].death)
        assert essential[0].death_edge == (-1, -1)

    def test_real_fit_pairing_is_deterministic(self):
        diagram, fin, ess, D = self._fit(self._figure_eight())
        assert pair_generators(fin, ess, diagram, D) == pair_generators(
            fin, ess, diagram, D
        )
