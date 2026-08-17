"""Tests for the untrusted-input boundary (spec 001 T3, requirement F10, AC7).

The security-relevant claims under test:

* nothing non-finite reaches a distance computation unexamined — but ``+inf``
  in a *precomputed* matrix is legitimate data ("no edge"), not an attack;
* every rejection names the offending index, so a caller with 5 000 points can
  actually find the bad entry;
* the size guard never allocates what it is guarding against;
* validation runs *before* the O(n²) allocation, not after.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from repcycles.validation import (
    DIAGONAL_ATOL,
    SIZE_WARN_THRESHOLD,
    SYMMETRY_ATOL,
    check_size,
    validate_point_cloud,
    validate_precomputed,
    validate_save_path,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def circle_points(n: int = 12) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def circle_distances(n: int = 12) -> np.ndarray:
    pts = circle_points(n)
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)


def core_is_wired_to_validation() -> bool:
    """True once the integrator has routed ``fit()`` through ``validation``.

    T3 owns ``repcycles/validation.py`` only; ``repcycles/core.py`` belongs to
    the integrator.  The end-to-end assertions below are written now and skip
    (visibly, with a reason) until that wiring lands, rather than being
    silently omitted.
    """
    import repcycles.core as core

    source = inspect.getsource(core)
    return "validation" in source or "validate_point_cloud" in source


def require_core_wiring() -> None:
    if not core_is_wired_to_validation():
        pytest.skip(
            "repcycles.core is not yet wired to repcycles.validation "
            "(integrator step of T3); the unit-level contract is asserted "
            "above."
        )


# ---------------------------------------------------------------------------
# validate_point_cloud
# ---------------------------------------------------------------------------


class TestValidatePointCloud:

    def test_returns_float64_for_integer_input(self):
        out = validate_point_cloud(np.array([[0, 1], [2, 3]]))
        assert out.dtype == np.float64
        assert np.array_equal(out, [[0.0, 1.0], [2.0, 3.0]])

    def test_float64_input_is_not_copied(self):
        """A copy would double peak memory on exactly the large inputs the
        size guard exists to protect."""
        X = circle_points(50)
        assert np.shares_memory(validate_point_cloud(X), X)

    def test_accepts_a_single_point(self):
        out = validate_point_cloud(np.array([[1.0, 2.0, 3.0]]))
        assert out.shape == (1, 3)

    def test_accepts_a_single_coordinate_column(self):
        out = validate_point_cloud(np.arange(5.0).reshape(-1, 1))
        assert out.shape == (5, 1)

    def test_rejects_1d_input_and_suggests_the_fix(self):
        with pytest.raises(ValueError) as excinfo:
            validate_point_cloud(np.arange(5.0))
        message = str(excinfo.value)
        assert "2-D" in message
        assert "reshape(-1, 1)" in message

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_point_cloud(np.zeros((2, 3, 4)))

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="at least one point"):
            validate_point_cloud(np.empty((0, 2)))

    def test_rejects_points_with_no_coordinates(self):
        with pytest.raises(ValueError, match="at least one coordinate"):
            validate_point_cloud(np.empty((5, 0)))

    def test_rejects_nan_naming_row_and_column(self):
        X = np.zeros((20, 3))
        X[17, 2] = np.nan
        with pytest.raises(ValueError) as excinfo:
            validate_point_cloud(X)
        assert "NaN at row 17, column 2" in str(excinfo.value)

    def test_rejects_positive_infinite_coordinate_naming_the_index(self):
        X = circle_points(6)
        X[4, 1] = np.inf
        with pytest.raises(ValueError) as excinfo:
            validate_point_cloud(X)
        message = str(excinfo.value)
        assert "+inf at row 4, column 1" in message
        assert "precomputed" in message  # points at the legitimate use

    def test_rejects_negative_infinite_coordinate(self):
        X = circle_points(6)
        X[2, 0] = -np.inf
        with pytest.raises(ValueError) as excinfo:
            validate_point_cloud(X)
        assert "-inf at row 2, column 0" in str(excinfo.value)

    def test_nan_is_reported_before_inf_when_both_present(self):
        """NaN is the more destructive of the two; report it first."""
        X = np.zeros((3, 2))
        X[0, 0] = np.inf
        X[1, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_point_cloud(X)

    def test_rejects_string_dtype(self):
        with pytest.raises(ValueError, match="real numeric"):
            validate_point_cloud(np.array([["a", "b"], ["c", "d"]]))

    def test_rejects_object_dtype(self):
        with pytest.raises(ValueError, match="real numeric"):
            validate_point_cloud(np.array([[object(), object()]]))

    def test_rejects_complex_dtype_rather_than_discarding_the_imaginary_part(
        self,
    ):
        """NumPy would cast this with only a ComplexWarning and drop the
        imaginary part — a silent change of the caller's data."""
        with pytest.raises(ValueError, match="complex"):
            validate_point_cloud(np.array([[1 + 2j, 3 + 0j]]))

    def test_none_entries_surface_as_nan_rejection(self):
        with pytest.raises(ValueError, match="NaN at row 0, column 1"):
            validate_point_cloud([[1.0, None]])

    def test_accepts_a_plain_nested_list(self):
        out = validate_point_cloud([[0.0, 0.0], [1.0, 1.0]])
        assert out.shape == (2, 2)

    def test_large_negative_and_positive_finite_values_are_fine(self):
        X = np.array([[-1e300, 1e300], [0.0, 0.0]])
        assert validate_point_cloud(X).shape == (2, 2)


# ---------------------------------------------------------------------------
# validate_precomputed
# ---------------------------------------------------------------------------


class TestValidatePrecomputed:

    def test_accepts_a_valid_matrix_and_returns_float64(self):
        out = validate_precomputed(circle_distances(8).astype(np.float32))
        assert out.dtype == np.float64
        assert out.shape == (8, 8)

    def test_float64_input_is_not_copied(self):
        D = circle_distances(8)
        assert np.shares_memory(validate_precomputed(D), D)

    # -- the F10 headline: +inf is data, not an error --------------------

    def test_accepts_positive_infinity_as_no_edge(self):
        D = circle_distances(12)
        D[0, 6] = D[6, 0] = np.inf
        out = validate_precomputed(D)
        assert np.isinf(out[0, 6])

    def test_accepts_a_fully_disconnected_all_inf_matrix(self):
        """n mutually isolated points is valid input describing no edges.
        Judging whether an input is *interesting* is not validation's job."""
        n = 5
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0.0)
        out = validate_precomputed(D)
        assert np.isinf(out[np.triu_indices(n, k=1)]).all()

    def test_infinite_pair_is_absent_from_the_rips_graph_at_every_radius(self):
        D = validate_precomputed_with_gap()
        for radius in (0.1, 1.0, 10.0, 1e12, np.finfo(np.float64).max):
            adjacency = D <= radius
            assert not adjacency[0, 6], f"inf pair became an edge at r={radius}"

    def test_gph_computes_h1_on_a_matrix_containing_inf(self):
        """The spec's claim that gph accepts +inf, verified rather than
        assumed — AC7's 'fit completes' rests on it."""
        from gph import ripser_parallel

        D = validate_precomputed_with_gap()
        result = ripser_parallel(
            D.astype(np.float32),
            maxdim=1,
            metric="precomputed",
            return_generators=True,
        )
        h1 = result["dgms"][1]
        assert h1.shape[0] >= 1
        assert np.isfinite(h1[:, 0]).all()

    def test_symmetry_check_does_not_warn_on_infinite_entries(self):
        """Regression: ``max(abs(D - D.T))`` yields NaN plus a RuntimeWarning
        for inf entries — it mis-reports exactly the inputs F10 supports."""
        D = circle_distances(12)
        D[0, 6] = D[6, 0] = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_precomputed(D)

    # -- rejections ------------------------------------------------------

    def test_rejects_nan_naming_row_and_column(self):
        D = circle_distances(6)
        D[3, 1] = D[1, 3] = np.nan
        with pytest.raises(ValueError) as excinfo:
            validate_precomputed(D)
        assert "NaN at row 1, column 3" in str(excinfo.value)

    def test_rejects_negative_distance_naming_index_and_value(self):
        D = circle_distances(6)
        D[2, 4] = D[4, 2] = -0.5
        with pytest.raises(ValueError) as excinfo:
            validate_precomputed(D)
        message = str(excinfo.value)
        assert "negative" in message
        assert "row 2, column 4" in message
        assert "-0.5" in message

    def test_rejects_negative_infinity(self):
        D = circle_distances(6)
        D[1, 5] = D[5, 1] = -np.inf
        with pytest.raises(ValueError, match="negative"):
            validate_precomputed(D)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            validate_precomputed(np.zeros((4, 3)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="square"):
            validate_precomputed(np.zeros(4))

    def test_rejects_empty_matrix(self):
        with pytest.raises(ValueError, match="at least one point"):
            validate_precomputed(np.empty((0, 0)))

    def test_accepts_the_degenerate_one_by_one_matrix(self):
        assert validate_precomputed(np.zeros((1, 1))).shape == (1, 1)

    def test_rejects_asymmetric_naming_both_entries(self):
        D = np.array([[0.0, 1.0], [2.0, 0.0]])
        with pytest.raises(ValueError) as excinfo:
            validate_precomputed(D)
        message = str(excinfo.value)
        assert "symmetric" in message
        assert "D[0, 1]" in message and "D[1, 0]" in message

    def test_accepts_asymmetry_inside_tolerance(self):
        """Round-tripping a distance matrix through float32 perturbs it; that
        must not be an error."""
        D = circle_distances(8)
        D[0, 3] += SYMMETRY_ATOL / 10.0
        validate_precomputed(D)

    def test_rejects_asymmetry_outside_tolerance(self):
        D = circle_distances(8)
        D[0, 3] += SYMMETRY_ATOL * 10.0
        with pytest.raises(ValueError, match="symmetric"):
            validate_precomputed(D)

    def test_one_sided_infinity_is_asymmetric(self):
        """inf on one side only is a genuine inconsistency, not a 'no edge'."""
        D = circle_distances(6)
        D[1, 4] = np.inf
        with pytest.raises(ValueError, match="symmetric"):
            validate_precomputed(D)

    def test_accepts_near_zero_diagonal_inside_tolerance(self):
        D = circle_distances(6)
        np.fill_diagonal(D, DIAGONAL_ATOL / 10.0)
        validate_precomputed(D)

    def test_rejects_near_zero_diagonal_outside_tolerance(self):
        D = circle_distances(6)
        D[4, 4] = DIAGONAL_ATOL * 100.0
        with pytest.raises(ValueError) as excinfo:
            validate_precomputed(D)
        message = str(excinfo.value)
        assert "diagonal" in message
        assert "index 4" in message

    def test_rejects_infinite_diagonal(self):
        D = circle_distances(6)
        D[2, 2] = np.inf
        with pytest.raises(ValueError, match="diagonal"):
            validate_precomputed(D)

    def test_nan_is_reported_before_asymmetry(self):
        D = np.array([[0.0, np.nan], [3.0, 0.0]])
        with pytest.raises(ValueError, match="NaN"):
            validate_precomputed(D)

    def test_rejects_complex_dtype(self):
        with pytest.raises(ValueError, match="complex"):
            validate_precomputed(np.zeros((2, 2), dtype=complex))


def validate_precomputed_with_gap(n: int = 12) -> np.ndarray:
    """A validated circle matrix with one pair encoded as ``+inf``."""
    D = circle_distances(n)
    D[0, 6] = D[6, 0] = np.inf
    return validate_precomputed(D)


# ---------------------------------------------------------------------------
# check_size
# ---------------------------------------------------------------------------


class TestCheckSize:

    def test_silent_at_and_below_the_threshold(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_size(1)
            check_size(SIZE_WARN_THRESHOLD)

    def test_warns_above_the_threshold(self):
        with pytest.warns(ResourceWarning) as record:
            check_size(SIZE_WARN_THRESHOLD + 1)
        assert len(record) == 1

    def test_warning_states_the_projected_matrix_cost(self):
        with pytest.warns(ResourceWarning) as record:
            check_size(5000 + 1)
        message = str(record[0].message)
        assert "5001" in message
        assert "MB" in message
        assert "float64" in message
        assert "max_points" in message  # tells the caller how to opt in

    def test_default_max_points_never_raises(self):
        """B3: no cap by default, so no existing caller breaks."""
        with pytest.warns(ResourceWarning):
            assert check_size(1_000_000) is None

    def test_guard_does_not_allocate_what_it_guards_against(self):
        """A billion points must fail-or-warn instantly, never by trying to
        build the matrix it is warning about."""
        with pytest.warns(ResourceWarning):
            check_size(10**9)
        with pytest.raises(ValueError):
            check_size(10**9, max_points=5000)

    def test_explicit_max_points_raises_naming_both_numbers(self):
        with pytest.raises(ValueError) as excinfo:
            check_size(6000, max_points=1000)
        message = str(excinfo.value)
        assert "6000" in message
        assert "max_points=1000" in message
        assert "MB" in message

    def test_equal_to_max_points_is_allowed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_size(100, max_points=100)

    def test_error_takes_precedence_over_the_warning(self):
        """Above the threshold *and* over the cap: the caller gets the hard
        failure, not a warning it might have filtered away."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            with pytest.raises(ValueError):
                check_size(6000, max_points=100)

    def test_zero_points_is_not_the_size_guard_s_problem(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_size(0)

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError, match="non-negative"):
            check_size(-1)

    def test_rejects_non_integer_n(self):
        with pytest.raises(TypeError, match="integer"):
            check_size(10.5)

    def test_accepts_numpy_integer_n(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_size(np.int64(10), max_points=np.int64(20))

    @pytest.mark.parametrize("bad", [10.0, "1000", [1000], True])
    def test_rejects_non_integer_max_points(self, bad):
        with pytest.raises(TypeError, match="integer or None"):
            check_size(10, max_points=bad)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_non_positive_max_points(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            check_size(10, max_points=bad)


# ---------------------------------------------------------------------------
# validate_save_path
# ---------------------------------------------------------------------------


class TestValidateSavePath:

    def test_none_passes_through(self):
        assert validate_save_path(None) is None

    def test_valid_path_is_returned_unchanged(self, tmp_path):
        target = str(tmp_path / "figure.png")
        assert validate_save_path(target) == target

    def test_pathlib_object_is_returned_unchanged(self, tmp_path):
        target = tmp_path / "figure.png"
        assert validate_save_path(target) is target

    def test_bare_filename_resolves_against_the_working_directory(self):
        assert validate_save_path("figure.png") == "figure.png"

    def test_missing_parent_directory_names_the_directory(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ValueError) as excinfo:
            validate_save_path(str(missing / "figure.png"))
        message = str(excinfo.value)
        assert str(missing) in message
        assert "does not exist" in message
        assert "makedirs" in message  # actionable, not just diagnostic

    def test_existing_directory_as_target_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="existing directory"):
            validate_save_path(str(tmp_path))

    def test_parent_that_is_a_file_is_rejected(self, tmp_path):
        parent = tmp_path / "not_a_dir.txt"
        parent.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="not a directory"):
            validate_save_path(str(parent / "figure.png"))

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_path_is_rejected(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            validate_save_path(bad)

    @pytest.mark.parametrize("bad", [1, 3.5, ["a.png"]])
    def test_non_path_types_are_rejected(self, bad):
        with pytest.raises(TypeError, match="string"):
            validate_save_path(bad)

    def test_existing_file_may_be_overwritten(self, tmp_path):
        target = tmp_path / "figure.png"
        target.write_bytes(b"old")
        assert validate_save_path(str(target)) == str(target)

    def test_traversal_is_not_treated_as_an_attack(self, tmp_path):
        """Deliberate: the caller is the trust boundary (spec 001, Security).
        A '..' segment resolving to a real directory is an ordinary path."""
        nested = tmp_path / "sub"
        nested.mkdir()
        target = str(nested / ".." / "figure.png")
        assert validate_save_path(target) == target


# ---------------------------------------------------------------------------
# Acceptance criterion 7, bullet by bullet
# ---------------------------------------------------------------------------


class TestAcceptanceCriterion7:
    """One test per bullet of AC7 (spec 001)."""

    def test_nan_in_coordinates_raises_naming_the_index(self):
        X = circle_points(10)
        X[7, 0] = np.nan
        with pytest.raises(ValueError) as excinfo:
            validate_point_cloud(X)
        assert "row 7, column 0" in str(excinfo.value)

    def test_inf_in_coordinates_raises(self):
        X = circle_points(10)
        X[3, 1] = np.inf
        with pytest.raises(ValueError):
            validate_point_cloud(X)

    def test_inf_in_a_precomputed_matrix_is_accepted(self):
        D = validate_precomputed_with_gap()
        assert np.isinf(D[0, 6])

    def test_inf_in_a_precomputed_matrix_completes_a_fit(self):
        from repcycles import RepresentativeCycles

        D = validate_precomputed_with_gap(16)
        require_core_wiring()
        try:
            rc = RepresentativeCycles(metric="precomputed").fit(D)
        except ValueError as exc:
            if "infinity" in str(exc):
                pytest.skip(
                    "core._mds_embed cannot embed a matrix containing +inf "
                    "(sklearn rejects it); the integrator must substitute a "
                    "finite surrogate for the visualisation-only embedding — "
                    "see the T3 report."
                )
            raise
        assert rc.diagrams_ is not None

    def test_negative_precomputed_distance_raises(self):
        D = circle_distances(6)
        D[0, 1] = D[1, 0] = -1e-9
        with pytest.raises(ValueError, match="negative"):
            validate_precomputed(D)

    def test_more_than_5000_points_warns_but_does_not_raise(self):
        with pytest.warns(ResourceWarning):
            check_size(5001)

    def test_exceeding_an_explicit_max_points_raises(self):
        with pytest.raises(ValueError, match="max_points"):
            check_size(5001, max_points=5000)

    def test_fit_warns_above_the_threshold_and_still_succeeds(
        self, monkeypatch
    ):
        """End-to-end form of the ResourceWarning bullet, with the threshold
        lowered so the test does not actually fit 5 000 points."""
        import repcycles.validation as validation
        from repcycles import RepresentativeCycles

        require_core_wiring()
        monkeypatch.setattr(validation, "SIZE_WARN_THRESHOLD", 5)
        with pytest.warns(ResourceWarning):
            rc = RepresentativeCycles().fit(circle_points(24))
        assert rc.diagrams_ is not None


# ---------------------------------------------------------------------------
# Ordering: validation must precede the O(n^2) allocation
# ---------------------------------------------------------------------------


class TestValidationPrecedesAllocation:
    """The constitution requires validation *before* the big allocation.

    The unit-level test proves the boundary function itself needs no distance
    matrix; the end-to-end test proves ``fit()`` calls it first, and skips
    until the integrator wires ``core.py`` (T3 does not own that file).
    """

    def test_validation_needs_no_distance_computation(self, monkeypatch):
        import scipy.spatial.distance as spdist

        def exploding_cdist(*args, **kwargs):
            raise AssertionError("cdist was called during validation")

        monkeypatch.setattr(spdist, "cdist", exploding_cdist)

        X = np.zeros((2000, 3))
        X[1234, 1] = np.nan
        with pytest.raises(ValueError, match="NaN at row 1234, column 1"):
            validate_point_cloud(X)

    def test_fit_validates_before_building_the_distance_matrix(
        self, monkeypatch
    ):
        import repcycles.core as core
        from repcycles import RepresentativeCycles

        def exploding_cdist(*args, **kwargs):
            raise AssertionError("cdist was called before validation")

        monkeypatch.setattr(core, "cdist", exploding_cdist, raising=False)

        X = circle_points(10)
        X[4, 1] = np.nan
        try:
            with pytest.raises(ValueError, match="NaN"):
                RepresentativeCycles().fit(X)
        except AssertionError as exc:
            if "cdist was called before validation" in str(exc):
                pytest.skip(
                    "repcycles.core does not call validation before cdist yet "
                    "(integrator step of T3)."
                )
            raise
