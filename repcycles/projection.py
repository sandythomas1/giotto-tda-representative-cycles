"""Best-fit-plane projection for viewing a cycle inside a ≥3-D point cloud.

Spec 001, requirement **V4**.  Every view in this library used to flatten a
3-D cloud with ``X[:, :2]``, silently discarding the third coordinate.  On a
600-point torus that costs a tube-loop half its variance — the loop is drawn
as a near-degenerate smear and a reader dismisses it as noise.

This module fits the plane that is *optimal* for one cycle and then applies
that plane to the **whole** cloud, so the loop is shown in its surrounding
context rather than floating in isolation.

Why this is not a heuristic
---------------------------
The plane is the rank-2 truncated SVD of the cycle's centred vertices.  By the
Eckart-Young-Mirsky theorem it minimises the squared distance from those
vertices to any 2-flat, so::

    variance_retained >= baseline_retained

is a *theorem*, not a tuned threshold: the first two coordinate axes are one
particular 2-flat, and the best-fit plane is by definition no worse than it.
The absolute figure (≥ 0.90 on the torus benchmark) is, by contrast, an
empirical property of that fixture — a benchmark, not a guarantee.

Degenerate inputs never raise.  A cycle with fewer than three vertices, or one
whose vertices are collinear, has no well-defined plane; such calls fall back
to the first two coordinates, set ``is_degenerate=True``, and report the
variance the fallback *actually* retains.  Per ``specs/constitution.md``, a
degraded result is flagged in the returned structure and never dressed up as
a perfect one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["Projection", "project_for_cycle", "DEGENERACY_RATIO"]


#: A cycle is treated as rank-deficient (collinear) when its second singular
#: value falls below this fraction of its first.  Exact collinearity gives a
#: ratio at float round-off (~1e-16); a genuinely thin but two-dimensional
#: loop sits many orders of magnitude above this, so the test separates the
#: two cases cleanly without catching real loops.
DEGENERACY_RATIO: float = 1e-8


@dataclass(frozen=True)
class Projection:
    """A 2-D view of a point cloud, chosen to suit one representative cycle.

    Attributes
    ----------
    coords : np.ndarray, shape (n_points, 2)
        Projected coordinates for **every** point of the original cloud, in
        the original row order — not just the cycle's vertices.  In the
        best-fit-plane case these are centred on the cycle's centroid (the
        plane passes through it); in the fallback and 2-D pass-through cases
        they are the input's own first two coordinates, uncentred.
    variance_retained : float
        Fraction of the *cycle's* variance surviving this projection, in
        ``[0, 1]``.  ``1.0`` means the cycle is exactly planar in this view.
        For a degenerate cycle this is the fallback's true value, not ``1.0``.
    baseline_retained : float
        The same fraction under the naive ``X[:, :2]`` projection, so callers
        can quote the improvement.  Never greater than ``variance_retained``.
    is_degenerate : bool
        ``True`` when no plane could be fitted and the first two coordinates
        were used instead: fewer than three cycle vertices, collinear
        vertices, a near-zero second singular value, or a cloud with fewer
        than two dimensions.
    """

    coords: np.ndarray
    variance_retained: float
    baseline_retained: float
    is_degenerate: bool


def project_for_cycle(
    X: np.ndarray, cycle_vertices: np.ndarray
) -> Projection:
    """Project *X* onto the best-fit plane through one cycle's vertices.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, n_dims)
        The full point cloud.  ``n_dims <= 2`` passes through unchanged.
    cycle_vertices : np.ndarray
        Integer row indices into *X* naming the cycle's vertices.  Duplicates
        are ignored, so a closed ``cycle_path`` (whose first vertex is
        repeated at the end) yields the same plane as the deduplicated
        ``cycle_vertices``, and vertex order never affects the result.

    Returns
    -------
    Projection

    Raises
    ------
    ValueError
        *X* is not a 2-D array with at least one column, contains NaN or
        infinity, or *cycle_vertices* is not an integer array.
    IndexError
        A cycle vertex is out of range for *X*.

    Notes
    -----
    Cost is ``O(n_cycle * n_dims^2)`` for the SVD plus one ``O(n_points *
    n_dims)`` matrix product — the fit is over the cycle only, never the
    whole cloud, so this stays cheap for large clouds.

    The SVD fixes each basis vector's sign by forcing its largest-magnitude
    component positive.  Bare singular vectors carry an arbitrary sign, so
    without that convention two calls could return mirror-image coordinates.
    """
    X = _validate_cloud(X)
    n_points, n_dims = X.shape
    vertices = _validate_vertices(cycle_vertices, n_points)

    if n_dims < 2:
        # Nothing to choose between: pad to two columns so callers always get
        # a (n, 2) array, and flag it as no real plane.
        coords = np.column_stack([X[:, 0], np.zeros(n_points)])
        return Projection(coords, 1.0, 1.0, True)

    if n_dims == 2:
        return Projection(np.array(X, copy=True), 1.0, 1.0, False)

    centred, total_variance, baseline = _cycle_spread(X[vertices])

    if len(vertices) < 3 or total_variance == 0.0:
        return _fallback(X, baseline)

    # Rows of Vt are the principal directions of the cycle, ordered by the
    # variance they explain.
    singular_values, basis = _principal_plane(centred)
    if singular_values[1] <= DEGENERACY_RATIO * singular_values[0]:
        # Collinear (or near enough): the second direction would be noise.
        return _fallback(X, baseline)

    retained = float(singular_values[0] ** 2 + singular_values[1] ** 2)
    retained = _as_fraction(retained / total_variance)

    centroid = X[vertices].mean(axis=0)
    coords = (X - centroid) @ basis.T
    return Projection(coords, retained, baseline, False)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _fallback(X: np.ndarray, baseline: float) -> Projection:
    """First-two-coordinates view, reporting what it truly retains."""
    return Projection(
        coords=np.array(X[:, :2], copy=True),
        variance_retained=baseline,
        baseline_retained=baseline,
        is_degenerate=True,
    )


def _cycle_spread(V: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return the centred cycle vertices, their total variance, and the
    fraction of it that the naive ``X[:, :2]`` projection keeps.

    A cycle with no spread (identical or too few vertices) loses nothing under
    any projection, so its retained fraction is defined as ``1.0``.  Variance
    is reported as a sum of squares; the ``1/n`` factor cancels in the ratio.
    """
    if V.shape[0] == 0:
        return V, 0.0, 1.0

    centred = V - V.mean(axis=0)
    total = float((centred**2).sum())
    if total == 0.0:
        return centred, 0.0, 1.0

    baseline = _as_fraction(float((centred[:, :2] ** 2).sum()) / total)
    return centred, total, baseline


def _principal_plane(centred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Singular values and the sign-normalised top-2 right singular vectors."""
    _, singular_values, right_vectors = np.linalg.svd(
        centred, full_matrices=False
    )
    basis = np.array(right_vectors[:2], copy=True)

    # Deterministic sign: the largest-magnitude component of each basis
    # vector is made positive.  argmax takes the first maximum, so exact ties
    # resolve the same way on every call.
    pivots = np.argmax(np.abs(basis), axis=1)
    signs = np.sign(basis[np.arange(basis.shape[0]), pivots])
    signs[signs == 0.0] = 1.0
    return singular_values, basis * signs[:, np.newaxis]


def _as_fraction(value: float) -> float:
    """Clamp a variance ratio into ``[0, 1]`` against round-off overshoot."""
    return float(min(1.0, max(0.0, value)))


def _validate_cloud(X: np.ndarray) -> np.ndarray:
    """Return *X* as a float64 ``(n_points, n_dims)`` array.

    Non-finite coordinates are rejected here rather than propagated: a single
    NaN would silently turn the entire projection into NaN, which is exactly
    the kind of quiet wrong answer the constitution forbids.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D (n_points, n_dims); got shape {X.shape}."
        )
    if X.shape[1] < 1:
        raise ValueError("X must have at least one column.")
    if not np.isfinite(X).all():
        row, col = (int(i) for i in np.argwhere(~np.isfinite(X))[0])
        raise ValueError(
            f"X must contain only finite coordinates; found "
            f"{X[row, col]} at row {row}, column {col}."
        )
    return X


def _validate_vertices(
    cycle_vertices: np.ndarray, n_points: int
) -> np.ndarray:
    """Return unique, in-range integer indices into a cloud of *n_points*."""
    vertices = np.asarray(cycle_vertices)
    if vertices.size == 0:
        return np.empty(0, dtype=np.intp)
    if vertices.dtype.kind != "i" and vertices.dtype.kind != "u":
        # Rejected rather than cast: a bool array would otherwise be read as
        # the indices [0, 1] instead of as a mask, and a float array hides a
        # caller bug.
        raise ValueError(
            "cycle_vertices must be an integer array of row indices into X; "
            f"got dtype {vertices.dtype}."
        )

    vertices = np.unique(vertices.ravel())
    if vertices[0] < 0 or vertices[-1] >= n_points:
        offender = vertices[0] if vertices[0] < 0 else vertices[-1]
        raise IndexError(
            f"cycle vertex {int(offender)} is out of range for a cloud of "
            f"{n_points} points."
        )
    return vertices
