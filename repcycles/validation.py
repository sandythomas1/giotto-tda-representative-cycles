"""Input validation — the untrusted-input boundary of the package (spec 001 F10).

Everything a caller hands to :meth:`RepresentativeCycles.fit` passes through
here *before* any O(n²) allocation happens.  Two reasons, both from
``specs/constitution.md``:

1. **Correctness.** ``NaN`` poisons every comparison it touches silently: a
   Rips filtration built on NaN distances produces a plausible-looking diagram
   that is meaningless.  Rejecting it at the boundary is the only place the
   caller can still be told *which* entry was wrong.
2. **Resource safety.** Fitting allocates a dense ``n × n`` float64 distance
   matrix.  Validating first means a bad 50 000-point input fails in
   microseconds instead of after a 20 GB allocation attempt.

Non-finite values are handled **by kind, not wholesale** (F10):

===========================  ==========================  ====================
Value                        Euclidean coordinates       Precomputed matrix
===========================  ==========================  ====================
``NaN``                      rejected                    rejected
``+inf``                     rejected                    **accepted**
``-inf``                     rejected                    rejected (negative)
negative finite              accepted (a coordinate)     rejected
===========================  ==========================  ====================

``+inf`` in a precomputed matrix is not an error, it is the standard encoding
for *"there is no edge between these two points"* — the usual shape of a
geodesic distance matrix over a disconnected mesh, which this project's README
advertises as a supported input.  Such a pair satisfies ``D[u, v] <= r`` for no
finite radius ``r``, so it is simply absent from the Rips graph at every scale.
``gph.ripser_parallel`` accepts it (verified against gph on this project's
pinned version).

Error messages name the offending index — ``"NaN at row 17, column 2"`` — never
just "invalid input"; a caller with a 5 000-point cloud cannot act on the
latter.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Union

import numpy as np

__all__ = [
    "SIZE_WARN_THRESHOLD",
    "SYMMETRY_ATOL",
    "DIAGONAL_ATOL",
    "validate_point_cloud",
    "validate_precomputed",
    "check_size",
    "validate_save_path",
]

#: Above this many points, :func:`check_size` emits a :class:`ResourceWarning`.
SIZE_WARN_THRESHOLD: int = 5000

#: Absolute tolerance for the ``D == D.T`` check (matches the pre-spec
#: behaviour of ``RepresentativeCycles._validate_precomputed``).
SYMMETRY_ATOL: float = 1e-6

#: Absolute tolerance for the zero-diagonal check.
DIAGONAL_ATOL: float = 1e-8

_BYTES_PER_FLOAT64 = 8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _location(mask: np.ndarray) -> str:
    """Describe the first ``True`` in *mask* as human-readable coordinates.

    Uses ``argmax`` rather than ``argwhere`` so the description costs O(1)
    extra memory even when most of a large matrix is offending.
    """
    index = np.unravel_index(int(np.argmax(mask)), mask.shape)
    if len(index) == 1:
        return f"index {int(index[0])}"
    return f"row {int(index[0])}, column {int(index[1])}"


def _first_index(mask: np.ndarray) -> tuple:
    """Return the index tuple of the first ``True`` in *mask*."""
    return np.unravel_index(int(np.argmax(mask)), mask.shape)


def _as_float64(array, name: str) -> np.ndarray:
    """Coerce *array* to a float64 ndarray, or raise a ``ValueError`` saying why.

    Complex input is rejected explicitly: NumPy would otherwise cast it with
    only a ``ComplexWarning`` and silently discard the imaginary part, which is
    exactly the kind of quiet data loss the constitution forbids.
    """
    raw = np.asarray(array)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"{name} has complex dtype {raw.dtype}; only real numeric data is "
            f"supported (casting would silently discard the imaginary part). "
            f"Pass the real part explicitly if that is what you meant."
        )
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a real numeric array; could not interpret dtype "
            f"{raw.dtype} as float64 ({exc})."
        ) from exc


def _reject_nan(values: np.ndarray, name: str) -> None:
    mask = np.isnan(values)
    if mask.any():
        raise ValueError(
            f"{name} contains NaN at {_location(mask)}. NaN has no topological "
            f"meaning and silently poisons every distance comparison, so it is "
            f"rejected rather than propagated. Remove or impute these entries "
            f"before fitting."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_point_cloud(X: np.ndarray) -> np.ndarray:
    """Validate a Euclidean point cloud and return it as float64.

    Parameters
    ----------
    X : array-like
        Point cloud of shape ``(n_points, n_dims)``.

    Returns
    -------
    np.ndarray
        The same data as a float64 array of shape ``(n_points, n_dims)``.  No
        copy is made when the input is already float64.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, is empty, has zero coordinate columns, is not real
        numeric, or contains ``NaN``, ``+inf`` or ``-inf``.  ``±inf`` is
        rejected here because an infinite *coordinate* has no Euclidean
        meaning — every distance from it would be ``inf`` or ``NaN``.  (In a
        *precomputed matrix* ``+inf`` is meaningful and accepted; see
        :func:`validate_precomputed`.)

    Notes
    -----
    Checks run cheapest-and-most-fundamental first (dtype, shape, size) so the
    caller gets the most actionable message, and all of them run before any
    caller allocates the ``n × n`` distance matrix.
    """
    values = _as_float64(X, "X")

    if values.ndim != 2:
        raise ValueError(
            f"For metric='euclidean', X must be a 2-D array of shape "
            f"(n_points, n_dims); got a {values.ndim}-D array with shape "
            f"{values.shape}. A flat list of scalars becomes a valid 1-D cloud "
            f"with X.reshape(-1, 1)."
        )

    n_points, n_dims = values.shape
    if n_points < 1:
        raise ValueError(
            f"X must contain at least one point; got an empty array of shape "
            f"{values.shape}."
        )
    if n_dims < 1:
        raise ValueError(
            f"X must have at least one coordinate column; got shape "
            f"{values.shape}, which describes {n_points} points with no "
            f"coordinates."
        )

    _reject_nan(values, "X")

    infinite = np.isinf(values)
    if infinite.any():
        index = _first_index(infinite)
        sign = "+inf" if values[index] > 0 else "-inf"
        raise ValueError(
            f"X contains {sign} at row {int(index[0])}, column {int(index[1])}. "
            f"An infinite coordinate has no Euclidean meaning — every distance "
            f"to it is inf or NaN. (+inf is only meaningful in a precomputed "
            f"distance matrix, where it encodes 'no edge'; pass "
            f"metric='precomputed' if that is what you intended.)"
        )

    return values


def validate_precomputed(D: np.ndarray) -> np.ndarray:
    """Validate a precomputed distance matrix and return it as float64.

    Parameters
    ----------
    D : array-like
        Square symmetric matrix of shape ``(n_points, n_points)`` with zeros on
        the diagonal.

    Returns
    -------
    np.ndarray
        The same data as a float64 array.  No copy is made when the input is
        already float64.

    Raises
    ------
    ValueError
        If ``D`` is not square, is empty, is not real numeric, contains ``NaN``,
        contains a negative value (``-inf`` included), has a non-zero diagonal
        beyond :data:`DIAGONAL_ATOL`, or is asymmetric beyond
        :data:`SYMMETRY_ATOL`.

    Notes
    -----
    ``+inf`` is **accepted** and means "no edge between these two points"
    (F10).  Because ``inf <= r`` is false for every finite radius, such a pair
    never enters the Rips graph, at any scale.  A matrix that is entirely
    ``+inf`` off the diagonal is therefore valid: it describes ``n`` mutually
    isolated points and yields no H₁ classes.  Judging whether an input is
    *topologically interesting* is not validation's job.

    Symmetry is checked with :func:`numpy.isclose` rather than
    ``max(abs(D - D.T))``: ``inf - inf`` is ``NaN``, so the subtraction form
    both mis-reports the error and emits a spurious ``RuntimeWarning`` on
    exactly the inputs F10 exists to support.  ``isclose`` treats two equal
    infinities as close, which is the intended semantics here.
    """
    values = _as_float64(D, "X")

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"When metric='precomputed', X must be a square 2-D matrix of "
            f"shape (n_points, n_points); got shape {values.shape}."
        )

    n_points = values.shape[0]
    if n_points < 1:
        raise ValueError(
            f"The precomputed distance matrix must describe at least one "
            f"point; got an empty array of shape {values.shape}."
        )

    _reject_nan(values, "The precomputed distance matrix")

    negative = values < 0.0
    if negative.any():
        index = _first_index(negative)
        raise ValueError(
            f"The precomputed distance matrix contains the negative value "
            f"{float(values[index]):g} at row {int(index[0])}, column "
            f"{int(index[1])}. Distances must be non-negative. (+inf is "
            f"allowed and means 'no edge between these two points'.)"
        )

    diagonal = np.diag(values)
    bad_diagonal = ~(np.abs(diagonal) <= DIAGONAL_ATOL)  # NaN-safe, inf-safe
    if bad_diagonal.any():
        index = int(_first_index(bad_diagonal)[0])
        raise ValueError(
            f"The precomputed distance matrix must have zeros on the "
            f"diagonal; found {float(diagonal[index]):g} at index {index} "
            f"(tolerance {DIAGONAL_ATOL:g}). D[i, i] is the distance from a "
            f"point to itself."
        )

    asymmetric = ~np.isclose(values, values.T, rtol=0.0, atol=SYMMETRY_ATOL)
    if asymmetric.any():
        i, j = (int(k) for k in _first_index(asymmetric))
        raise ValueError(
            f"The precomputed distance matrix must be symmetric; "
            f"D[{i}, {j}] = {float(values[i, j]):g} but D[{j}, {i}] = "
            f"{float(values[j, i]):g} (tolerance {SYMMETRY_ATOL:g})."
        )

    return values


def check_size(n: int, max_points: Optional[int] = None) -> None:
    """Guard the O(n²) distance-matrix allocation against memory exhaustion.

    Parameters
    ----------
    n : int
        Number of points about to be fitted.
    max_points : int or None, default ``None``
        Hard limit.  ``None`` — the default — means *no* limit, which preserves
        the behaviour of every release before spec 001 (B3): existing callers
        who legitimately fit large clouds keep working.

    Raises
    ------
    ValueError
        If ``n`` exceeds an explicitly-set ``max_points``.
    TypeError
        If ``max_points`` is neither an integer nor ``None``.

    Warns
    -----
    ResourceWarning
        When ``n`` exceeds :data:`SIZE_WARN_THRESHOLD` (5000) and no hard limit
        was breached.  The message states the projected float64 distance-matrix
        cost, because that allocation — not the persistent-homology step — is
        what makes a large fit fail on a laptop.

        Note that CPython ignores :class:`ResourceWarning` under its default
        filters; a caller who wants to *see* the warning runs with
        ``-W default::ResourceWarning``, and a caller who wants to *act* on the
        limit sets ``max_points``.  The category is prescribed by spec 001 B3.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an integer; got {type(n).__name__}.")
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")

    projected_mb = (n * n * _BYTES_PER_FLOAT64) / 1e6

    if max_points is not None:
        if isinstance(max_points, bool) or not isinstance(
            max_points, (int, np.integer)
        ):
            raise TypeError(
                f"max_points must be an integer or None; got "
                f"{type(max_points).__name__}."
            )
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError(
                f"max_points must be a positive integer or None; got "
                f"{max_points}."
            )
        if n > max_points:
            raise ValueError(
                f"{n} points exceeds max_points={max_points}. Fitting would "
                f"allocate a {n} x {n} float64 distance matrix "
                f"(~{projected_mb:.0f} MB). Subsample the input, or raise "
                f"max_points if you have the memory."
            )

    if n > SIZE_WARN_THRESHOLD:
        warnings.warn(
            f"Fitting {n} points allocates a {n} x {n} float64 distance matrix "
            f"(~{projected_mb:.0f} MB), plus a persistence computation that "
            f"grows faster still. Pass max_points to fail fast instead of "
            f"warning, or subsample.",
            ResourceWarning,
            stacklevel=2,
        )


def validate_save_path(
    path: Optional[Union[str, os.PathLike]],
) -> Optional[Union[str, os.PathLike]]:
    """Check that a figure can actually be written to *path*, and return it.

    Parameters
    ----------
    path : str, os.PathLike or None
        Destination for a figure.  ``None`` means "do not save" and passes
        through unchanged.

    Returns
    -------
    The *path* object exactly as given, so callers can hand it straight to
    ``Figure.savefig`` / ``plotly.io.write_html`` without a type surprise.

    Raises
    ------
    ValueError
        If the path is empty, its parent directory does not exist or is not a
        directory, or the path itself names an existing directory.
    TypeError
        If *path* is neither ``None`` nor a string/path-like object.

    Notes
    -----
    This is a **usability** check, not a path-traversal defence, and
    deliberately so (spec 001, Non-Functional/Security): this is a library
    whose caller *is* the trust boundary — the caller already runs arbitrary
    Python in the same process, so there is no privilege gradient for a
    ``../`` check to protect.  What the caller genuinely needs is to learn
    that ``output/figures/`` does not exist *before* a 30-second fit, and to
    learn it by name rather than through a raw ``OSError`` from deep inside
    matplotlib.
    """
    if path is None:
        return None

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            f"save_path must be a string, a path-like object, or None; got "
            f"{type(path).__name__}."
        )

    text = os.fspath(path)
    if not text.strip():
        raise ValueError(
            "save_path must be a non-empty path, or None to skip saving."
        )

    # Paths are quoted, never repr'd: on Windows repr doubles every backslash,
    # so the user cannot copy the path out of the message.
    if os.path.isdir(text):
        raise ValueError(
            f'Cannot save to "{text}": that is an existing directory, not a '
            f"file path. Append a filename, e.g. "
            f'"{os.path.join(text, "figure.png")}".'
        )

    parent = os.path.dirname(os.path.abspath(text))
    if not os.path.exists(parent):
        raise ValueError(
            f'Cannot save to "{text}": the parent directory "{parent}" does '
            f"not exist. Create it first, e.g. "
            f'os.makedirs("{parent}", exist_ok=True).'
        )
    if not os.path.isdir(parent):
        raise ValueError(
            f'Cannot save to "{text}": "{parent}" is a file, not a directory, '
            f"so it cannot contain the output."
        )

    return path
