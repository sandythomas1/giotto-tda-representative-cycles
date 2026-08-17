"""Verified pairing between ripser generators and persistence-diagram rows.

ripser returns two parallel structures for H₁: a persistence *diagram*
(``(n, 2)`` birth/death values) and a set of *generators* (the vertex indices
of the birth and death simplices).  Reading feature ``i``'s birth/death from
diagram row ``i`` assumes the two structures are emitted in the same order —
undocumented behaviour that happens to hold today.  This module replaces that
assumption with a check: a generator is paired to a diagram row only when the
row's birth value equals the length of the generator's birth edge,
``dist_matrix[u, v]``.

Two facts shape the algorithm (spec 001, F4/F4a/F4b):

1. **Tolerance is float32-sized.**  ``gph``/ripser computes the filtration in
   float32 while this package keeps distances in float64, so the diagram's
   birth value is a float32-rounded copy of ``dist_matrix[u, v]``.  Exact
   equality never holds in general; ``rtol=1e-5`` is float32 epsilon with
   headroom, ``atol=1e-9`` covers birth values at or near zero.

2. **Ties are normal, not exceptional.**  Congruent features produce diagram
   rows identical in *both* birth and death — three congruent hexagonal holes
   give three rows of exactly ``[1.0, 1.7320508]``.  Disambiguating by death
   value therefore disambiguates nothing.  A persistence diagram is a
   *multiset*: identical rows carry no information distinguishing one from
   another, so they are interchangeable and any assignment is equally correct.
   Matching claims the lowest unclaimed matching row and moves on.

An error is raised only when a generator matches **zero** unclaimed rows,
because that is the case where emitting a feature would attach one homology
class's birth/death values to another class's cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .errors import CycleReconstructionError

#: Relative tolerance for birth matching — float32 epsilon (~1.19e-7) with
#: headroom for accumulated rounding in the filtration value itself.
BIRTH_RTOL: float = 1e-5

#: Absolute tolerance, so that birth values at or near zero still match.
BIRTH_ATOL: float = 1e-9

#: Placeholder death simplex for essential classes, which never die.
ESSENTIAL_DEATH_EDGE: Tuple[int, int] = (-1, -1)

#: Cap on how many candidate rows an error message enumerates.
_MAX_REPORTED_CANDIDATES: int = 10


@dataclass(frozen=True)
class PairedRow:
    """One generator, matched to the diagram row it was born in.

    Attributes
    ----------
    generator_index : int
        Row within the generator's *own* array.  Finite and essential
        generators are indexed separately, so ``is_essential`` is required to
        interpret this value.
    diagram_index : int
        Row of the H₁ diagram claimed by this generator.  Unique across the
        returned list.
    birth, death : float
        Taken from the claimed diagram row, not recomputed from distances, so
        that reported values agree with the diagram the user plots.  ``death``
        is ``np.inf`` for essential classes.
    birth_edge : tuple[int, int]
        Vertices of the edge whose appearance created the class.
    death_edge : tuple[int, int]
        Vertices of the triangle edge that killed it, or
        ``(-1, -1)`` when the class is essential.
    is_essential : bool
        ``True`` when the class never dies within the filtration.
    """

    generator_index: int
    diagram_index: int
    birth: float
    death: float
    birth_edge: Tuple[int, int]
    death_edge: Tuple[int, int]
    is_essential: bool


def pair_generators(
    finite_gens: np.ndarray,
    essential_gens: np.ndarray,
    diagram: np.ndarray,
    dist_matrix: np.ndarray,
    rtol: float = BIRTH_RTOL,
    atol: float = BIRTH_ATOL,
) -> List[PairedRow]:
    """Match each H₁ generator to its own row of the persistence diagram.

    Matching is by *value*: a generator whose birth edge is ``(u, v)`` may
    claim a diagram row whose birth is within tolerance of
    ``dist_matrix[u, v]``.  Finite generators may claim only rows with a
    finite death; essential generators may claim only rows with an infinite
    death.  Among the rows a generator matches, the lowest unclaimed index is
    taken (see the module docstring: equal rows are interchangeable).

    Parameters
    ----------
    finite_gens : np.ndarray, shape (k, 4)
        ``[birth_v0, birth_v1, death_v0, death_v1]`` per finite H₁ class —
        ``gens[1][0]`` from ``ripser_parallel(..., return_generators=True)``.
    essential_gens : np.ndarray, shape (m, 2)
        ``[birth_v0, birth_v1]`` per essential H₁ class — ``gens[3][0]``.
        Essential generators carry a birth edge only.
    diagram : np.ndarray, shape (n, 2)
        The H₁ diagram, ``[birth, death]`` per row; deaths may be ``+inf``.
    dist_matrix : np.ndarray, shape (n_points, n_points)
        float64 pairwise distances — the same matrix the filtration was
        computed from.
    rtol, atol : float
        Birth-comparison tolerances, defaulting to :data:`BIRTH_RTOL` /
        :data:`BIRTH_ATOL`.  A row matches when
        ``abs(row_birth - length) <= atol + rtol * abs(length)``.

    Returns
    -------
    list[PairedRow]
        Finite generators in their input order, then essential generators in
        their input order.  Every returned row has a distinct
        ``diagram_index``.  Diagram rows with no generator are simply absent.

    Raises
    ------
    CycleReconstructionError
        When a generator matches no unclaimed diagram row.  The message names
        the generator, its birth-edge length, and the candidate rows.
    ValueError
        On malformed input: wrong array shapes, a non-square distance matrix,
        a vertex index outside the distance matrix, a NaN in the diagram, or
        negative tolerances.

    Notes
    -----
    The claim order is a **greedy heuristic**: generators are processed in
    input order and each takes the lowest unclaimed row it matches.  It is
    exact whenever distinct birth values are separated by more than the
    tolerance — the only realistic case, since the tolerance is float32-sized.
    It does *not* guarantee a maximum matching under pathologically chained
    near-ties (births spaced just under the tolerance apart, forming an
    overlap chain), where a greedy claim can leave a later generator without a
    row.  That case raises rather than mispairing, so it is never silent.
    """
    finite = _as_generator_array(finite_gens, 4, "finite_gens")
    essential = _as_generator_array(essential_gens, 2, "essential_gens")
    dgm = _as_diagram(diagram)
    distances = _as_distance_matrix(dist_matrix)

    if rtol < 0 or atol < 0:
        raise ValueError(
            f"Tolerances must be non-negative; got rtol={rtol}, atol={atol}."
        )

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    finite_rows = np.isfinite(deaths)
    essential_rows = np.isposinf(deaths)

    claimed = np.zeros(len(dgm), dtype=bool)
    paired: List[PairedRow] = []

    for generator_index, row in enumerate(finite):
        birth_edge = (int(row[0]), int(row[1]))
        diagram_index = _claim_row(
            generator_index=generator_index,
            birth_edge=birth_edge,
            is_essential=False,
            eligible=finite_rows,
            claimed=claimed,
            births=births,
            dgm=dgm,
            distances=distances,
            rtol=rtol,
            atol=atol,
        )
        claimed[diagram_index] = True
        paired.append(
            PairedRow(
                generator_index=generator_index,
                diagram_index=diagram_index,
                birth=float(births[diagram_index]),
                death=float(deaths[diagram_index]),
                birth_edge=birth_edge,
                death_edge=(int(row[2]), int(row[3])),
                is_essential=False,
            )
        )

    for generator_index, row in enumerate(essential):
        birth_edge = (int(row[0]), int(row[1]))
        diagram_index = _claim_row(
            generator_index=generator_index,
            birth_edge=birth_edge,
            is_essential=True,
            eligible=essential_rows,
            claimed=claimed,
            births=births,
            dgm=dgm,
            distances=distances,
            rtol=rtol,
            atol=atol,
        )
        claimed[diagram_index] = True
        paired.append(
            PairedRow(
                generator_index=generator_index,
                diagram_index=diagram_index,
                birth=float(births[diagram_index]),
                death=np.inf,
                birth_edge=birth_edge,
                death_edge=ESSENTIAL_DEATH_EDGE,
                is_essential=True,
            )
        )

    return paired


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _claim_row(
    *,
    generator_index: int,
    birth_edge: Tuple[int, int],
    is_essential: bool,
    eligible: np.ndarray,
    claimed: np.ndarray,
    births: np.ndarray,
    dgm: np.ndarray,
    distances: np.ndarray,
    rtol: float,
    atol: float,
) -> int:
    """Return the lowest unclaimed eligible row matching this generator."""
    length = _birth_edge_length(distances, birth_edge, generator_index)
    available = eligible & ~claimed
    matches = available & np.isclose(births, length, rtol=rtol, atol=atol)

    if not matches.any():
        raise CycleReconstructionError(
            _no_match_message(
                generator_index=generator_index,
                birth_edge=birth_edge,
                is_essential=is_essential,
                length=length,
                available=available,
                eligible=eligible,
                dgm=dgm,
                rtol=rtol,
                atol=atol,
            )
        )

    # argmax on a boolean array returns the first True — the lowest index.
    return int(np.argmax(matches))


def _birth_edge_length(
    distances: np.ndarray, birth_edge: Tuple[int, int], generator_index: int
) -> float:
    """Return ``distances[u, v]``, with a clear error on out-of-range indices."""
    n_points = distances.shape[0]
    u, v = birth_edge
    if not (0 <= u < n_points and 0 <= v < n_points):
        raise ValueError(
            f"Generator {generator_index} has birth edge {birth_edge}, which "
            f"indexes outside the {n_points}x{n_points} distance matrix."
        )
    return float(distances[u, v])


def _no_match_message(
    *,
    generator_index: int,
    birth_edge: Tuple[int, int],
    is_essential: bool,
    length: float,
    available: np.ndarray,
    eligible: np.ndarray,
    dgm: np.ndarray,
    rtol: float,
    atol: float,
) -> str:
    kind = "essential" if is_essential else "finite"
    death_kind = "infinite" if is_essential else "finite"
    candidates = _describe_rows(dgm, available)
    n_eligible = int(eligible.sum())
    n_available = int(available.sum())
    return (
        f"{kind.capitalize()} generator {generator_index} with birth edge "
        f"{birth_edge} (birth-edge length {length!r}) matches no unclaimed H1 "
        f"diagram row within rtol={rtol!r}, atol={atol!r}. "
        f"{n_available} of {n_eligible} rows with {death_kind} death were "
        f"still unclaimed: {candidates}. "
        "Pairing was abandoned rather than attaching another class's "
        "birth/death values to this cycle."
    )


def _describe_rows(dgm: np.ndarray, mask: np.ndarray) -> str:
    """Render the masked diagram rows as ``index: [birth, death]``, truncated."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return "none"
    shown = indices[:_MAX_REPORTED_CANDIDATES]
    parts = [
        f"{i}: [{float(dgm[i, 0])!r}, {float(dgm[i, 1])!r}]"
        for i in shown.tolist()
    ]
    if indices.size > shown.size:
        parts.append(f"... ({indices.size - shown.size} more)")
    return "[" + ", ".join(parts) + "]"


def _as_generator_array(
    gens: Optional[np.ndarray], n_columns: int, name: str
) -> np.ndarray:
    """Coerce a generator array to ``(k, n_columns)`` int, allowing empties.

    ripser hands back an empty array of unhelpful shape when a dimension has
    no classes, so any empty input is normalised to ``(0, n_columns)``.
    """
    if gens is None:
        return np.empty((0, n_columns), dtype=np.int64)
    array = np.asarray(gens)
    if array.size == 0:
        return np.empty((0, n_columns), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != n_columns:
        raise ValueError(
            f"{name} must have shape (k, {n_columns}); got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError(
                f"{name} must contain integer vertex indices; "
                f"got dtype {array.dtype} with non-integral values."
            )
        array = array.astype(np.int64)
    return array


def _as_diagram(diagram: Optional[np.ndarray]) -> np.ndarray:
    """Coerce the H₁ diagram to a float64 ``(n, 2)`` array."""
    if diagram is None:
        return np.empty((0, 2), dtype=np.float64)
    array = np.asarray(diagram, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"diagram must have shape (n, 2) of [birth, death]; "
            f"got {array.shape}."
        )
    nan_rows = np.flatnonzero(np.isnan(array).any(axis=1))
    if nan_rows.size:
        raise ValueError(
            f"diagram contains NaN at row {int(nan_rows[0])}: "
            f"{array[nan_rows[0]].tolist()}. NaN has no topological meaning "
            "and cannot be matched against a birth-edge length."
        )
    return array


def _as_distance_matrix(dist_matrix: np.ndarray) -> np.ndarray:
    """Coerce the distance matrix to float64 and check it is square."""
    array = np.asarray(dist_matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"dist_matrix must be a square 2-D array; got {array.shape}."
        )
    return array
