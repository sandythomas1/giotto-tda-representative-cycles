"""Representative-cycle reconstruction on the Vietoris-Rips 1-skeleton.

Given the birth edge ``(u, v)`` of an H₁ class and the radius at which it is
born, the representative loop is the shortest ``u``→``v`` path in the Rips
graph at that radius *excluding* the birth edge, closed with the birth edge.

This is a **heuristic**, not an algebraic canonical representative — see
:func:`reconstruct_cycle` for exactly what it does and does not guarantee.
Every returned :class:`CycleResult` therefore carries the evidence a reader
needs to judge it: the traversal order, the total geodesic length, and an
``is_verified`` flag that is ``True`` only when the loop was checked and
passed.

A fit reconstructs one cycle per H₁ class, and every one of them needs the
Rips 1-skeleton at its own birth radius.  :class:`RipsGraphCache` builds the
filtration-sorted edge list **once per fit** so that each feature costs a
slice instead of a fresh ``n × n`` scan.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra

from .errors import CycleReconstructionWarning

__all__ = [
    "CycleResult",
    "reconstruct_cycle",
    "RipsGraphCache",
    "RADIUS_RTOL",
]

#: Relative slack allowed when checking an edge length against the birth
#: radius.  The comparison is ``length <= radius * (1 + RADIUS_RTOL)``; the
#: slack absorbs float64 round-off in the distance matrix, nothing more.
RADIUS_RTOL = 1e-9


@dataclass
class CycleResult:
    """A reconstructed representative cycle, with its supporting evidence.

    Attributes
    ----------
    edges : np.ndarray, shape (n_edges, 2)
        Edge list of the cycle, in traversal order.  ``edges[k]`` is always
        ``(path[k], path[k + 1])``, so the two fields cannot disagree.
        Degenerate exception: on a degraded reconstruction this is the single
        birth edge ``[[u, v]]`` (see :func:`reconstruct_cycle`).
    path : np.ndarray, shape (n_vertices + 1,)
        Vertices in loop traversal order with the first vertex repeated at the
        end, so the array plots directly as a closed polygon.
    length : float
        Sum of the edge lengths.  This is the quantity that is invariant
        across implementations: the shortest-path *distance* between two
        vertices is unique even where the shortest *path* is not, so two
        correct implementations that pick different equal-cost paths still
        agree here.
    is_verified : bool
        ``True`` only when **both** checks passed: every vertex of the cycle
        has even degree (the loop is closed as a ℤ/2 cycle), and every edge is
        no longer than the birth radius (the loop really exists in the Rips
        complex at that filtration value).  ``False`` means the loop is
        reported but must not be trusted as a representative.
    """

    edges: np.ndarray
    path: np.ndarray
    length: float
    is_verified: bool

    @property
    def vertices(self) -> np.ndarray:
        """Unique vertex indices of the cycle, sorted.

        Convenience for callers that need a set rather than an order; use
        :attr:`path` when traversal order matters.
        """
        if self.edges.size == 0:
            return np.empty(0, dtype=int)
        return np.unique(self.edges)


def reconstruct_cycle(
    dist_matrix: np.ndarray,
    birth_edge: Tuple[int, int],
    birth_radius: float,
    graph_cache: Optional["RipsGraphCache"] = None,
) -> CycleResult:
    """Reconstruct a representative loop for one H₁ class.

    **This is a shortest-geodesic-path heuristic.**  It returns *a* loop in
    the homology class of the birth edge, chosen to be short; it does not
    return the algebraically canonical representative that boundary-matrix
    reduction (JavaPlex, GUDHI) would produce, and it does not solve the
    minimal-homologous-cycle problem (NP-hard over ℤ in general).

    What it guarantees, when ``is_verified`` is ``True``:

    - the returned edges form a closed loop over ℤ/2 (every vertex has even
      degree), and
    - every edge is present in the Rips complex at ``birth_radius``, so the
      loop is drawn from simplices that genuinely exist at that filtration
      value.

    What it does **not** guarantee, ever:

    - that the loop is the shortest loop in its homology class,
    - that it is non-trivial in homology beyond being anchored on the birth
      edge that ripser attributes to this class, or
    - byte-identity with any other implementation.  Equal-cost shortest paths
      are not unique, so a different (equally valid) path may be selected by a
      different graph construction.  :attr:`CycleResult.length` is the stable
      quantity to compare across implementations.

    Determinism is guaranteed *within* an implementation: two calls with the
    same inputs build the same CSR graph in the same order and therefore
    return byte-identical edges.

    **Algorithm**

    1. Take the Rips 1-skeleton at ``birth_radius`` with the birth edge
       ``(u, v)`` removed, forcing any ``u``→``v`` path to go *around* the
       loop.
    2. Run ``scipy.sparse.csgraph.dijkstra`` (C level) from ``u`` and trace the
       predecessor array back from ``v``.
    3. Close the traced path with the birth edge, then verify it.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Float64 pairwise distances.
    birth_edge : tuple[int, int]
        The edge ``(u, v)`` that creates the H₁ class.  ``u != v``.
    birth_radius : float
        The filtration value (edge length) at which the feature is born.
    graph_cache : RipsGraphCache, optional
        Pre-built Rips graph provider — see :class:`RipsGraphCache`.  When
        ``None`` — the default — the graph is built for this call alone from a
        dense ``n × n`` mask, which is correct but costs O(n²) per feature.  A
        cache is shared across features, so it must return the **full** graph
        at the radius *including* the birth edge; removing the birth edge is
        done here, per call.

    Returns
    -------
    CycleResult

    Warns
    -----
    CycleReconstructionWarning
        When ``u`` and ``v`` lie in different connected components of the Rips
        graph at ``birth_radius`` once the birth edge is removed, so no loop
        exists to trace.  The result then degrades to the birth edge alone,
        with ``is_verified=False``.  The degradation is never silent: the
        constitution forbids reporting a degraded topological object as if it
        were a verified one.
    """
    D = dist_matrix
    u, v = _validate_birth_edge(birth_edge, D)

    with _rips_graph_without_birth_edge(
        D, birth_radius, u, v, graph_cache
    ) as graph:
        dist_row, pred = _sp_dijkstra(
            graph, directed=False, indices=u, return_predecessors=True
        )

    if not np.isfinite(dist_row[v]):
        return _degraded_result(D, u, v, birth_radius)

    path = np.asarray(_trace_path(pred, u, v), dtype=int)
    closed_path = np.append(path, path[0])
    edges = np.column_stack([closed_path[:-1], closed_path[1:]])

    lengths = D[edges[:, 0], edges[:, 1]]
    is_verified = _is_closed_over_z2(edges) and _fits_within_radius(
        lengths, _reference_radius(D, u, v, birth_radius)
    )

    return CycleResult(
        edges=edges,
        path=closed_path,
        length=float(lengths.sum()),
        is_verified=is_verified,
    )


class RipsGraphCache:
    """The Rips 1-skeleton at any radius, from one sorted edge list per fit.

    Reconstruction needs the graph at every feature's birth radius.  Built
    per feature it costs a dense ``n × n`` comparison plus an ``n × n`` scan
    *each time*; built here it costs one sort up front and a
    :func:`numpy.searchsorted` per feature, because the edges present in
    Rips(*r*) are exactly a prefix of the edges sorted by length.

    Call :meth:`graph_at` with **ascending** radii.  The assembled CSR is
    rebuilt only when the prefix actually changes, so features that share a
    filtration value — and ripser's births cluster heavily — reuse the same
    graph object.  Only the most recent graph is retained; out-of-order radii
    stay correct, they just rebuild more often.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (n, n)
        Float64 pairwise distances.  Held **by reference**, not copied: it is
        the source of truth for edge weights, so it must not be mutated while
        the cache is alive.  Non-finite entries (``+inf`` = "no edge", per the
        precomputed-matrix contract) are never stored as edges.
    max_edge_length : float, default ``np.inf``
        Edges longer than this are not stored at all.  Passing the fit's own
        threshold is what keeps the structure small; :meth:`graph_at` refuses
        radii above it rather than quietly returning a truncated complex.
        A caller that knows the largest birth radius up front can pass that
        instead, and pay for nothing beyond it.

    Notes
    -----
    **Memory.**  The structure is ``n(n - 1) / 2`` edges × 12 bytes — a
    ``float32`` sort key and two ``int32`` endpoints — versus ``n²`` bytes for
    the boolean mask it replaces.  That is ~6× larger at the same *n* when no
    ``max_edge_length`` is set (13.5 MB at n=1500, 150 MB at n=5000), and the
    reason the dtypes are pinned: ``float64`` keys with ``int64`` endpoints
    would be 16 bytes and 200 MB.  The trade is deliberate — the mask was
    allocated once *per feature* (426 times on the 1500-point torus) while
    this is allocated once per fit — but anyone fitting more than ~10⁴ points
    should set ``max_edge_length``.  Construction transiently needs roughly
    twice the final size for the sort permutation.

    **Precision.**  The sort key is the edge length rounded *up* to
    ``float32``, so it is never below the true length.  The prefix
    ``key <= radius`` is therefore always a subset of the true Rips complex,
    and the handful of edges whose rounding straddles the cut are re-tested
    against the exact ``float64`` distance before being admitted.  The result
    is exactly ``{(i, j) : dist_matrix[i, j] <= radius}`` — the float32 key
    buys memory, not approximation.

    Examples
    --------
    >>> cache = RipsGraphCache(dist_matrix, max_edge_length=1.0)
    >>> for birth_edge, radius in sorted(generators, key=lambda g: g[1]):
    ...     result = reconstruct_cycle(dist_matrix, birth_edge, radius, cache)
    """

    def __init__(
        self, dist_matrix: np.ndarray, max_edge_length: float = np.inf
    ) -> None:
        D = np.asarray(dist_matrix)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(
                f"dist_matrix must be square 2-D; got shape {D.shape}."
            )
        limit = float(max_edge_length)
        if np.isnan(limit):
            raise ValueError("max_edge_length must not be NaN.")

        self._dist_matrix = D
        self._max_edge_length = limit
        self._key, self._rows, self._cols = _sorted_edge_list(D, limit)
        self._signature: Optional[Tuple[int, bytes]] = None
        self._graph: Optional[csr_matrix] = None

    # -- introspection ------------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of vertices."""
        return int(self._dist_matrix.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of edges stored, i.e. of length ≤ ``max_edge_length``."""
        return int(self._key.size)

    @property
    def max_edge_length(self) -> float:
        """Longest edge the cache can answer for."""
        return self._max_edge_length

    @property
    def nbytes(self) -> int:
        """Bytes held by the sorted edge structure."""
        return int(self._key.nbytes + self._rows.nbytes + self._cols.nbytes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(n_points={self.n_points}, "
            f"n_edges={self.n_edges}, nbytes={self.nbytes})"
        )

    # -- the one operation that matters -------------------------------------

    def graph_at(self, radius: float) -> csr_matrix:
        """Return the Rips 1-skeleton at *radius* as an upper-triangular CSR.

        Only ``(min, max)`` endpoint pairs are stored; SciPy's csgraph
        routines are called with ``directed=False``, which reads an edge in
        both directions, so storing it twice would double the memory and the
        build cost for nothing.

        The returned matrix is the cache's own object.  Callers may mutate its
        ``data`` in place — :func:`reconstruct_cycle` masks the birth edge that
        way — provided they restore it before returning, which is exactly what
        :func:`_rips_graph_without_birth_edge` guarantees.  It is not
        thread-safe, by the same deliberate choice made in T6.

        Raises
        ------
        ValueError
            If *radius* is NaN, or exceeds ``max_edge_length``.  The cache
            simply does not hold the edges needed to answer, and returning a
            silently truncated complex would be exactly the kind of quiet
            degradation the constitution forbids.
        """
        r = float(radius)
        if np.isnan(r):
            raise ValueError("radius must not be NaN.")
        if r > self._max_edge_length:
            raise ValueError(
                f"radius {r:.6g} exceeds this cache's max_edge_length "
                f"{self._max_edge_length:.6g}; edges longer than the limit "
                f"were never stored, so the graph at this radius is not "
                f"available. Rebuild the cache with a larger limit."
            )

        cut, straddling = self._prefix_at(r)
        signature = (cut, straddling.tobytes())
        if signature != self._signature or self._graph is None:
            self._graph = self._build_graph(cut, straddling)
            self._signature = signature
        return self._graph

    # -- internals ----------------------------------------------------------

    def _prefix_at(self, radius: float) -> Tuple[int, np.ndarray]:
        """Split Rips(*radius*) into a clean prefix and the straddling edges.

        Every edge below ``cut`` has a rounded-up key ≤ *radius*, hence a true
        length ≤ *radius*, and is in beyond doubt.  Rounding up can push an
        edge that belongs in the complex just past the cut, but never by more
        than one ``float32`` ulp, so those candidates form a short contiguous
        run which is settled by reading the exact ``float64`` distances.

        Both cuts search for a ``float32`` needle rather than the ``float64``
        radius.  This is not a micro-optimisation: NumPy resolves a mixed-dtype
        :func:`numpy.searchsorted` by promoting the *array*, so a float64
        needle silently converts all ``n(n-1)/2`` keys on **every call** —
        measured at 1.5 s of the 2.4 s budget on the 1500-point torus, which
        would have wiped out the whole optimisation.  Flooring is exact for
        this purpose: a ``float32`` key is ≤ a float64 radius exactly when it
        is ≤ the largest ``float32`` not exceeding that radius.
        """
        cut = int(
            np.searchsorted(self._key, _floor_to_float32(radius), side="right")
        )
        upper = radius + abs(radius) * _KEY_ROUNDING_RTOL
        end = int(
            np.searchsorted(self._key, _floor_to_float32(upper), side="right")
        )
        if end <= cut:
            return cut, _NO_EDGES

        candidates = np.arange(cut, end)
        lengths = self._dist_matrix[
            self._rows[cut:end], self._cols[cut:end]
        ]
        return cut, candidates[lengths <= radius]

    def _build_graph(self, cut: int, straddling: np.ndarray) -> csr_matrix:
        """Assemble the CSR for a prefix plus its admitted straddling edges."""
        rows, cols = self._rows[:cut], self._cols[:cut]
        if straddling.size:
            rows = np.concatenate([rows, self._rows[straddling]])
            cols = np.concatenate([cols, self._cols[straddling]])

        weights = self._dist_matrix[rows, cols]
        n = self.n_points
        return csr_matrix((weights, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

#: Relative width of the band above a radius that ``float32`` key rounding can
#: push a genuine Rips edge into.  One ``float32`` ulp is ``2**-23``; twice
#: that is a margin, not a tuned constant — a band that is slightly too wide
#: costs a few exact distance lookups, while one that is too narrow would drop
#: real edges.
_KEY_ROUNDING_RTOL = 2.0**-22

#: Rows of the distance matrix compared at a time when building the edge list.
#: Bounds the temporaries to a few tens of MB at n=5000 instead of materialising
#: an ``n × n`` mask and ``n²``-sized ``int64`` index arrays in one go.
_EDGE_BLOCK_ROWS = 256

_NO_EDGES = np.empty(0, dtype=np.intp)


def _sorted_edge_list(
    dist_matrix: np.ndarray, max_edge_length: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(key, rows, cols)`` for every edge ≤ *max_edge_length*.

    ``key`` is ``float32`` ascending, ``rows``/``cols`` are ``int32`` with
    ``rows < cols`` — the dtypes the memory requirement pins down (AC13a).
    """
    n = int(dist_matrix.shape[0])
    keys: List[np.ndarray] = []
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []

    for start in range(0, n, _EDGE_BLOCK_ROWS):
        stop = min(start + _EDGE_BLOCK_ROWS, n)
        block = dist_matrix[start:stop]

        # Upper triangle only, and never a non-finite distance: `+inf` in a
        # precomputed matrix means "no edge", so it must not become one.
        row_ids = np.arange(start, stop)[:, None]
        col_ids = np.arange(n)[None, :]
        within = (
            block <= max_edge_length
            if np.isfinite(max_edge_length)
            else np.isfinite(block)
        )
        block_rows, block_cols = np.nonzero(within & (col_ids > row_ids))
        if block_rows.size == 0:
            continue

        keys.append(_ceil_to_float32(block[block_rows, block_cols]))
        rows.append((block_rows + start).astype(np.int32))
        cols.append(block_cols.astype(np.int32))

    if not keys:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    key = np.concatenate(keys)
    order = np.argsort(key, kind="stable")
    return key[order], np.concatenate(rows)[order], np.concatenate(cols)[order]


def _ceil_to_float32(lengths: np.ndarray) -> np.ndarray:
    """Round *lengths* up to the nearest ``float32``.

    Rounding *up* is what makes the cheap key safe: ``key >= length``, so
    ``key <= radius`` implies ``length <= radius`` and the prefix can never
    contain an edge that is absent from the true Rips complex.  Plain
    ``astype`` rounds to nearest and would admit edges up to an ulp too long,
    whose presence a cycle's own radius check would then — correctly — report
    as unverified.
    """
    key = lengths.astype(np.float32)
    rounded_down = key < lengths
    if rounded_down.any():
        key[rounded_down] = np.nextafter(
            key[rounded_down], np.float32(np.inf)
        )
    return key


def _floor_to_float32(value: float) -> np.float32:
    """Round *value* down to the nearest ``float32``.

    The mirror of :func:`_ceil_to_float32`, and needed for the opposite
    reason: it turns a ``float64`` *radius* into a needle that can be compared
    against the ``float32`` keys without NumPy promoting the whole key array
    on every :func:`numpy.searchsorted` call.  Rounding *down* keeps the
    comparison exact -- a ``float32`` key is ``<=`` a float64 radius exactly
    when it is ``<=`` the largest ``float32`` not exceeding that radius -- so
    the prefix is neither widened nor narrowed by the conversion.
    """
    key = np.float32(value)
    if key > value:
        key = np.nextafter(key, np.float32(-np.inf))
    return key


def _validate_birth_edge(
    birth_edge: Tuple[int, int], dist_matrix: np.ndarray
) -> Tuple[int, int]:
    """Return the birth edge as ``(int, int)``, rejecting nonsense endpoints.

    A self-loop or an out-of-range vertex cannot come from a ripser generator,
    so this guards against a caller wiring the wrong array in rather than
    against a real topological case.
    """
    u, v = int(birth_edge[0]), int(birth_edge[1])
    n = dist_matrix.shape[0]
    if u == v:
        raise ValueError(
            f"birth_edge must join two distinct vertices (got ({u}, {v}))."
        )
    if not (0 <= u < n and 0 <= v < n):
        raise ValueError(
            f"birth_edge ({u}, {v}) is out of range for a distance matrix "
            f"with {n} points."
        )
    return u, v


@contextmanager
def _rips_graph_without_birth_edge(
    dist_matrix: np.ndarray,
    radius: float,
    u: int,
    v: int,
    graph_cache: Optional["RipsGraphCache"],
) -> Iterator[csr_matrix]:
    """Yield the Rips 1-skeleton at *radius* with ``(u, v)`` unusable.

    Without a cache the graph is built for this call and simply omits the
    edge.  With a cache the graph is shared across features, so it is masked
    in place for the duration of the block and restored on exit: an O(degree)
    edit instead of an O(|E|) copy per feature, which matters because the
    cache exists precisely to avoid per-feature O(|E|) work.  The mutation is
    not re-entrant or thread-safe — this library is single-threaded by
    construction, and a cache that wants immutability can return a copy from
    ``graph_at``.

    The masked weight is ``inf``, not ``0``: in a SciPy sparse csgraph a
    stored zero is an ambiguous "zero-length edge or no edge at all", so
    zeroing would also erase genuine zero-length edges between duplicate
    points.  An infinite weight is unambiguous — it can never lie on a finite
    shortest path, and when it is the only connection the resulting distance
    is ``inf``, which is exactly the disconnected case we need to report.
    """
    if graph_cache is None:
        yield _build_graph_excluding_edge(dist_matrix, radius, u, v)
        return

    graph = graph_cache.graph_at(radius)
    positions = np.concatenate(
        [_edge_positions(graph, u, v), _edge_positions(graph, v, u)]
    )
    saved = graph.data[positions].copy()
    graph.data[positions] = np.inf
    try:
        yield graph
    finally:
        graph.data[positions] = saved


def _edge_positions(graph: csr_matrix, row: int, col: int) -> np.ndarray:
    """Indices into ``graph.data`` holding the weight of edge ``(row, col)``.

    Empty when the edge is absent, which is legitimate: a cache cut just below
    the birth edge's length simply does not contain it.
    """
    start, end = graph.indptr[row], graph.indptr[row + 1]
    return start + np.flatnonzero(graph.indices[start:end] == col)


def _build_graph_excluding_edge(
    dist_matrix: np.ndarray, radius: float, u: int, v: int
) -> csr_matrix:
    """Build the CSR graph for a single call, omitting the edge ``(u, v)``.

    ``np.nonzero`` walks the mask in row-major order, so the CSR arrays — and
    hence Dijkstra's predecessor output — are a deterministic function of the
    inputs (F8).
    """
    D = dist_matrix
    n = D.shape[0]

    mask = D <= radius
    mask[u, v] = False
    mask[v, u] = False
    np.fill_diagonal(mask, False)

    rows, cols = np.nonzero(mask)
    return csr_matrix((D[rows, cols], (rows, cols)), shape=(n, n))


def _trace_path(pred: np.ndarray, u: int, v: int) -> List[int]:
    """Walk the Dijkstra predecessor array from *v* back to *u*."""
    reverse_path: List[int] = []
    cur = int(v)
    while cur != u and cur >= 0:
        reverse_path.append(cur)
        cur = int(pred[cur])
    reverse_path.append(u)
    reverse_path.reverse()
    return reverse_path


def _is_closed_over_z2(edges: np.ndarray) -> bool:
    """``True`` when every vertex of *edges* has even degree."""
    if edges.size == 0:
        return False
    _, counts = np.unique(edges, return_counts=True)
    return bool(np.all(counts % 2 == 0))


def _fits_within_radius(lengths: np.ndarray, radius: float) -> bool:
    """``True`` when every edge exists in the Rips complex at *radius*."""
    if lengths.size == 0:
        return False
    return bool(np.all(lengths <= radius * (1.0 + RADIUS_RTOL)))


def _reference_radius(
    dist_matrix: np.ndarray, u: int, v: int, birth_radius: float
) -> float:
    """Radius the edge-length check measures against.

    The birth edge is, by definition, present in the complex at the moment the
    class is born — its exact float64 length *is* the birth radius.  Callers
    commonly hold a slightly different value, because ripser computes the
    persistence diagram in float32 while our distance matrix is float64, and
    that gap (~1e-7 relative) is far wider than :data:`RADIUS_RTOL`.  Taking
    the larger of the two absorbs that representation mismatch instead of
    reporting every feature as unverified over a rounding artefact.

    This loosens nothing in practice: every *other* edge of the cycle comes
    out of a graph already cut at ``birth_radius``, so the birth edge is the
    only one this can affect.  The check remains an independent runtime
    re-measurement of the returned edges rather than a restatement of how the
    graph was built.
    """
    return max(float(birth_radius), float(dist_matrix[u, v]))


def _degraded_result(
    dist_matrix: np.ndarray, u: int, v: int, radius: float
) -> CycleResult:
    """Fall back to the birth edge alone, loudly.

    ``path`` retraces the single edge (``[u, v, u]``) so it still reads as a
    closed polygon for plotting, while ``edges`` keeps the one real edge; the
    two therefore have different lengths in this case only, and
    ``is_verified`` is ``False`` to say so.
    """
    warnings.warn(
        f"No path between vertices {u} and {v} in the Rips graph at radius "
        f"{radius:.6g} once the birth edge is removed: they lie in different "
        f"connected components, so no loop can be traced. Falling back to the "
        f"birth edge alone; the result is reported with is_verified=False and "
        f"must not be read as a representative cycle.",
        CycleReconstructionWarning,
        stacklevel=3,
    )
    return CycleResult(
        edges=np.array([[u, v]], dtype=int),
        path=np.array([u, v, u], dtype=int),
        length=float(dist_matrix[u, v]),
        is_verified=False,
    )
