"""Persistent homology extraction and representative-cycle assembly.

Architecture note (see ``specs/constitution.md``): this module must not import
matplotlib or plotly at module scope.  The ``plot_*`` methods delegate to
``repcycles.plotting`` behind function-local imports so that the computation
layer stays importable without any plotting stack installed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from gph import ripser_parallel

from .feature import CycleFeature
from .pairing import pair_generators
from .reconstruction import reconstruct_cycle
from .validation import (
    check_size,
    validate_point_cloud,
    validate_precomputed,
    validate_save_path,
)


def _feature_sort_key(f: CycleFeature):
    """Total ordering for ``features_`` (spec 001, F8a).

    Essential first, then persistence descending, then birth ascending, then
    diagram row ascending.  The last two keys are what make the order *total*:
    every essential feature has ``persistence = inf``, so sorting on
    persistence alone would leave their relative order to whatever order
    ripser happened to emit, and "deterministic" would be a claim we could not
    honour.
    """
    return (
        not f.is_essential,        # False (essential) sorts before True
        -f.persistence if np.isfinite(f.persistence) else -np.inf,
        f.birth,
        f.index,
    )


class RepresentativeCycles:
    """Compute and visualise representative H₁ cycles from a point cloud
    or a precomputed distance matrix.

    Parameters
    ----------
    min_persistence : float, default 0.0
        Discard H₁ features with persistence below this threshold.
    max_edge_length : float, default np.inf
        Maximum edge length in the Vietoris-Rips filtration.
    metric : str, default ``'euclidean'``
        Distance metric.  Use ``'euclidean'`` when ``X`` passed to
        :meth:`fit` is a point cloud of shape ``(n_points, n_dims)``, or
        ``'precomputed'`` when ``X`` is a symmetric ``(n_points, n_points)``
        distance matrix with zeros on the diagonal.

        When ``metric='precomputed'`` a 2-D MDS embedding of the distance
        matrix is computed internally and used *only* for visualisation; all
        topological computations use the original distances.
    coeff : int, default 2
        Coefficient field (prime).  Use 2 for Z/2Z.
    n_threads : int, default 1
        Threads passed to ``ripser_parallel``.
    reconstruct_cycles : bool, default True
        If ``True``, compute the full representative cycle for each feature
        via shortest-path reconstruction on the Rips graph at birth radius.
        Set to ``False`` to obtain only persistence diagrams and birth/death
        simplices (faster for large inputs when only the barcode is needed).
    include_essential : bool, default True
        Include *essential* H₁ classes — those that never die within the
        filtration, reported by ripser with ``death = inf``.  They are the
        dominant features of a truncated filtration: a 60-point circle fitted
        with ``max_edge_length=1.0`` has exactly one H₁ class and it is
        essential.  Set ``False`` to restrict ``features_`` to finite bars.

        Note that with the default ``max_edge_length=inf`` the complex
        eventually becomes a full simplex, every class dies, and no essential
        classes arise at all — so this flag only changes results for callers
        who truncate the filtration.
    max_points : int or None, default None
        Hard cap on ``n_points``.  ``None`` (the default) means no cap, which
        preserves historical behaviour.  Above 5000 points a
        ``ResourceWarning`` reports the projected cost of the float64
        distance matrix, which grows as O(n²) — 5000 points is roughly
        200 MB.  CPython suppresses ``ResourceWarning`` under default
        filters; run with ``-W default::ResourceWarning`` to see it.

    Notes
    -----
    See the package docstring for important notes on float32 precision and the
    difference between this heuristic and algebraic cycle representatives.
    """

    def __init__(
        self,
        min_persistence: float = 0.0,
        max_edge_length: float = np.inf,
        metric: str = "euclidean",
        coeff: int = 2,
        n_threads: int = 1,
        reconstruct_cycles: bool = True,
        include_essential: bool = True,
        max_points: Optional[int] = None,
    ) -> None:
        if metric not in ("euclidean", "precomputed"):
            raise ValueError(
                f"metric must be 'euclidean' or 'precomputed', got {metric!r}."
            )
        self.min_persistence = min_persistence
        self.max_edge_length = max_edge_length
        self.metric = metric
        self.coeff = coeff
        self.n_threads = n_threads
        self.reconstruct_cycles = reconstruct_cycles
        self.include_essential = include_essential
        self.max_points = max_points

        self.point_cloud_: Optional[np.ndarray] = None
        self.diagrams_: Optional[List[np.ndarray]] = None
        self.features_: List[CycleFeature] = []
        self._dist_matrix_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "RepresentativeCycles":
        """Compute persistent homology and extract representative H₁ cycles.

        Parameters
        ----------
        X : np.ndarray
            * ``metric='euclidean'`` — shape ``(n_points, n_dims)``, a point
              cloud in any-dimensional Euclidean space.
            * ``metric='precomputed'`` — shape ``(n_points, n_points)``,
              a symmetric distance matrix with zeros on the diagonal.

            .. note::
               Internally, ``X`` is cast to **float32** for ``ripser_parallel``
               (which requires it), but inter-point distances for Dijkstra cycle
               reconstruction are computed in **float64** to preserve accuracy.
               If your input is already a float32 array and precision at the
               boundary of adjacent floating-point values matters, consider
               up-casting to float64 before calling ``fit``.

        Returns
        -------
        self
        """
        # Validate before allocating: the distance matrix is O(n²) float64,
        # so a bad input should be rejected while it is still cheap.
        if self.metric == "precomputed":
            X = validate_precomputed(X)
            check_size(X.shape[0], self.max_points)
            self._dist_matrix_ = X
            self.point_cloud_ = self._mds_embed(X)
        else:
            X = validate_point_cloud(X)
            check_size(X.shape[0], self.max_points)
            self._dist_matrix_ = cdist(X, X)  # float64 pairwise distances
            self.point_cloud_ = X

        thresh = (
            self.max_edge_length
            if np.isfinite(self.max_edge_length)
            else np.inf
        )

        # ripser_parallel needs float32; we cast only here
        result = ripser_parallel(
            X.astype(np.float32),
            maxdim=1,
            thresh=thresh,
            coeff=self.coeff,
            n_threads=self.n_threads,
            return_generators=True,
            metric=self.metric,
        )

        self.diagrams_ = result["dgms"]
        gens = result["gens"]

        # gens layout for maxdim=1:
        #   gens[0]  ndarray (n_H0_finite, 3)         — H0 finite pairs
        #   gens[1]  list[ndarray (n_H1_finite, 4)]   — H1 finite pairs
        #   gens[2]  ndarray (n_H0_essential,)         — H0 essential
        #   gens[3]  list[ndarray (n_H1_essential, 2)] — H1 essential
        #
        # gens[1][0][i] = [birth_v0, birth_v1, death_v0, death_v1]
        # gens[3][0][i] = [birth_v0, birth_v1]            (essential — no death)
        h1_dgm = self.diagrams_[1]
        finite_gens = (
            gens[1][0] if len(gens[1]) > 0 else np.empty((0, 4), dtype=int)
        )
        essential_gens = (
            gens[3][0]
            if self.include_essential and len(gens[3]) > 0
            else np.empty((0, 2), dtype=int)
        )

        # Pair generators to diagram rows by *value* rather than by position.
        # Row order agreement between `dgms` and `gens` is undocumented ripser
        # behaviour, and it demonstrably does not survive essential bars.
        paired = pair_generators(
            finite_gens, essential_gens, h1_dgm, self._dist_matrix_
        )

        self.features_ = []
        for row in paired:
            # min_persistence is applied *after* pairing: dropping a generator
            # earlier would shift every later generator's claim.
            # An essential class has infinite persistence and so is never
            # filtered out, whatever the threshold (F3).
            if (
                not row.is_essential
                and (row.death - row.birth) < self.min_persistence
            ):
                continue

            cycle_edges = np.empty((0, 2), dtype=int)
            cycle_path = np.empty(0, dtype=int)
            cycle_length = 0.0
            is_verified = False
            cycle_verts = np.array(list(row.birth_edge))

            if self.reconstruct_cycles:
                # The graph is cut at the exact float64 edge length, not at the
                # float32-derived diagram birth, which can sit ~1e-7 below it.
                u, v = row.birth_edge
                birth_radius = float(self._dist_matrix_[u, v])
                result = reconstruct_cycle(
                    self._dist_matrix_, row.birth_edge, birth_radius
                )
                cycle_edges = result.edges
                cycle_path = result.path
                cycle_length = result.length
                is_verified = result.is_verified
                if len(result.edges):
                    cycle_verts = result.vertices

            self.features_.append(
                CycleFeature(
                    index=row.diagram_index,
                    birth=row.birth,
                    death=row.death,
                    persistence=row.death - row.birth,
                    birth_edge=row.birth_edge,
                    death_edge=row.death_edge,
                    cycle_edges=cycle_edges,
                    cycle_vertices=cycle_verts,
                    cycle_length=cycle_length,
                    cycle_path=cycle_path,
                    is_essential=row.is_essential,
                    is_verified=is_verified,
                )
            )

        self.features_.sort(key=_feature_sort_key)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mds_embed(D: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Return a 2-D MDS embedding of *D* for visualisation purposes.

        Notes
        -----
        ``+inf`` entries are legal input — they mean "no edge between these
        points", the standard encoding for geodesic distances on a
        disconnected mesh — but scikit-learn's MDS rejects any non-finite
        value.  Since this embedding is used *only* to position dots on a
        plot and never for topology, infinite distances are replaced by twice
        the largest finite distance purely for layout.

        That substitution is a lie about the geometry, so it is stated rather
        than hidden: disconnected points are drawn far apart, but their drawn
        separation is arbitrary and carries no metric meaning.  Every
        topological quantity is computed from the original matrix.
        """
        from sklearn.manifold import MDS

        n = D.shape[0]
        if n < 2:
            return np.zeros((n, n_components))

        if not np.isfinite(D).all():
            finite = D[np.isfinite(D)]
            surrogate = 2.0 * finite.max() if finite.size else 1.0
            D = np.where(np.isfinite(D), D, surrogate)
        k = min(n_components, n - 1)
        mds = MDS(
            n_components=k,
            dissimilarity="precomputed",
            random_state=0,
            n_init=1,
            normalized_stress="auto",
        )
        coords = mds.fit_transform(D)
        if k < n_components:
            # pad with zeros if embedding dimension < requested
            coords = np.pad(coords, ((0, 0), (0, n_components - k)))
        return coords

    def _reconstruct_cycle(
        self, birth_edge: Tuple[int, int], birth_radius: float
    ) -> np.ndarray:
        """Backwards-compatible alias returning just the cycle's edge array.

        Reconstruction itself now returns a
        :class:`repcycles.reconstruction.CycleResult` carrying the loop's
        length and verification status; call
        :func:`repcycles.reconstruction.reconstruct_cycle` directly for those.
        """
        return reconstruct_cycle(
            self._dist_matrix_, birth_edge, birth_radius
        ).edges

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a table of all extracted H₁ features.

        The ``Ok`` column is the verification flag: ``yes`` means the
        reconstructed loop is closed over ℤ/2 and lies entirely within its
        birth radius, ``NO`` means it does not and must not be read as a
        representative cycle.  It is shown for every feature rather than only
        on failure, so that a clean table is positive evidence rather than an
        absence of complaint.
        """
        if not self.features_:
            print("No H1 features found (try lowering min_persistence).")
            return
        print(
            f"{'#':>3}  {'Birth':>8}  {'Death':>8}  {'Persist':>8}  "
            f"{'BirthEdge':>11}  {'Edges':>6}  {'Length':>8}  {'Ok':>3}"
        )
        print("-" * 70)
        for f in self.features_:
            death_str = f"{f.death:.4f}" if np.isfinite(f.death) else "∞"
            pers_str = (
                f"{f.persistence:.4f}" if np.isfinite(f.persistence) else "∞"
            )
            print(
                f"{f.index:>3}  {f.birth:>8.4f}  {death_str:>8}  "
                f"{pers_str:>8}  "
                f"({f.birth_edge[0]:>3},{f.birth_edge[1]:>3})  "
                f"{f.n_edges:>6}  {f.cycle_length:>8.4f}  "
                f"{'yes' if f.is_verified else 'NO':>3}"
            )
        n_bad = sum(not f.is_verified for f in self.features_)
        if n_bad:
            print(
                f"\n{n_bad} of {len(self.features_)} cycles failed "
                "verification and are not valid representatives."
            )

    def to_dataframe(self):
        """Return the extracted features as a pandas ``DataFrame``.

        One row per feature, in ``features_`` order (essential first, then
        persistence descending).  Columns: ``index``, ``birth``, ``death``,
        ``persistence``, ``birth_u``, ``birth_v``, ``death_u``, ``death_v``,
        ``n_edges``, ``cycle_length``, ``is_essential``, ``is_verified``.

        Essential features carry ``inf`` in ``death`` and ``persistence``;
        they are not dropped or coerced, so ``df[np.isfinite(df.death)]`` is
        the caller's explicit choice rather than something done behind their
        back.

        Raises
        ------
        ImportError
            If pandas is not installed.  It is an optional dependency —
            everything else in this package works without it.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe() requires pandas, which is an optional "
                "dependency of this package. Install it: pip install pandas"
            ) from exc

        return pd.DataFrame(
            [
                {
                    "index": f.index,
                    "birth": f.birth,
                    "death": f.death,
                    "persistence": f.persistence,
                    "birth_u": f.birth_edge[0],
                    "birth_v": f.birth_edge[1],
                    "death_u": f.death_edge[0],
                    "death_v": f.death_edge[1],
                    "n_edges": f.n_edges,
                    "cycle_length": f.cycle_length,
                    "is_essential": f.is_essential,
                    "is_verified": f.is_verified,
                }
                for f in self.features_
            ],
            columns=[
                "index", "birth", "death", "persistence",
                "birth_u", "birth_v", "death_u", "death_v",
                "n_edges", "cycle_length", "is_essential", "is_verified",
            ],
        )

    # ------------------------------------------------------------------
    # Visualisation (delegated — imports are function-local by design)
    # ------------------------------------------------------------------

    def plot_matplotlib(
        self,
        max_cycles: int = 6,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Representative H₁ Cycles",
        save_path: Optional[str] = None,
    ):
        """Point cloud + persistence diagram + one panel per representative cycle.

        See :func:`repcycles.plotting.panels.plot_matplotlib`.
        """
        from .plotting.panels import plot_matplotlib as _plot

        return _plot(
            self,
            max_cycles=max_cycles,
            figsize=figsize,
            title=title,
            save_path=save_path,
        )

    def plot_barcode(
        self,
        figsize: Tuple[int, int] = (8, 4),
        save_path: Optional[str] = None,
    ):
        """Horizontal H₁ barcode sorted by persistence.

        See :func:`repcycles.plotting.barcode.plot_barcode`.
        """
        if self.diagrams_ is None:
            raise RuntimeError("Call fit() first.")
        from .palette import cycle_colors
        from .plotting.barcode import plot_barcode as _plot

        return _plot(
            self.diagrams_[1],
            features=self.features_,
            colors=cycle_colors(self.features_),
            figsize=figsize,
            save_path=validate_save_path(save_path),
        )

    def plot_plotly(
        self,
        max_cycles: int = 6,
        title: str = "Representative H₁ Cycles (Interactive)",
        save_html: Optional[str] = None,
        show_skeleton: bool = False,
        skeleton_max_edges: int = 20_000,
    ):
        """Interactive Plotly figure with toggleable cycle traces.

        See :func:`repcycles.plotting.interactive.plot_plotly`.
        """
        from .plotting.interactive import plot_plotly as _plot

        return _plot(
            self,
            max_cycles=max_cycles,
            title=title,
            save_html=validate_save_path(save_html),
            show_skeleton=show_skeleton,
            skeleton_max_edges=skeleton_max_edges,
        )
