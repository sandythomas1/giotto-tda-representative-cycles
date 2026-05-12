"""
Representative Cycles for giotto-tda
=====================================
An add-on layer for giotto-tda / gph-ripser that turns the raw
``return_generators=True`` simplex pairs into drawable representative
H₁ cycles on the original point cloud.

Background
----------
``gph.ripser_parallel(return_generators=True)`` returns *persistent pairs*:
for each finite H₁ bar, the birth edge (u, v) and the death
triangle that kills it.  That birth edge is *not* the cycle — it is the
single edge whose addition to the Vietoris-Rips complex first creates the
homological class.  To obtain a drawable, closed loop a second step is
needed: find the shortest path from u to v in the Rips 1-skeleton at the
birth radius (excluding the birth edge itself), then close the loop with
the birth edge.

This module provides that second step together with matplotlib and Plotly
visualisations, filling the gap that JavaPlex and GUDHI fill for Java /
MATLAB / C++ users, but entirely in Python.

Notes on correctness
--------------------
**float32 precision**
    gph-ripser requires float32 inputs.  Coordinates (or a precomputed
    distance matrix) are cast to float32 only for the ripser_parallel call;
    inter-point distances used for Dijkstra cycle reconstruction are kept in
    float64 so that path weights are as accurate as the original input.

**Cycle representative vs. algebraic representative**
    JavaPlex and GUDHI compute cycle representatives via algebraic
    boundary-matrix reduction, which yields the canonical representative
    chosen by the matrix pivot algorithm.  This module instead uses a
    shortest-geodesic-path heuristic: given the birth edge (u, v), it finds
    the shortest path from u to v in the Rips graph at birth radius, then
    appends (u, v) to close the loop.  Both approaches yield a *valid*
    representative of the same H₁ homology class; they may choose
    different edges.  The shortest-path approach tends to produce tighter,
    more visually interpretable loops.

Usage
-----
>>> import numpy as np
>>> from representative_cycles import RepresentativeCycles
>>> theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
>>> X = np.column_stack([np.cos(theta), np.sin(theta)])
>>> rc = RepresentativeCycles(min_persistence=0.2)
>>> rc.fit(X)
>>> rc.summary()
>>> rc.plot_matplotlib()
>>> rc.plot_plotly()

Precomputed distance matrix
>>> D = np.array([[0, 1, 2, 2],
...               [1, 0, 2, 1],
...               [2, 2, 0, 1],
...               [2, 1, 1, 0]], dtype=float)
>>> rc2 = RepresentativeCycles(metric="precomputed")
>>> rc2.fit(D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
from gph import ripser_parallel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CycleFeature:
    """One persistent H₁ feature with its representative cycle.

    Attributes
    ----------
    index : int
        Row index in the H₁ persistence diagram.
    birth : float
        Filtration value at which the feature is born.
    death : float
        Filtration value at which the feature dies (``np.inf`` = essential).
    persistence : float
        ``death - birth``.
    birth_edge : tuple[int, int]
        The edge ``(u, v)`` whose addition creates this H₁ class.
    death_edge : tuple[int, int]
        The edge ``(u, v)`` whose triangle kills this H₁ class.
    cycle_edges : np.ndarray, shape (n_edges, 2)
        Full representative cycle computed via shortest-path reconstruction.
        Empty if ``reconstruct_cycles=False`` or reconstruction failed.
    cycle_vertices : np.ndarray, shape (n_vertices,)
        Unique vertex indices in the representative cycle.
    """

    index: int
    birth: float
    death: float
    persistence: float
    birth_edge: Tuple[int, int] = (0, 0)
    death_edge: Tuple[int, int] = (0, 0)
    cycle_edges: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=int)
    )
    cycle_vertices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

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

    Notes
    -----
    See module docstring for important notes on float32 precision and the
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
        X = np.asarray(X, dtype=np.float64)  # keep float64 for distances

        if self.metric == "precomputed":
            self._validate_precomputed(X)
            self._dist_matrix_ = X
            self.point_cloud_ = self._mds_embed(X)
        else:
            if X.ndim != 2:
                raise ValueError(
                    f"For metric='euclidean', X must be 2-D (n_points, n_dims); "
                    f"got shape {X.shape}."
                )
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
        h1_dgm = self.diagrams_[1]
        h1_pairs = (
            gens[1][0] if len(gens[1]) > 0 else np.empty((0, 4), dtype=int)
        )

        self.features_ = []
        for i, row in enumerate(h1_pairs):
            birth = float(h1_dgm[i, 0])
            death = float(h1_dgm[i, 1])
            persistence = death - birth
            if persistence < self.min_persistence:
                continue

            birth_edge = (int(row[0]), int(row[1]))
            death_edge = (int(row[2]), int(row[3]))

            cycle_edges = np.empty((0, 2), dtype=int)
            if self.reconstruct_cycles:
                cycle_edges = self._reconstruct_cycle(birth_edge, birth)

            cycle_verts = (
                np.unique(cycle_edges)
                if len(cycle_edges)
                else np.array(list(birth_edge))
            )

            self.features_.append(
                CycleFeature(
                    index=i,
                    birth=birth,
                    death=death,
                    persistence=persistence,
                    birth_edge=birth_edge,
                    death_edge=death_edge,
                    cycle_edges=cycle_edges,
                    cycle_vertices=cycle_verts,
                )
            )

        self.features_.sort(key=lambda f: f.persistence, reverse=True)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_precomputed(X: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(
                f"When metric='precomputed', X must be a square 2-D matrix "
                f"(got shape {X.shape})."
            )
        if not np.allclose(X, X.T, atol=1e-6, rtol=0):
            raise ValueError(
                "Precomputed distance matrix must be symmetric "
                "(max asymmetry = "
                f"{np.max(np.abs(X - X.T)):.3e})."
            )
        if not np.allclose(np.diag(X), 0.0, atol=1e-8):
            raise ValueError(
                "Precomputed distance matrix must have zeros on the diagonal."
            )

    @staticmethod
    def _mds_embed(D: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Return a 2-D MDS embedding of *D* for visualisation purposes."""
        from sklearn.manifold import MDS

        n = D.shape[0]
        if n < 2:
            return np.zeros((n, n_components))
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
        """Compute the representative cycle via scipy C-level Dijkstra.

        **Algorithm**

        1. Build a boolean adjacency mask over all edges whose length
           ``<= birth_radius``, excluding the birth edge ``(u, v)``
           (so that the path is forced to go *around* the loop).
        2. Construct a CSR sparse graph from non-zero mask entries using
           NumPy advanced indexing — O(1) array ops instead of an O(n²)
           Python loop.
        3. Run ``scipy.sparse.csgraph.dijkstra`` (C-level) from ``u``; trace
           the predecessor array back to ``v``; close the loop with ``(u, v)``.

        Parameters
        ----------
        birth_edge : tuple[int, int]
            The edge ``(u, v)`` that creates the H₁ class.
        birth_radius : float
            The filtration value (edge length) at which the feature is born.

        Returns
        -------
        np.ndarray, shape (n_edges, 2)
            Edge list of the representative cycle.  Falls back to
            ``[[u, v]]`` (the birth edge alone) if no path exists, which can
            happen for disconnected filtration graphs.
        """
        u, v = birth_edge
        D = self._dist_matrix_
        n = D.shape[0]

        # Boolean adjacency mask: edges present in Rips(birth_radius),
        # excluding the birth edge and self-loops.
        mask = D <= birth_radius
        mask[u, v] = False
        mask[v, u] = False
        np.fill_diagonal(mask, False)

        # Build CSR sparse graph directly from non-zero mask indices.
        # This replaces the O(n²) Python adjacency-list loop entirely.
        rows, cols = np.nonzero(mask)
        data = D[rows, cols]
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))

        # scipy C-level Dijkstra — substantially faster than Python heapq
        dist_row, pred = _sp_dijkstra(
            graph, directed=False, indices=u, return_predecessors=True
        )

        if np.isinf(dist_row[v]):
            # No path without the birth edge — return just the birth edge
            return np.array([[u, v]], dtype=int)

        # Trace predecessor path from v back to u
        path: List[int] = []
        cur = int(v)
        while cur != u and cur >= 0:
            path.append(cur)
            cur = int(pred[cur])
        path.append(u)
        path.reverse()

        edges = [[path[k], path[k + 1]] for k in range(len(path) - 1)]
        edges.append([u, v])
        return np.array(edges, dtype=int)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a table of all extracted H₁ features."""
        if not self.features_:
            print("No H1 features found (try lowering min_persistence).")
            return
        print(
            f"{'#':>3}  {'Birth':>8}  {'Death':>8}  {'Persist':>8}  "
            f"{'BirthEdge':>12}  {'CycleLen':>8}"
        )
        print("-" * 62)
        for f in self.features_:
            death_str = f"{f.death:.4f}" if np.isfinite(f.death) else "   inf"
            print(
                f"{f.index:>3}  {f.birth:>8.4f}  {death_str:>8}  "
                f"{f.persistence:>8.4f}  "
                f"({f.birth_edge[0]:>3},{f.birth_edge[1]:>3})  "
                f"{len(f.cycle_edges):>8}"
            )

    # ------------------------------------------------------------------
    # Matplotlib: main figure
    # ------------------------------------------------------------------

    def plot_matplotlib(
        self,
        max_cycles: int = 6,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Representative H\u2081 Cycles",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Point cloud + persistence diagram + one panel per representative cycle.

        Parameters
        ----------
        max_cycles : int, default 6
            Maximum number of cycles to draw.
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.
        title : str
            Figure super-title.
        save_path : str, optional
            If given, save the figure to this path at 150 DPI.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.point_cloud_ is None:
            raise RuntimeError("Call fit() before plot_matplotlib().")

        cycles = self.features_[:max_cycles]
        n_panels = len(cycles) + 1  # +1 for persistence diagram
        if figsize is None:
            figsize = (4 * n_panels + 2, 5)

        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        self._draw_persistence_diagram(axes[0])

        palette = cm.tab10(np.linspace(0, 1, max(len(cycles), 1)))
        for i, (feature, color) in enumerate(zip(cycles, palette)):
            self._draw_cycle_panel(axes[i + 1], feature, color, rank=i + 1)

        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")

        return fig

    def _draw_persistence_diagram(self, ax: plt.Axes) -> None:
        ax.set_title("Persistence Diagram (H\u2081)", fontsize=11)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")

        if self.diagrams_ is None or len(self.diagrams_) < 2:
            return
        dgm = self.diagrams_[1]
        if len(dgm) == 0:
            return

        finite_mask = np.isfinite(dgm[:, 1])
        b, d = dgm[finite_mask, 0], dgm[finite_mask, 1]
        pers = d - b

        lo = min(b.min(), d.min())
        hi = max(b.max(), d.max())
        pad = max((hi - lo) * 0.05, 0.01)

        ax.plot(
            [lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=0.8, alpha=0.5,
        )
        scatter = ax.scatter(
            b, d, c=pers, cmap="plasma", s=55, zorder=3,
            edgecolors="k", linewidths=0.4,
        )
        plt.colorbar(scatter, ax=ax, label="Persistence", shrink=0.85)

        shown_idx = [f.index for f in self.features_]
        if shown_idx:
            ax.scatter(
                b[shown_idx], d[shown_idx],
                s=140, facecolors="none", edgecolors="limegreen",
                linewidths=1.5, zorder=4, label="shown",
            )
            ax.legend(fontsize=8)

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")

    def _draw_cycle_panel(
        self,
        ax: plt.Axes,
        feature: CycleFeature,
        color: np.ndarray,
        rank: int,
    ) -> None:
        X = self.point_cloud_
        coords = X[:, :2]

        ax.scatter(coords[:, 0], coords[:, 1], s=10, c="lightgray", zorder=1)

        if len(feature.cycle_edges) > 0:
            birth_set = {(min(feature.birth_edge), max(feature.birth_edge))}
            normal_segs: List = []
            birth_segs: List = []
            for e in feature.cycle_edges:
                key = (min(e[0], e[1]), max(e[0], e[1]))
                (birth_segs if key in birth_set else normal_segs).append(
                    [coords[e[0]], coords[e[1]]]
                )
            if normal_segs:
                ax.add_collection(
                    mc.LineCollection(
                        normal_segs, colors=[color],
                        linewidths=2.0, zorder=2, alpha=0.85,
                    )
                )
            if birth_segs:
                ax.add_collection(
                    mc.LineCollection(
                        birth_segs, colors=["black"],
                        linewidths=2.5, zorder=3, linestyles="dashed",
                    )
                )

        cv = feature.cycle_vertices
        ax.scatter(
            coords[cv, 0], coords[cv, 1],
            s=40, c=[color], zorder=4, edgecolors="k", linewidths=0.4,
        )

        u, v = feature.birth_edge
        ax.scatter(
            [coords[u, 0], coords[v, 0]],
            [coords[u, 1], coords[v, 1]],
            s=80, c="black", marker="*", zorder=5,
        )

        death_str = (
            f"{feature.death:.3f}" if np.isfinite(feature.death) else "\u221e"
        )
        ax.set_title(
            f"Rank {rank} cycle  (H\u2081 #{feature.index})\n"
            f"birth={feature.birth:.3f}  death={death_str}\n"
            f"p={feature.persistence:.3f}  |cycle|={len(feature.cycle_edges)} edges",
            fontsize=8,
        )
        ax.set_aspect("equal")
        ax.autoscale_view()

        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D([0], [0], color=color, lw=2, label="cycle path"),
                Line2D([0], [0], color="black", lw=2, ls="--", label="birth edge"),
            ],
            fontsize=7,
            loc="best",
        )

    # ------------------------------------------------------------------
    # Barcode
    # ------------------------------------------------------------------

    def plot_barcode(
        self,
        figsize: Tuple[int, int] = (8, 4),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal H₁ barcode sorted by persistence.

        Parameters
        ----------
        figsize : tuple, default (8, 4)
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.diagrams_ is None:
            raise RuntimeError("Call fit() first.")

        dgm = self.diagrams_[1]
        if len(dgm) == 0:
            raise ValueError("No H1 features found.")

        finite = dgm[np.isfinite(dgm[:, 1])]
        b, d = finite[:, 0], finite[:, 1]
        pers = d - b
        order = np.argsort(pers)[::-1]
        b, d, pers = b[order], d[order], pers[order]

        fig, ax = plt.subplots(figsize=figsize)
        colors = cm.plasma(pers / pers.max())
        for i, (bi, di, ci) in enumerate(zip(b, d, colors)):
            ax.plot([bi, di], [i, i], lw=2.5, color=ci, solid_capstyle="round")

        ax.set_yticks([])
        ax.set_xlabel("Filtration value")
        ax.set_title("H\u2081 Barcode  (sorted by persistence)")
        sm = cm.ScalarMappable(
            cmap="plasma", norm=plt.Normalize(pers.min(), pers.max())
        )
        plt.colorbar(sm, ax=ax, label="Persistence")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Plotly (interactive)
    # ------------------------------------------------------------------

    def plot_plotly(
        self,
        max_cycles: int = 6,
        title: str = "Representative H\u2081 Cycles (Interactive)",
        save_html: Optional[str] = None,
    ):
        """Interactive Plotly figure with toggleable cycle traces.

        Each cycle's edges can be shown/hidden via the legend.
        Supports 2-D and 3-D point clouds.

        Parameters
        ----------
        max_cycles : int, default 6
        title : str
        save_html : str, optional

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.colors import qualitative
        except ImportError:
            raise ImportError("Install plotly: pip install plotly")

        if self.point_cloud_ is None:
            raise RuntimeError("Call fit() before plot_plotly().")

        X = self.point_cloud_
        is_3d = X.shape[1] >= 3
        cycles = self.features_[:max_cycles]
        palette = qualitative.Plotly

        fig = go.Figure()

        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2],
                    mode="markers",
                    marker=dict(size=2.5, color="lightgray", opacity=0.5),
                    name="Point cloud",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1],
                    mode="markers",
                    marker=dict(size=5, color="lightgray", opacity=0.5),
                    name="Point cloud",
                )
            )

        for i, feature in enumerate(cycles):
            color = palette[i % len(palette)]
            death_str = (
                f"{feature.death:.3f}" if np.isfinite(feature.death) else "\u221e"
            )
            grp = f"cycle{i}"
            label = (
                f"H\u2081 #{feature.index} | "
                f"b={feature.birth:.3f} d={death_str} "
                f"p={feature.persistence:.3f}"
            )

            if len(feature.cycle_edges) > 0:
                birth_set = {
                    (min(*feature.birth_edge), max(*feature.birth_edge))
                }
                normal_x: List = []
                normal_y: List = []
                normal_z: List = []
                birth_x: List = []
                birth_y: List = []
                birth_z: List = []

                for e in feature.cycle_edges:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    tx = birth_x if key in birth_set else normal_x
                    ty = birth_y if key in birth_set else normal_y
                    tz = birth_z if key in birth_set else normal_z
                    if is_3d:
                        tx += [X[e[0], 0], X[e[1], 0], None]
                        ty += [X[e[0], 1], X[e[1], 1], None]
                        tz += [X[e[0], 2], X[e[1], 2], None]
                    else:
                        tx += [X[e[0], 0], X[e[1], 0], None]
                        ty += [X[e[0], 1], X[e[1], 1], None]

                def _add_edge_trace(
                    ex: List, ey: List, ez: List, is_birth: bool
                ) -> None:
                    dash = "dash" if is_birth else "solid"
                    w = 4 if is_birth else 3
                    c = "black" if is_birth else color
                    nm = "birth edge" if is_birth else label
                    if is_3d:
                        fig.add_trace(
                            go.Scatter3d(
                                x=ex, y=ey, z=ez, mode="lines",
                                line=dict(color=c, width=w, dash=dash),
                                name=nm, legendgroup=grp,
                                showlegend=is_birth,
                            )
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=ex, y=ey, mode="lines",
                                line=dict(color=c, width=w, dash=dash),
                                name=nm, legendgroup=grp,
                                showlegend=is_birth,
                            )
                        )

                if normal_x:
                    _add_edge_trace(normal_x, normal_y, normal_z, False)
                if birth_x:
                    _add_edge_trace(birth_x, birth_y, birth_z, True)

            cv = feature.cycle_vertices
            if is_3d:
                fig.add_trace(
                    go.Scatter3d(
                        x=X[cv, 0], y=X[cv, 1], z=X[cv, 2],
                        mode="markers",
                        marker=dict(size=5, color=color),
                        name=label, legendgroup=grp, showlegend=True,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X[cv, 0], y=X[cv, 1],
                        mode="markers",
                        marker=dict(size=9, color=color, symbol="circle"),
                        name=label, legendgroup=grp, showlegend=True,
                    )
                )

        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data") if is_3d else {},
            legend=dict(itemsizing="constant", groupclick="togglegroup"),
            template="plotly_white",
        )

        if save_html:
            fig.write_html(save_html)
            print(f"Saved interactive HTML to {save_html}")

        return fig
