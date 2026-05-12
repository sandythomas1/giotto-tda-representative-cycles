"""
Representative Cycles for giotto-tda
=====================================
Contribution to giotto-tda: visualizing representative cycles extracted
from persistent homology via gph.ripser_parallel(return_generators=True).

Background
----------
``gph.ripser_parallel(return_generators=True)`` returns *persistent pairs*:
for each finite H_d bar, the birth simplex and death simplex that generate it.
For H_1 (loops), this is one birth edge (the edge that creates the loop) and
one death edge (representing the triangle that kills it).

From the birth edge we reconstruct the actual representative cycle via a
shortest-path search on the 1-skeleton of the Vietoris-Rips complex built at
the birth radius.  The cycle is: (shortest path from u to v in the graph
without the birth edge) + (birth edge itself).

Usage
-----
>>> from representative_cycles import RepresentativeCycles
>>> import numpy as np
>>> theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
>>> X = np.column_stack([np.cos(theta), np.sin(theta)])
>>> rc = RepresentativeCycles(min_persistence=0.2)
>>> rc.fit(X)
>>> rc.summary()
>>> rc.plot_matplotlib()
>>> rc.plot_plotly()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
from scipy.spatial.distance import cdist
from gph import ripser_parallel


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CycleFeature:
    """One persistent H_1 feature with its representative cycle.

    Attributes
    ----------
    index : int
        Row index in the H_1 persistence diagram.
    birth : float
        Filtration value at which the feature is born.
    death : float
        Filtration value at which the feature dies (np.inf = essential).
    persistence : float
        death - birth.
    birth_edge : tuple[int, int]
        The edge (u, v) whose addition creates this H_1 class.
    death_edge : tuple[int, int]
        The edge (u, v) whose triangle kills this H_1 class.
    cycle_edges : np.ndarray, shape (n_edges, 2)
        Full representative cycle computed via shortest-path reconstruction.
        May be empty if reconstruction was not requested or failed.
    cycle_vertices : np.ndarray, shape (n_vertices,)
        Unique vertex indices in the representative cycle.
    """
    index: int
    birth: float
    death: float
    persistence: float
    birth_edge: Tuple[int, int] = (0, 0)
    death_edge: Tuple[int, int] = (0, 0)
    cycle_edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    cycle_vertices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RepresentativeCycles:
    """Compute and visualize representative H_1 cycles from a point cloud.

    Parameters
    ----------
    min_persistence : float, default 0.0
        Discard H_1 features with persistence below this threshold.
    max_edge_length : float, default np.inf
        Maximum edge length in the Vietoris-Rips filtration.
    coeff : int, default 2
        Coefficient field (prime). Use 2 for Z/2Z.
    n_threads : int, default 1
        Threads for ripser_parallel.
    reconstruct_cycles : bool, default True
        If True, compute the full representative cycle for each feature
        via shortest-path reconstruction on the Rips graph at birth radius.
        Slightly slower but yields geometrically meaningful loops.
    """

    def __init__(
        self,
        min_persistence: float = 0.0,
        max_edge_length: float = np.inf,
        coeff: int = 2,
        n_threads: int = 1,
        reconstruct_cycles: bool = True,
    ):
        self.min_persistence = min_persistence
        self.max_edge_length = max_edge_length
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
        """Compute persistent homology and extract representative H_1 cycles.

        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_dims)
            Input point cloud.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        self.point_cloud_ = X

        thresh = self.max_edge_length if np.isfinite(self.max_edge_length) else np.inf
        result = ripser_parallel(
            X,
            maxdim=1,
            thresh=thresh,
            coeff=self.coeff,
            n_threads=self.n_threads,
            return_generators=True,
        )

        self.diagrams_ = result["dgms"]
        gens = result["gens"]

        # gens layout (maxdim=1):
        #   gens[0]  ndarray (n_H0_finite, 3):  H0 finite pairs
        #   gens[1]  list[ndarray (n_H1_finite, 4)]: H1 finite pairs (dim-1 list)
        #   gens[2]  ndarray (n_H0_essential,): H0 essential
        #   gens[3]  list[ndarray (n_H1_essential, 2)]: H1 essential
        #
        # Each row of gens[1][0] = [birth_v0, birth_v1, death_v0, death_v1]:
        #   birth_v0, birth_v1 = the edge whose addition creates the H1 class
        #   death_v0, death_v1 = edge representing the killing triangle

        h1_dgm = self.diagrams_[1]               # shape (n_H1, 2)
        h1_pairs = gens[1][0] if len(gens[1]) > 0 else np.empty((0, 4), dtype=int)
        # h1_pairs[i] aligns with h1_dgm[i]

        # Pre-compute distance matrix once for cycle reconstruction
        if self.reconstruct_cycles:
            self._dist_matrix_ = cdist(X.astype(np.float64), X.astype(np.float64))

        self.features_ = []
        for i, row in enumerate(h1_pairs):
            birth, death = float(h1_dgm[i, 0]), float(h1_dgm[i, 1])
            persistence = death - birth
            if persistence < self.min_persistence:
                continue

            birth_edge = (int(row[0]), int(row[1]))
            death_edge = (int(row[2]), int(row[3]))

            cycle_edges = np.empty((0, 2), dtype=int)
            if self.reconstruct_cycles:
                cycle_edges = self._reconstruct_cycle(birth_edge, birth)

            cycle_verts = np.unique(cycle_edges) if len(cycle_edges) else np.array(list(birth_edge))

            self.features_.append(CycleFeature(
                index=i,
                birth=birth,
                death=death,
                persistence=persistence,
                birth_edge=birth_edge,
                death_edge=death_edge,
                cycle_edges=cycle_edges,
                cycle_vertices=cycle_verts,
            ))

        # Sort by persistence descending
        self.features_.sort(key=lambda f: f.persistence, reverse=True)
        return self

    def _reconstruct_cycle(
        self, birth_edge: Tuple[int, int], birth_radius: float
    ) -> np.ndarray:
        """Compute the representative cycle via shortest-path reconstruction.

        Build the 1-skeleton of the Rips complex at `birth_radius` (all edges
        with length <= birth_radius), find the shortest path from u to v that
        avoids the birth edge, then close it with the birth edge itself.

        Returns edges as an (n, 2) int array, or just the birth edge if no
        path exists.
        """
        import heapq

        u, v = birth_edge
        D = self._dist_matrix_
        n = D.shape[0]

        # Build adjacency list for the Rips graph at birth_radius
        # Exclude the birth edge so the path must go "around"
        adj: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i == u and j == v:
                    continue
                if j == u and i == v:
                    continue
                if D[i, j] <= birth_radius:
                    adj[i].append((D[i, j], j))
                    adj[j].append((D[i, j], i))

        # Dijkstra from u to v
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        dist[u] = 0.0
        heap = [(0.0, u)]

        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist[cur]:
                continue
            if cur == v:
                break
            for w, nb in adj[cur]:
                nd = d + w
                if nd < dist[nb]:
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(heap, (nd, nb))

        if dist[v] == np.inf:
            # No path without the birth edge — return just the birth edge
            return np.array([[u, v]])

        # Trace back path
        path = []
        cur = v
        while cur != u:
            path.append(cur)
            cur = prev[cur]
        path.append(u)
        path.reverse()

        # Convert path to edges and append the closing birth edge
        edges = [[path[k], path[k + 1]] for k in range(len(path) - 1)]
        edges.append([u, v])
        return np.array(edges, dtype=int)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a table of all extracted H_1 features."""
        if not self.features_:
            print("No H_1 features found (try lowering min_persistence).")
            return
        print(f"{'#':>3}  {'Birth':>8}  {'Death':>8}  {'Persist':>8}  "
              f"{'BirthEdge':>12}  {'CycleLen':>8}")
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
        title: str = "Representative H₁ Cycles",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Point cloud + persistence diagram + one panel per representative cycle.

        Parameters
        ----------
        max_cycles : int, default 6
        figsize : tuple, optional
        title : str
        save_path : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.point_cloud_ is None:
            raise RuntimeError("Call fit() before plot_matplotlib().")

        cycles = self.features_[:max_cycles]
        n_panels = len(cycles) + 1          # +1 for persistence diagram
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
        ax.set_title("Persistence Diagram (H₁)", fontsize=11)
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

        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)

        scatter = ax.scatter(
            b, d, c=pers, cmap="plasma", s=55, zorder=3,
            edgecolors="k", linewidths=0.4,
        )
        plt.colorbar(scatter, ax=ax, label="Persistence", shrink=0.85)

        # highlight features that passed the min_persistence filter
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

        # background cloud
        ax.scatter(coords[:, 0], coords[:, 1], s=10, c="lightgray", zorder=1)

        # cycle edges
        if len(feature.cycle_edges) > 0:
            segs = [[coords[e[0]], coords[e[1]]] for e in feature.cycle_edges]
            # birth edge drawn thicker / dashed to distinguish
            birth_set = {(min(feature.birth_edge), max(feature.birth_edge))}
            normal_segs, birth_segs = [], []
            for e in feature.cycle_edges:
                key = (min(e[0], e[1]), max(e[0], e[1]))
                (birth_segs if key in birth_set else normal_segs).append(
                    [coords[e[0]], coords[e[1]]]
                )

            if normal_segs:
                ax.add_collection(mc.LineCollection(
                    normal_segs, colors=[color], linewidths=2.0, zorder=2, alpha=0.85,
                ))
            if birth_segs:
                ax.add_collection(mc.LineCollection(
                    birth_segs, colors=["black"], linewidths=2.5, zorder=3,
                    linestyles="dashed",
                ))

        # cycle vertices
        cv = feature.cycle_vertices
        ax.scatter(
            coords[cv, 0], coords[cv, 1],
            s=40, c=[color], zorder=4, edgecolors="k", linewidths=0.4,
        )

        # birth/death vertex markers
        u, v = feature.birth_edge
        ax.scatter(
            [coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
            s=80, c="black", marker="*", zorder=5,
        )

        death_str = f"{feature.death:.3f}" if np.isfinite(feature.death) else "∞"
        ax.set_title(
            f"Rank {rank} cycle  (H₁ #{feature.index})\n"
            f"birth={feature.birth:.3f}  death={death_str}\n"
            f"p={feature.persistence:.3f}  |cycle|={len(feature.cycle_edges)} edges",
            fontsize=8,
        )
        ax.set_aspect("equal")
        ax.autoscale_view()

        # legend patch
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color=color, lw=2, label="cycle path"),
                Line2D([0], [0], color="black", lw=2, ls="--", label="birth edge"),
            ],
            fontsize=7, loc="best",
        )

    # ------------------------------------------------------------------
    # Barcode
    # ------------------------------------------------------------------

    def plot_barcode(
        self,
        figsize: Tuple[int, int] = (8, 4),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Horizontal H_1 barcode sorted by persistence."""
        if self.diagrams_ is None:
            raise RuntimeError("Call fit() first.")

        dgm = self.diagrams_[1]
        if len(dgm) == 0:
            raise ValueError("No H₁ features found.")

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
        ax.set_title("H₁ Barcode  (sorted by persistence)")
        sm = cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(pers.min(), pers.max()))
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
        title: str = "Representative H₁ Cycles (Interactive)",
        save_html: Optional[str] = None,
    ):
        """Interactive Plotly figure with toggleable cycle traces.

        Each cycle's edges can be shown/hidden via the legend.
        Supports 2-D and 3-D point clouds.

        Parameters
        ----------
        max_cycles : int
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

        # --- background cloud ---
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode="markers",
                marker=dict(size=2.5, color="lightgray", opacity=0.5),
                name="Point cloud",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode="markers",
                marker=dict(size=5, color="lightgray", opacity=0.5),
                name="Point cloud",
            ))

        for i, feature in enumerate(cycles):
            color = palette[i % len(palette)]
            death_str = f"{feature.death:.3f}" if np.isfinite(feature.death) else "∞"
            grp = f"cycle{i}"
            label = (
                f"H₁ #{feature.index} | "
                f"b={feature.birth:.3f} d={death_str} "
                f"p={feature.persistence:.3f}"
            )

            # --- cycle edges ---
            if len(feature.cycle_edges) > 0:
                birth_set = {(min(*feature.birth_edge), max(*feature.birth_edge))}
                normal_x, normal_y, normal_z = [], [], []
                birth_x, birth_y, birth_z = [], [], []

                for e in feature.cycle_edges:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    target_x = birth_x if key in birth_set else normal_x
                    target_y = birth_y if key in birth_set else normal_y
                    target_z = birth_z if key in birth_set else normal_z

                    if is_3d:
                        target_x += [X[e[0], 0], X[e[1], 0], None]
                        target_y += [X[e[0], 1], X[e[1], 1], None]
                        target_z += [X[e[0], 2], X[e[1], 2], None]
                    else:
                        target_x += [X[e[0], 0], X[e[1], 0], None]
                        target_y += [X[e[0], 1], X[e[1], 1], None]

                def _add_edge_trace(ex, ey, ez, is_birth):
                    dash = "dash" if is_birth else "solid"
                    w = 4 if is_birth else 3
                    c = "black" if is_birth else color
                    nm = f"{'birth edge' if is_birth else label}"
                    if is_3d:
                        fig.add_trace(go.Scatter3d(
                            x=ex, y=ey, z=ez, mode="lines",
                            line=dict(color=c, width=w, dash=dash),
                            name=nm, legendgroup=grp, showlegend=is_birth,
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=ex, y=ey, mode="lines",
                            line=dict(color=c, width=w, dash=dash),
                            name=nm, legendgroup=grp, showlegend=is_birth,
                        ))

                if normal_x:
                    _add_edge_trace(normal_x, normal_y, normal_z, False)
                if birth_x:
                    _add_edge_trace(birth_x, birth_y, birth_z, True)

            # --- cycle vertices ---
            cv = feature.cycle_vertices
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=X[cv, 0], y=X[cv, 1], z=X[cv, 2],
                    mode="markers",
                    marker=dict(size=5, color=color),
                    name=label, legendgroup=grp, showlegend=True,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=X[cv, 0], y=X[cv, 1],
                    mode="markers",
                    marker=dict(size=9, color=color, symbol="circle"),
                    name=label, legendgroup=grp, showlegend=True,
                ))

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
