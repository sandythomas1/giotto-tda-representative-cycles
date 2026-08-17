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

This package provides that second step together with matplotlib and Plotly
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
    chosen by the matrix pivot algorithm.  This package instead uses a
    shortest-geodesic-path *heuristic*: given the birth edge (u, v), it finds
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

from .core import RepresentativeCycles
from .feature import CycleFeature

__all__ = ["RepresentativeCycles", "CycleFeature"]

__version__ = "0.2.0.dev0"
