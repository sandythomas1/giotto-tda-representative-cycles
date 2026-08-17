"""The :class:`CycleFeature` data contract.

This module is deliberately dependency-free (numpy only): it is the shared
vocabulary between the computation layer (``repcycles.core``,
``repcycles.reconstruction``) and the visualisation layer
(``repcycles.plotting``), and neither may drag the other's dependencies in
through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


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
        ``death - birth`` (``np.inf`` for essential features).
    birth_edge : tuple[int, int]
        The edge ``(u, v)`` whose addition creates this H₁ class.
    death_edge : tuple[int, int]
        The edge ``(u, v)`` whose triangle kills this H₁ class.  For an
        essential feature this is ``(-1, -1)``: nothing kills it.
    cycle_edges : np.ndarray, shape (n_edges, 2)
        Full representative cycle computed via shortest-path reconstruction.
        Empty if ``reconstruct_cycles=False`` or reconstruction failed.
    cycle_vertices : np.ndarray, shape (n_vertices,)
        Unique vertex indices in the representative cycle (sorted; use
        :attr:`cycle_path` when traversal order matters).
    cycle_length : float
        Total geodesic length of the representative cycle — the sum of its
        edge lengths.  ``0.0`` when no cycle was reconstructed.  This is the
        quantity that is invariant across implementations: the shortest-path
        *distance* is unique even where the shortest *path* is not.
    cycle_path : np.ndarray, shape (n_vertices + 1,)
        Vertices in loop traversal order, with the first vertex repeated at
        the end so the array is directly plottable as a closed polygon.
        Empty when no cycle was reconstructed.
    is_essential : bool
        ``True`` for a class that never dies within the filtration
        (``death == np.inf``).  Essential classes come from ripser's
        ``gens[3]`` and are the dominant features of a truncated filtration.
    is_verified : bool
        ``True`` only when the reconstructed loop passed both checks: every
        vertex has even degree (closed over ℤ/2), and every edge is no longer
        than the birth radius.  ``False`` means the loop is reported but
        should not be trusted as a representative — see
        :class:`repcycles.errors.CycleReconstructionWarning`.

    Notes
    -----
    New fields are appended last with defaults, so positional construction
    through ``death_edge`` keeps working for existing callers.
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
    cycle_length: float = 0.0
    cycle_path: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    is_essential: bool = False
    is_verified: bool = False

    @property
    def n_edges(self) -> int:
        """Number of edges in the representative cycle."""
        return int(len(self.cycle_edges))
