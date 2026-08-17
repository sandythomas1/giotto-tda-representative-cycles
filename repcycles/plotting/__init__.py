"""Visualisation layer for representative H₁ cycles.

Split by *view* rather than lumped into one module: each view is an
independent unit of work with its own file, and only ``panels`` depends on
``repcycles.projection``.

Importing this package pulls in matplotlib.  ``repcycles.core`` therefore
imports it lazily, inside the ``plot_*`` methods.
"""

from .barcode import plot_barcode
from .diagram import draw_persistence_diagram
from .interactive import plot_plotly
from .panels import draw_cycle_panel, plot_matplotlib

__all__ = [
    "draw_persistence_diagram",
    "draw_cycle_panel",
    "plot_matplotlib",
    "plot_barcode",
    "plot_plotly",
]
