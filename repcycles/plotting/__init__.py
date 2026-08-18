"""Visualisation layer for representative H₁ cycles.

Split by *view* rather than lumped into one module: each view is an
independent unit of work with its own file, and only ``panels`` depends on
``repcycles.projection``.

Importing this package pulls in matplotlib.  ``repcycles.core`` therefore
imports it lazily, inside the ``plot_*`` methods.
"""

from .barcode import draw_barcode, plot_barcode
from .diagram import draw_persistence_diagram
from .interactive import plot_plotly
from .overview import plot_cycle, plot_overview
from .panels import draw_cycle_overlay, draw_cycle_panel, plot_matplotlib

__all__ = [
    "draw_persistence_diagram",
    "draw_barcode",
    "draw_cycle_panel",
    "draw_cycle_overlay",
    "plot_matplotlib",
    "plot_barcode",
    "plot_overview",
    "plot_cycle",
    "plot_plotly",
]
