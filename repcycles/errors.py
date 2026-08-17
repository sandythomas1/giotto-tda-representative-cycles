"""Exception and warning types.

``specs/constitution.md`` forbids silent degradation: when reconstruction
cannot produce a trustworthy cycle, that fact must reach the caller through
the type system or the warnings machinery — never through a ``print``.
"""

from __future__ import annotations


class RepCyclesError(Exception):
    """Base class for all errors raised by this package."""


class CycleReconstructionError(RepCyclesError):
    """A representative cycle could not be attributed to a persistence pair.

    Raised when a ripser generator cannot be matched to any row of the
    persistence diagram.  This is a hard error rather than a warning because
    the alternative — emitting a feature whose birth/death values belong to a
    *different* homology class — would silently produce a wrong figure.
    """


class CycleReconstructionWarning(UserWarning):
    """A cycle was reconstructed, but in a degraded form.

    Emitted when no path exists between the endpoints of the birth edge in the
    Rips graph at birth radius, so the "cycle" falls back to the birth edge
    alone.  The corresponding feature carries ``is_verified=False``.
    """
