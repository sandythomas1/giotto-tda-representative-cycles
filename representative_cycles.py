"""Backwards-compatible import shim for the :mod:`repcycles` package.

``from representative_cycles import RepresentativeCycles`` is the documented
public API (README, examples, external users' scripts) and
``specs/constitution.md`` names it a stable contract.  The implementation now
lives in the :mod:`repcycles` package — computation in ``repcycles.core`` and
``repcycles.reconstruction``, visualisation in ``repcycles.plotting`` — but
this import path keeps working unchanged.

New code should prefer ``from repcycles import RepresentativeCycles``.
"""

from repcycles import CycleFeature, RepresentativeCycles, __version__

__all__ = ["RepresentativeCycles", "CycleFeature", "__version__"]
