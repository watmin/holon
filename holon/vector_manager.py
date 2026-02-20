"""Deterministic atom-to-vector mapping.

Alias for holon.kernel.vector_manager — both paths are supported.
"""

from .kernel.vector_manager import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "VectorManager",
    "DeterministicVectorManager",
]
