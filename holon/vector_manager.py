"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.vector_manager import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.vector_manager import *  # noqa: F401, F403

__all__ = [
    'VectorManager',
    'DeterministicVectorManager',
]
