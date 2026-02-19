"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.similarity import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.similarity import *  # noqa: F401, F403

# Private functions need explicit import for tests
from .kernel.similarity import (  # noqa: F401
    _find_similar_vectors_parallel,
    _find_similar_vectors_single,
)
