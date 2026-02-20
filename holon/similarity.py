"""Similarity metrics and engines.

Alias for holon.kernel.similarity — both paths are supported.
"""

# Private functions need explicit import for tests
from .kernel.similarity import *  # noqa: F401, F403
from .kernel.similarity import (  # noqa: F401
    _find_similar_vectors_parallel,
    _find_similar_vectors_single,
)
