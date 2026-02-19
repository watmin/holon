"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.walkable import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.walkable import *  # noqa: F401, F403

__all__ = [
    'Walkable',
    'WalkableDict',
    'WalkableList',
    'WalkableScalar',
    'WalkableSet',
    'WalkType',
    'LinearScale',
    'LogScale',
    'as_walkable',
    'is_walkable',
    'register_walkable',
    'register_walkable_adapter',
    'walk_iter',
]
