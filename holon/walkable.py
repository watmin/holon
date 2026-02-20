"""Zero-serialization walkable encoding interface.

Alias for holon.kernel.walkable — both paths are supported.
"""

from .kernel.walkable import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "Walkable",
    "WalkableDict",
    "WalkableList",
    "WalkableScalar",
    "WalkableSet",
    "WalkType",
    "LinearScale",
    "LogScale",
    "TimeScale",
    "as_walkable",
    "is_walkable",
    "register_walkable",
    "register_walkable_adapter",
    "walk_iter",
]
