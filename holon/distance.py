"""Distance metrics for vector comparison.

Alias for holon.kernel.distance — both paths are supported.
"""

from .kernel.distance import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "DistanceEngine",
    "DistanceMetric",
    "cosine_similarity",
    "compare_metrics",
    "get_recommended_metric",
    "significance",
]
