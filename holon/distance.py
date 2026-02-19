"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.distance import *

This file will be removed in version 0.2.0.
"""

# Re-export from the new location
from .kernel.distance import *  # noqa: F401, F403

__all__ = [
    'DistanceEngine',
    'DistanceMetric',
    'cosine_similarity',
    'compare_metrics',
    'get_recommended_metric',
    'significance',
]
