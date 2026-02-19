"""
DEPRECATED: This module exists for backward compatibility only.
Use: from holon.kernel.primitives import *

This file will be removed in version 0.2.0.
"""

# Re-export everything from the new location
from .kernel.primitives import *  # noqa: F401, F403

__all__ = ['bind', 'unbind', 'bundle', 'permute', 'negate', 'amplify', 'difference',
           'threshold_bipolar', 'prototype', 'prototype_add', 'resonance', 'blend',
           'cleanup', 'similarity_profile', 'attend', 'analogy', 'project',
           'conditional_bind', 'segment', 'invert', 'complexity', 'sparsify',
           'centroid', 'flip', 'topk_similar', 'similarity_matrix', 'entropy',
           'random_project', 'power', 'autocorrelate', 'cross_correlate', 'reject',
           'bundle_with_confidence', 'coherence', 'grover_amplify', 'drift_rate']
