"""Core VSA/HDC algebraic primitives.

Alias for holon.kernel.primitives — both paths are supported.
"""

from .kernel.primitives import *  # noqa: F401, F403

__all__ = [  # noqa: F405
    "bind",
    "unbind",
    "bundle",
    "permute",
    "negate",
    "amplify",
    "difference",
    "threshold_bipolar",
    "prototype",
    "prototype_add",
    "resonance",
    "blend",
    "cleanup",
    "similarity_profile",
    "attend",
    "analogy",
    "project",
    "conditional_bind",
    "segment",
    "invert",
    "complexity",
    "sparsify",
    "centroid",
    "flip",
    "topk_similar",
    "similarity_matrix",
    "entropy",
    "random_project",
    "power",
    "autocorrelate",
    "cross_correlate",
    "reject",
    "bundle_with_confidence",
    "coherence",
    "grover_amplify",
    "drift_rate",
]
