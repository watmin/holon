"""
Holon Kernel Encoders - Extended Encoding Capabilities

This module provides specialized encoders that extend the base kernel encoder:
- Domain encoders: Mathematical pattern recognition (fractals, waves, graphs)
- Enhanced encoder: Advanced geometric primitives for substring matching
- Semantic encoder: Combines domain encoders for rich mathematical semantics

All encoders build on the kernel's foundational primitives.
"""

from .domain import (
    MathematicalPatternEncoder,
    GraphTopologyEncoder,
)
from .enhanced import (
    EnhancedEncoder,
    EnhancedListEncodeMode,
)
from .semantic import (
    SemanticEncoder,
)

__all__ = [
    # Domain encoders
    "MathematicalPatternEncoder",
    "GraphTopologyEncoder",
    # Enhanced encoder
    "EnhancedEncoder",
    "EnhancedListEncodeMode",
    # Semantic encoder
    "SemanticEncoder",
]
