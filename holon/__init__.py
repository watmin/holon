# Holon: Programmatic Neural Memory
# Version: 0.1.0
#
# Primary Interface:
#   HolonClient - The main class for all Holon operations
#
# Typical Usage:
#   from holon import HolonClient
#   client = HolonClient()
#   vec = client.encode({"type": "billing"})
#   accum = client.create_accumulator()
#   accum = client.accumulate(accum, vec)

from .client import HolonClient  # noqa: F401 - re-exported for public API
from .cpu_store import CPUStore, Store  # noqa: F401 - re-exported for public API

# Distance metrics
from .distance import (  # noqa: F401 - re-exported for public API
    DistanceEngine,
    DistanceMetric,
    compare_metrics,
    get_recommended_metric,
)

# Encoder and encoding modes
from .encoder import (  # noqa: F401 - re-exported for public API
    Encoder,
    ListEncodeMode,
    MathematicalPrimitive,
    TimeResolution,
)

# Vector managers
from .vector_manager import (  # noqa: F401 - re-exported for public API
    DeterministicVectorManager,
    VectorManager,
)

# Walkable interface for zero-serialization encoding
from .walkable import (  # noqa: F401 - re-exported for public API
    Walkable,
    WalkableDict,
    WalkableList,
    WalkableScalar,
    WalkableSet,
    WalkType,
    as_walkable,
    is_walkable,
    register_walkable,
    register_walkable_adapter,
    walk_iter,
)

# Optional Qdrant backend (requires qdrant-client)
try:
    from .qdrant_store import QdrantStore  # noqa: F401 - re-exported for public API
except ImportError:
    QdrantStore = None  # type: ignore

__version__ = "0.1.0"


# Convenience function for quick start
def create_client(dimensions: int = 4096) -> HolonClient:
    """
    Create a standalone HolonClient with default settings.

    This is the quickest way to get started with Holon.

    Args:
        dimensions: Vector dimensionality (default 4096)

    Returns:
        Configured HolonClient

    Example:
        >>> from holon import create_client
        >>> client = create_client()
        >>> vec = client.encode({"type": "billing"})
    """
    return HolonClient(dimensions=dimensions)
