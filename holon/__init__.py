# Holon: Programmatic Neural Memory
# Version: 0.1.0

from .client import HolonClient  # noqa: F401 - re-exported for public API
from .cpu_store import CPUStore, Store  # noqa: F401 - re-exported for public API
from .encoder import TimeResolution  # noqa: F401 - re-exported for public API

# Optional Qdrant backend (requires qdrant-client)
try:
    from .qdrant_store import QdrantStore  # noqa: F401 - re-exported for public API
except ImportError:
    QdrantStore = None  # type: ignore

__version__ = "0.1.0"
