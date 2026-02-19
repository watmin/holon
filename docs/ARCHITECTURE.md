# Holon Architecture

## Overview

Holon follows a **layered architecture** inspired by operating system design: a minimal, stable kernel with increasingly opinionated layers built on top. This design ensures the foundation remains stable while allowing experimentation and customization at higher levels.

## Three-Layer Design

```
┌─────────────────────────────────────────────┐
│   holon.highlevel                           │  ← Convenience & Composition
│   - HolonClient (unified facade)            │
│   - Query DSL ($or, guards, negations)      │
│   - JSON/EDN convenience methods            │
└─────────────────────────────────────────────┘
                    ↓ uses
┌─────────────────────────────────────────────┐
│   holon.memory                              │  ← Programmatic Neural Memory
│   - OnlineSubspace (CCIPCA manifolds)       │
│   - Engram/EngramLibrary (pattern memory)   │
│   - Anomaly detection & learning            │
└─────────────────────────────────────────────┘
                    ↓ uses
┌─────────────────────────────────────────────┐
│   holon.kernel                              │  ← Foundational VSA/HDC
│   - Primitives (bind, bundle, unbind, ...)  │
│   - Encoder (structural encoding)           │
│   - Store (CPUStore, backends)              │
│   - Scalar, vector, distance, similarity    │
│   - kernel/encoders/ (extended encoders)    │
│   - kernel/utils/ (atomizer, etc.)          │
└─────────────────────────────────────────────┘
```

### Layer 1: `holon.kernel` - The Foundation

**Purpose**: Minimal, stable VSA/HDC primitives that rarely change.

**Key Modules**:
- `primitives.py` - Core VSA operations (~40 functions)
- `encoder.py` - Structure-preserving data encoding (1000 lines)
- `store.py` - Vector storage backends (CPUStore, etc.)
- `scalar.py` - Scalar encoding (log-scale, circular, positional)
- `accumulator.py` - Accumulator primitives for vector composition
- `vector_manager.py` - Deterministic vector allocation
- `walkable.py` - Zero-allocation encoding interface
- `distance.py` - Distance metrics and significance testing
- `similarity.py` - Similarity search algorithms

**Subpackages**:
- `kernel/utils/` - Utilities (atomizer for JSON/EDN parsing)
- `kernel/encoders/` - Extended encoders (domain, enhanced, semantic)

**Import Examples**:
```python
from holon.kernel import bind, bundle, Encoder, CPUStore
from holon.kernel.encoders import SemanticEncoder
```

**Design Principle**: No dependencies on `memory/` or `highlevel/`. Mirrors `holon-rs` API for cross-language parity.

### Layer 2: `holon.memory` - The Innovation

**Purpose**: Novel programmatic neural memory system built on kernel primitives.

**Key Modules**:
- `subspace.py` - `OnlineSubspace` for CCIPCA-based manifold learning
- `engram.py` - `Engram` and `EngramLibrary` for pattern snapshots

**Import Examples**:
```python
from holon.memory import OnlineSubspace, EngramLibrary
```

**Design Principle**: This is Holon's crown jewel. Uses kernel primitives to implement anomaly detection, pattern learning, and single-packet DDoS classification.

### Layer 3: `holon.highlevel` - The Convenience Layer

**Purpose**: Ergonomic APIs and composition for common use cases.

**Key Modules**:
- `client.py` - `HolonClient` unified facade

**Import Examples**:
```python
from holon.highlevel import HolonClient
```

**Design Principle**: Thin wrapper providing convenience methods (`insert_json`, `search_json`) and query DSL features ($markers, guards, negations).

## Import Patterns

### Explicit Layer Imports (Recommended for Libraries)

```python
# Clear, explicit, shows layer boundaries
from holon.kernel import bind, bundle, CPUStore
from holon.memory import OnlineSubspace
from holon.highlevel import HolonClient
```

**Benefits**:
- Clear layer dependencies
- Easier to understand architecture
- Better for library code

### Top-Level Convenience Imports (Quick Scripts)

```python
# Convenient for scripts and examples
from holon import bind, bundle, CPUStore, OnlineSubspace, HolonClient
```

**Benefits**:
- Less verbose
- Backward compatible with flat structure
- Good for quick scripts

### Accessing Extended Features

```python
# Extended encoders
from holon.kernel.encoders import SemanticEncoder, EnhancedEncoder

# Utilities
from holon.kernel.utils import parse_data, atomize
```

## Backward Compatibility

All modules from the original flat structure remain accessible via shims at the package root:

```python
# These still work (via shims)
from holon.primitives import bind      # → holon.kernel.primitives
from holon.client import HolonClient   # → holon.highlevel.client
from holon.subspace import OnlineSubspace  # → holon.memory.subspace
```

**Deprecation Timeline**: Shims will be removed in version 0.2.0 with appropriate warnings.

## Alternative Backends (Root Level)

Some modules remain at the package root because they are alternative implementations, not part of the core layers:

- `torchhd_encoder.py` (31K) - Alternative backend using TorchHD library
- `qdrant_store.py` (25K) - Qdrant vector database integration

These are kept separate to avoid bloating the kernel and because they are optional dependencies.

## Design Rationale

### Why Three Layers?

1. **Stability**: Kernel API is stable, changes are rare. Memory and highlevel can evolve faster.
2. **Clarity**: Clear boundaries make it obvious what depends on what.
3. **Testability**: Kernel can be tested independently of higher layers.
4. **Cross-Language**: Kernel mirrors `holon-rs` API, enabling language interop.
5. **Flexibility**: Users can choose their level of abstraction.

### Why `accumulator` in kernel, not memory?

Accumulators are primitive operations (like `bundle`) that compose vectors. They don't depend on learning or memory concepts, so they belong in the kernel.

### Why `kernel/encoders/` namespace?

The extended encoders (`SemanticEncoder`, `EnhancedEncoder`, `MathematicalPatternEncoder`) build on the base `Encoder` but add domain-specific capabilities. They're kernel extensions, not optional backends, so they live in a `kernel/encoders/` namespace.

## Dependencies

```
holon.highlevel
    ↓ depends on
holon.memory
    ↓ depends on
holon.kernel
    ↓ depends on
numpy, (optional: cupy)
```

**No circular dependencies**: Each layer only imports from layers below it.

## File Organization Summary

```
holon/
├── __init__.py (7K)           # Top-level exports for convenience
├── [16 shim files] (~300B)    # Backward compatibility (removed in 0.2.0)
├── kernel/                    # 14 modules, foundational primitives
├── memory/                    # 2 modules, novel memory system
├── highlevel/                 # 1 module, convenience facade
├── qdrant_store.py (25K)      # Alternative: Qdrant integration
└── torchhd_encoder.py (31K)   # Alternative: TorchHD backend
```

**Result**: Clean root with just 2 large files (alternatives) + shims for backward compat.

## Testing

All 603 tests pass with the layered architecture. The refactor was behavior-preserving:

- Unit tests verify each layer independently
- Integration tests verify cross-layer interactions
- Challenge scripts (batches 012-017) all pass
- No breaking changes to public API

## Migration Guide (for 0.2.0)

When shims are removed in version 0.2.0, update imports:

```python
# Old (will break in 0.2.0)
from holon.primitives import bind
from holon.client import HolonClient

# New (works now and after 0.2.0)
from holon.kernel import bind
from holon.highlevel import HolonClient

# Or use top-level convenience (also works)
from holon import bind, HolonClient
```

Deprecation warnings will guide you before 0.2.0 is released.
