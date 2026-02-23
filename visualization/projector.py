"""
3D Projection for high-dimensional Holon vectors.

Uses orthogonal random projection to map high-D vectors to 3D space
while approximately preserving distances (Johnson-Lindenstrauss).
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from holon.encoder import Encoder
from holon.vector_manager import VectorManager


@dataclass
class Point3D:
    """A point in 3D visualization space."""

    id: str
    position: Tuple[float, float, float]
    point_type: str  # "atom", "binding", "composite", "accumulator"
    created_at: float
    last_seen: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    components: List[str] = field(default_factory=list)  # For composites/bindings
    level: int = 0  # Hierarchy level: 0=atom, 1=binding, 2=composite

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position": list(self.position),
            "type": self.point_type,
            "level": self.level,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "age": time.time() - self.created_at,
            "idle": time.time() - self.last_seen,
            "metadata": self.metadata,
            "components": self.components,
        }


class HolonProjector:
    """
    Projects high-dimensional Holon vectors into 3D space for visualization.

    Uses a fixed random projection matrix so the same vector always maps
    to the same 3D location (deterministic).
    """

    def __init__(
        self,
        dimensions: int = 4096,
        global_seed: int = 42,
        projection_seed: int = 12345,
    ):
        self.dimensions = dimensions
        self.global_seed = global_seed

        # Initialize Holon components
        self.vector_manager = VectorManager(
            dimensions=dimensions,
            backend="cpu",
            deterministic=True,
            global_seed=global_seed,
        )
        self.encoder = Encoder(self.vector_manager)

        # Create deterministic random projection matrix (D x 3)
        # Using orthogonal random projection for better distance preservation
        rng = np.random.RandomState(projection_seed)
        # Generate random matrix and orthogonalize
        random_matrix = rng.randn(dimensions, 3)
        # QR decomposition gives us orthonormal columns
        q, _ = np.linalg.qr(random_matrix)
        self.projection_matrix = q.astype(np.float32)

        # State tracking
        self.points: Dict[str, Point3D] = {}
        self.vectors: Dict[str, np.ndarray] = {}  # Store original vectors

    def project_vector(self, vector: np.ndarray) -> Tuple[float, float, float]:
        """Project a high-D vector to 3D coordinates."""
        # Normalize vector for consistent scale
        vec_float = vector.astype(np.float32)
        norm = np.linalg.norm(vec_float)
        if norm > 0:
            vec_float = vec_float / norm

        # Project to 3D
        coords = vec_float @ self.projection_matrix

        # Scale to reasonable range for visualization
        # Random projections give small values (~0.01-0.05), so we scale up significantly
        # to spread points across the visible space (-15 to 15)
        return tuple(float(c) * 300 for c in coords)

    def add_atom(self, atom_str: str, metadata: Optional[Dict] = None) -> Point3D:
        """Add a raw atom to the visualization (level 0)."""
        vector = self.vector_manager.get_vector(atom_str)
        position = self.project_vector(vector)
        now = time.time()

        point_id = f"atom:{atom_str}"

        if point_id in self.points:
            # Update existing point
            self.points[point_id].last_seen = now
            self.points[point_id].metadata.update(metadata or {})
        else:
            # Create new point
            self.points[point_id] = Point3D(
                id=point_id,
                position=position,
                point_type="atom",
                created_at=now,
                last_seen=now,
                metadata=metadata or {"value": atom_str},
                level=0,
            )
            self.vectors[point_id] = vector

        return self.points[point_id]

    def add_binding(
        self,
        binding_id: str,
        key_atom: str,
        value_atom: str,
        metadata: Optional[Dict] = None,
    ) -> Point3D:
        """
        Add a binding (key-value pair) to the visualization (level 1).

        This represents bind(key_vector, value_vector) in VSA terms.
        """
        # Get the vectors for key and value
        key_vector = self.vector_manager.get_vector(key_atom)
        value_vector = self.vector_manager.get_vector(value_atom)

        # Compute the bound vector (element-wise multiplication)
        bound_vector = self.encoder.bind(key_vector, value_vector)

        position = self.project_vector(bound_vector)
        now = time.time()

        point_id = f"binding:{binding_id}"
        components = [key_atom, value_atom]

        if point_id in self.points:
            self.points[point_id].last_seen = now
            self.points[point_id].position = position
            self.points[point_id].metadata.update(metadata or {})
        else:
            self.points[point_id] = Point3D(
                id=point_id,
                position=position,
                point_type="binding",
                created_at=now,
                last_seen=now,
                metadata=metadata or {},
                components=components,
                level=1,
            )
            self.vectors[point_id] = bound_vector

        return self.points[point_id]

    def add_composite(
        self,
        composite_id: str,
        data: Any,
        metadata: Optional[Dict] = None,
        components: Optional[List[str]] = None,
    ) -> Point3D:
        """Add a composite (encoded data structure) to the visualization (level 2)."""
        vector = self.encoder.encode_data(data)
        position = self.project_vector(vector)
        now = time.time()

        point_id = f"composite:{composite_id}"

        # Find component bindings if not provided
        if components is None:
            components = self._extract_binding_ids(data)

        if point_id in self.points:
            self.points[point_id].last_seen = now
            self.points[point_id].position = position
            self.points[point_id].components = components
            self.points[point_id].metadata.update(metadata or {})
        else:
            self.points[point_id] = Point3D(
                id=point_id,
                position=position,
                point_type="composite",
                created_at=now,
                last_seen=now,
                metadata=metadata or {"data": str(data)[:100]},
                components=components,
                level=2,
            )
            self.vectors[point_id] = vector

        return self.points[point_id]

    def _extract_binding_ids(self, data: Any) -> List[str]:
        """Extract binding IDs from a data structure."""
        bindings = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    bindings.append(f"{k}:{v}")
        return bindings

    def add_accumulator(
        self,
        acc_id: str,
        accumulator: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Point3D:
        """Add an accumulator state to the visualization."""
        # Normalize for projection
        normalized = self.encoder.normalize_accumulator(accumulator)
        position = self.project_vector(normalized)
        now = time.time()

        point_id = f"accumulator:{acc_id}"

        if point_id in self.points:
            self.points[point_id].last_seen = now
            self.points[point_id].position = position
            self.points[point_id].metadata.update(metadata or {})
        else:
            self.points[point_id] = Point3D(
                id=point_id,
                position=position,
                point_type="accumulator",
                created_at=now,
                last_seen=now,
                metadata=metadata or {},
            )
            self.vectors[point_id] = normalized

        return self.points[point_id]

    def touch(self, point_id: str) -> Optional[Point3D]:
        """Update last_seen time for a point."""
        if point_id in self.points:
            self.points[point_id].last_seen = time.time()
            return self.points[point_id]
        return None

    def remove_old(self, max_age: float = 60.0) -> List[str]:
        """Remove points older than max_age seconds. Returns removed IDs."""
        now = time.time()
        to_remove = [
            pid for pid, p in self.points.items() if now - p.last_seen > max_age
        ]
        for pid in to_remove:
            del self.points[pid]
            if pid in self.vectors:
                del self.vectors[pid]
        return to_remove

    def get_all_points(self) -> List[dict]:
        """Get all points as serializable dicts."""
        return [p.to_dict() for p in self.points.values()]

    def get_edges(self) -> List[dict]:
        """Get edges connecting the hierarchy: composite → bindings → atoms."""
        edges = []
        for point in self.points.values():
            if point.point_type == "composite" and point.components:
                # Composite → Binding edges
                for comp in point.components:
                    binding_id = f"binding:{comp}"
                    if binding_id in self.points:
                        edges.append(
                            {
                                "from": point.id,
                                "to": binding_id,
                                "from_pos": list(point.position),
                                "to_pos": list(self.points[binding_id].position),
                                "level": 2,  # composite to binding
                            }
                        )
            elif point.point_type == "binding" and point.components:
                # Binding → Atom edges
                for atom_name in point.components:
                    atom_id = f"atom:{atom_name}"
                    if atom_id in self.points:
                        edges.append(
                            {
                                "from": point.id,
                                "to": atom_id,
                                "from_pos": list(point.position),
                                "to_pos": list(self.points[atom_id].position),
                                "level": 1,  # binding to atom
                            }
                        )
        return edges

    def _extract_atoms(self, data: Any, prefix: str = "") -> List[str]:
        """Extract atom strings from a data structure, matching atom ID format."""
        atoms = []

        if isinstance(data, dict):
            for k, v in data.items():
                # For dict entries, create key:value atom IDs
                if isinstance(v, (str, int, float, bool)):
                    atoms.append(f"{k}:{v}")
                else:
                    atoms.extend(self._extract_atoms(v, prefix=f"{k}:"))
        elif isinstance(data, (list, tuple)):
            for item in data:
                atoms.extend(self._extract_atoms(item, prefix))
        elif isinstance(data, (set, frozenset)):
            for item in data:
                atoms.extend(self._extract_atoms(item, prefix))
        else:
            # Scalar
            atoms.append(f"{prefix}{data}" if prefix else str(data))

        return atoms

    def similarity(self, id1: str, id2: str) -> Optional[float]:
        """Compute cosine similarity between two stored vectors."""
        if id1 not in self.vectors or id2 not in self.vectors:
            return None

        v1 = self.vectors[id1].astype(np.float32)
        v2 = self.vectors[id2].astype(np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def clear(self):
        """Clear all points."""
        self.points.clear()
        self.vectors.clear()

    def stats(self) -> dict:
        """Get statistics about current state."""
        type_counts = {}
        for p in self.points.values():
            type_counts[p.point_type] = type_counts.get(p.point_type, 0) + 1

        return {
            "total_points": len(self.points),
            "by_type": type_counts,
            "dimensions": self.dimensions,
            "atoms_in_codebook": len(self.vector_manager.atom_vectors),
        }
