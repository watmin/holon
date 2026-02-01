"""
Torchhd-backed encoder for Holon.

Uses torchhd as the VSA engine while providing Holon's structured data encoding API.
This gives us:
- PyTorch GPU acceleration (automatic)
- Level embeddings (numeric similarity)
- Circular embeddings (time/cyclical data)
- Built-in classifiers (AdaptHD, OnlineHD, etc.)
"""

import torch
import torchhd
from torchhd import embeddings
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


class TorchHDEncoder:
    """Structured data encoder using torchhd as the VSA backend."""
    
    def __init__(
        self,
        dimensions: int = 4096,
        device: str = "auto",
        vsa_model: str = "MAP",  # MAP, BSC, HRR, FHRR
    ):
        self.dimensions = dimensions
        self.vsa_model = vsa_model
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"TorchHD Encoder: {dimensions}D, {vsa_model}, device={self.device}")
        
        # Cache for field name hypervectors (role vectors)
        self._field_cache: Dict[str, torch.Tensor] = {}
        
        # Cache for categorical value hypervectors
        self._value_cache: Dict[str, torch.Tensor] = {}
        
        # Embeddings for numeric values (Level encoding = similarity between close values)
        self._level_embeddings: Dict[str, embeddings.Level] = {}
        
        # Embeddings for circular/time values
        self._circular_embedding: Optional[embeddings.Circular] = None
        
    def _get_field_hv(self, field_name: str) -> torch.Tensor:
        """Get or create hypervector for a field name (role vector).
        
        Uses sparse vectors (~1/3 zeros) to match original encoder behavior.
        """
        if field_name not in self._field_cache:
            # Deterministic seeding based on field name
            seed = int(hashlib.md5(field_name.encode()).hexdigest()[:8], 16)
            self._field_cache[field_name] = self._sparse_random(seed)
        return self._field_cache[field_name]
    
    def _get_value_hv(self, value: str) -> torch.Tensor:
        """Get or create hypervector for a categorical value.
        
        Uses sparse vectors (~1/3 zeros) to match original encoder behavior.
        """
        cache_key = str(value)
        if cache_key not in self._value_cache:
            seed = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
            self._value_cache[cache_key] = self._sparse_random(seed)
        return self._value_cache[cache_key]
    
    def _sparse_random(self, seed: int) -> torch.Tensor:
        """Generate a sparse random bipolar vector with ~1/3 zeros.
        
        This matches the original encoder's vector distribution.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # Generate values in {-1, 0, 1} with roughly equal probability
        # Use uniform random and threshold
        rand = torch.rand(self.dimensions, device=self.device, generator=generator)
        vec = torch.zeros(self.dimensions, device=self.device, dtype=torch.float32)
        vec[rand < 0.33] = -1.0
        vec[rand > 0.66] = 1.0
        # Middle 1/3 stays at 0
        return vec
    
    def _get_level_embedding(self, field_name: str, low: float, high: float, levels: int = 100) -> embeddings.Level:
        """Get or create Level embedding for numeric field."""
        cache_key = f"{field_name}:{low}:{high}:{levels}"
        if cache_key not in self._level_embeddings:
            self._level_embeddings[cache_key] = embeddings.Level(
                levels, self.dimensions, low=low, high=high, device=self.device
            )
        return self._level_embeddings[cache_key]
    
    def _encode_value(self, value: Any, field_name: str = "") -> torch.Tensor:
        """Encode a single value to a hypervector."""
        if value is None:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Handle EDN Keyword/Symbol
        if hasattr(value, 'name'):  # EDN Keyword or Symbol
            return self._get_value_hv(value.name)
        
        # Handle $time marker (circular encoding)
        if isinstance(value, dict) and "$time" in value:
            return self._encode_time(value["$time"])
        
        # Handle $any marker (wildcard)
        if isinstance(value, dict) and "$any" in value:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Handle nested dict (recursive)
        if isinstance(value, dict):
            return self._encode_dict(value)
        
        # Handle list
        if isinstance(value, list):
            return self._encode_list(value)
        
        # Handle set/frozenset
        if isinstance(value, (set, frozenset)):
            return self._encode_list(list(value))
        
        # Handle numeric (Level encoding - close values have similar vectors!)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return self._encode_numeric(value, field_name)
        
        # Handle boolean
        if isinstance(value, bool):
            return self._get_value_hv(f"bool:{value}")
        
        # Handle string (categorical)
        return self._get_value_hv(str(value))
    
    def _encode_numeric(self, value: float, field_name: str = "") -> torch.Tensor:
        """Encode numeric value using Level embedding (similar values = similar vectors)."""
        # Adaptive range based on field name hints
        if "status" in field_name.lower() or "code" in field_name.lower():
            # HTTP status codes
            level_emb = self._get_level_embedding(field_name, low=100, high=600, levels=50)
        elif "duration" in field_name.lower() or "time" in field_name.lower() or "ms" in field_name.lower():
            # Durations in ms
            level_emb = self._get_level_embedding(field_name, low=0, high=10000, levels=100)
        elif "score" in field_name.lower() or "percent" in field_name.lower():
            # Percentages/scores
            level_emb = self._get_level_embedding(field_name, low=0, high=100, levels=100)
        elif "count" in field_name.lower() or "num" in field_name.lower():
            # Counts
            level_emb = self._get_level_embedding(field_name, low=0, high=1000, levels=100)
        else:
            # General numeric
            level_emb = self._get_level_embedding(field_name or "default", low=-100, high=100, levels=100)
        
        return level_emb(torch.tensor([value], device=self.device)).squeeze(0)
    
    def _encode_time(self, timestamp) -> torch.Tensor:
        """Encode timestamp using Circular + Level (positional) embedding."""
        import datetime as dt_module
        
        # Handle ISO string timestamps
        if isinstance(timestamp, str):
            try:
                # Try ISO format parsing
                if 'T' in timestamp:
                    dt = dt_module.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = dt_module.datetime.fromisoformat(timestamp)
                timestamp = dt.timestamp()
            except ValueError:
                # Fallback to hash-based encoding for invalid strings
                return self._get_value_hv(f"time:{timestamp}")
        
        if self._circular_embedding is None:
            # 24 phases for hours in a day
            self._circular_embedding = embeddings.Circular(24, self.dimensions, device=self.device)
        
        # Extract datetime components
        dt = dt_module.datetime.fromtimestamp(timestamp)
        hour = dt.hour + dt.minute / 60.0
        
        # Circular components (periodic patterns)
        hour_hv = self._circular_embedding(torch.tensor([hour], device=self.device)).squeeze(0)
        dow_emb = self._get_value_hv(f"dow:{dt.weekday()}")
        month_emb = self._get_value_hv(f"month:{dt.month}")
        
        # Positional component using Level embedding (near times = similar vectors)
        # Range: 10 years of hours, centered around 2024
        if not hasattr(self, '_time_level_embedding'):
            # 10 years = ~87600 hours, 1000 levels for granularity
            base_2024 = dt_module.datetime(2024, 1, 1).timestamp()
            self._time_level_embedding = embeddings.Level(
                1000, self.dimensions,
                low=(base_2024 - 5*365*24*3600) / 3600,  # 5 years before 2024
                high=(base_2024 + 5*365*24*3600) / 3600,  # 5 years after 2024
                device=self.device
            )
        
        hours_since_epoch = timestamp / 3600
        position_hv = self._time_level_embedding(
            torch.tensor([hours_since_epoch], device=self.device)
        ).squeeze(0)
        
        # Bundle: balance circular and positional
        # Hour: same-hour-diff-day similarity (weight 2)
        # Position: near-time-more-similar (weight 1.5)
        return 2.0 * hour_hv + 0.3 * dow_emb + 0.3 * month_emb + 1.5 * position_hv
    
    def _encode_list(self, items: List[Any]) -> torch.Tensor:
        """Encode a list by bundling items with position binding."""
        if not items:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Encode each item bound with its position
        result = torch.zeros(self.dimensions, device=self.device)
        for i, item in enumerate(items):
            pos_hv = self._get_value_hv(f"pos:{i}")
            item_hv = self._encode_value(item)
            result = result + torchhd.bind(pos_hv, item_hv)
        
        return result
    
    def _encode_dict(self, data: Dict[str, Any]) -> torch.Tensor:
        """Encode a dictionary using hash_table structure (field * value bindings bundled)."""
        if not data:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Handle special case: dict IS a time marker
        if "$time" in data and len(data) == 1:
            return self._encode_time(data["$time"])
        if "$time" in data:
            # Has $time plus other fields - encode time specially
            pass  # Fall through to normal encoding
        
        # Handle $any marker (wildcard)
        if "$any" in data:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Build key-value pairs
        keys = []
        values = []
        
        for field, value in data.items():
            # Convert EDN Keyword/Symbol to string
            field_str = str(field)
            if hasattr(field, 'name'):  # EDN Keyword
                field_str = field.name
            
            # Skip internal Holon metadata fields (stored with data)
            # but DO encode user fields that start with "_" (like _type, _in_class)
            if field_str in ("_raw", "_parsed", "_encode_mode"):
                continue
            
            field_hv = self._get_field_hv(field_str)
            value_hv = self._encode_value(value, field_name=field_str)
            
            keys.append(field_hv)
            values.append(value_hv)
        
        if not keys:
            return torch.zeros(self.dimensions, device=self.device)
        
        # Use torchhd's hash_table for structured encoding
        keys_tensor = torch.stack(keys)
        values_tensor = torch.stack(values)
        
        return torchhd.hash_table(keys_tensor, values_tensor)
    
    def encode_data(self, data: Union[str, Dict]) -> torch.Tensor:
        """Encode structured data (JSON string or dict) to hypervector."""
        if isinstance(data, str):
            data = json.loads(data)
        
        result = self._encode_dict(data)
        # Threshold to bipolar for compatibility with original encoder
        return self._threshold_bipolar(result)
    
    def _threshold_bipolar(self, vec: torch.Tensor) -> torch.Tensor:
        """Threshold vector to bipolar values [-1, 0, 1].
        
        Uses a small threshold around 0 to create zeros, matching
        the original encoder's behavior.
        """
        result = torch.zeros_like(vec, dtype=torch.int8)
        # Use threshold of 0.5 to create zeros (values between -0.5 and 0.5 become 0)
        result[vec > 0.5] = 1
        result[vec < -0.5] = -1
        return result
    
    # VSA Primitives (delegated to torchhd)
    
    def bind(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bind two vectors (AND-like operation)."""
        return torchhd.bind(vec1, vec2)
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Bundle multiple vectors (OR-like operation)."""
        if not vectors:
            return torch.zeros(self.dimensions, device=self.device)
        # Sum all vectors (equivalent to bundling in MAP model)
        result = vectors[0]
        for v in vectors[1:]:
            result = result + v
        return result
    
    def similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity between vectors."""
        return torchhd.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def prototype(self, vectors: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        """Create prototype from list of vectors.
        
        Bundles all vectors and thresholds to bipolar for consistent similarity.
        """
        if not vectors:
            return torch.zeros(self.dimensions, device=self.device, dtype=torch.int8)
        # Sum all vectors to create prototype
        result = vectors[0].float()
        for v in vectors[1:]:
            result = result + v.float()
        # Threshold to bipolar (like original encoder)
        return self._threshold_bipolar(result)
    
    def difference(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Compute what changed between two states."""
        return after - before
    
    def amplify(self, superposition: torch.Tensor, component: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Amplify a component in a superposition."""
        return superposition + strength * component
    
    def negate(self, superposition: torch.Tensor, component: torch.Tensor, method: str = "subtract") -> torch.Tensor:
        """Remove a component from a superposition.
        
        Args:
            superposition: The base vector
            component: The component to remove
            method: "subtract" (default) or "orthogonalize"
        """
        if method == "orthogonalize":
            # Project component out of superposition
            dot = torch.dot(superposition.float(), component.float())
            norm_sq = torch.dot(component.float(), component.float())
            if norm_sq > 0:
                projection = (dot / norm_sq) * component
                return superposition - projection
            return superposition
        else:  # subtract
            return superposition - component
    
    def blend(self, vec1: torch.Tensor, vec2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Blend two vectors with given weight."""
        return alpha * vec1 + (1 - alpha) * vec2
    
    def resonance(self, vec: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Extract the part of vec that resonates with reference."""
        # Where they agree (same sign), keep the value
        agree = (vec * reference) > 0
        result = torch.zeros_like(vec)
        result[agree] = vec[agree]
        return result
    
    def to_numpy(self, tensor: torch.Tensor):
        """Convert tensor to numpy for compatibility."""
        return tensor.cpu().numpy()
    
    # Mathematical Primitives (for compatibility with original encoder)
    
    def encode_mathematical_primitive(self, primitive, value: float) -> torch.Tensor:
        """Encode fundamental mathematical properties."""
        from .encoder import MathematicalPrimitive
        
        if primitive == MathematicalPrimitive.CONVERGENCE_RATE:
            return self._encode_convergence_rate(value)
        elif primitive == MathematicalPrimitive.ITERATION_COMPLEXITY:
            return self._encode_iteration_complexity(value)
        elif primitive == MathematicalPrimitive.FREQUENCY_DOMAIN:
            return self._encode_frequency_domain(value)
        elif primitive == MathematicalPrimitive.AMPLITUDE_SCALE:
            return self._encode_amplitude_scale(value)
        elif primitive == MathematicalPrimitive.POWER_LAW_EXPONENT:
            return self._encode_power_law_exponent(value)
        elif primitive == MathematicalPrimitive.CLUSTERING_COEFFICIENT:
            return self._encode_clustering_coefficient(value)
        elif primitive == MathematicalPrimitive.TOPOLOGICAL_DISTANCE:
            return self._encode_topological_distance(value)
        elif primitive == MathematicalPrimitive.SELF_SIMILARITY:
            return self._encode_self_similarity(value)
        else:
            raise ValueError(f"Unknown mathematical primitive: {primitive}")
    
    def _encode_convergence_rate(self, rate: float) -> torch.Tensor:
        """Encode mathematical convergence properties."""
        if rate < 0.2:
            category = "very_slow_convergence"
        elif rate < 0.4:
            category = "slow_convergence"
        elif rate < 0.6:
            category = "moderate_slow_convergence"
        elif rate < 0.8:
            category = "moderate_convergence"
        elif rate < 0.9:
            category = "fast_convergence"
        elif rate < 0.95:
            category = "very_fast_convergence"
        else:
            category = "divergent"
        return self._get_value_hv(category)
    
    def _encode_iteration_complexity(self, iterations: int) -> torch.Tensor:
        """Encode computational iteration complexity."""
        if iterations < 10:
            category = "low_complexity"
        elif iterations < 50:
            category = "moderate_complexity"
        elif iterations < 200:
            category = "high_complexity"
        else:
            category = "extreme_complexity"
        return self._get_value_hv(category)
    
    def _encode_frequency_domain(self, freq: float) -> torch.Tensor:
        """Encode frequency domain properties."""
        if freq < 0.01:
            category = "very_low_frequency"
        elif freq < 0.1:
            category = "low_frequency"
        elif freq < 1.0:
            category = "medium_low_frequency"
        elif freq < 10.0:
            category = "medium_frequency"
        elif freq < 100.0:
            category = "high_frequency"
        else:
            category = "ultrasonic_frequency"
        return self._get_value_hv(category)
    
    def _encode_amplitude_scale(self, amp: float) -> torch.Tensor:
        """Encode amplitude scale properties."""
        if amp < 0.01:
            category = "micro_amplitude"
        elif amp < 0.1:
            category = "small_amplitude"
        elif amp < 1.0:
            category = "medium_amplitude"
        elif amp < 10.0:
            category = "large_amplitude"
        else:
            category = "macro_amplitude"
        return self._get_value_hv(category)
    
    def _encode_power_law_exponent(self, exponent: float) -> torch.Tensor:
        """Encode power law scaling exponent."""
        if exponent < -2:
            category = "steep_decay"
        elif exponent < -1:
            category = "moderate_decay"
        elif exponent < 0:
            category = "slow_decay"
        elif exponent < 1:
            category = "slow_growth"
        elif exponent < 2:
            category = "moderate_growth"
        else:
            category = "steep_growth"
        return self._get_value_hv(category)
    
    def _encode_clustering_coefficient(self, coeff: float) -> torch.Tensor:
        """Encode graph clustering coefficient."""
        if coeff < 0.1:
            category = "sparse_clustering"
        elif coeff < 0.3:
            category = "low_clustering"
        elif coeff < 0.5:
            category = "moderate_clustering"
        elif coeff < 0.7:
            category = "high_clustering"
        else:
            category = "dense_clustering"
        return self._get_value_hv(category)
    
    def _encode_topological_distance(self, distance: float) -> torch.Tensor:
        """Encode topological distance."""
        if distance < 1:
            category = "adjacent"
        elif distance < 3:
            category = "nearby"
        elif distance < 6:
            category = "moderate_distance"
        else:
            category = "far"
        return self._get_value_hv(category)
    
    def _encode_self_similarity(self, measure: float) -> torch.Tensor:
        """Encode self-similarity (fractal dimension)."""
        if measure < 1.2:
            category = "low_complexity_fractal"
        elif measure < 1.5:
            category = "moderate_fractal"
        elif measure < 1.8:
            category = "complex_fractal"
        else:
            category = "highly_complex_fractal"
        return self._get_value_hv(category)
    
    def mathematical_bind(self, *vectors) -> torch.Tensor:
        """Bind mathematical properties together."""
        if not vectors:
            return torch.zeros(self.dimensions, device=self.device)
        result = vectors[0]
        for vec in vectors[1:]:
            result = torchhd.bind(result, vec)
        return result
    
    def mathematical_bundle(self, vectors: List[torch.Tensor], weights: List[float] = None) -> torch.Tensor:
        """Bundle mathematical properties with optional weighting."""
        if not vectors:
            return torch.zeros(self.dimensions, device=self.device)
        if weights is None:
            weights = [1.0] * len(vectors)
        result = torch.zeros(self.dimensions, device=self.device, dtype=torch.float32)
        for vec, weight in zip(vectors, weights):
            result = result + weight * vec.float()
        return result


class TorchHDStore:
    """Simple in-memory store using TorchHD encoder."""
    
    def __init__(self, dimensions: int = 4096, device: str = "auto"):
        self.encoder = TorchHDEncoder(dimensions=dimensions, device=device)
        self.vectors: List[torch.Tensor] = []
        self.data: List[Dict] = []
        self.ids: List[str] = []
    
    def insert(self, data: Union[str, Dict], format: str = "json", id: Optional[str] = None) -> str:
        """Insert data and return ID."""
        if isinstance(data, str):
            data_dict = json.loads(data)
        else:
            data_dict = data
        
        vec = self.encoder.encode_data(data_dict)
        
        if id is None:
            id = f"item_{len(self.ids)}"
        
        self.vectors.append(vec)
        self.data.append(data_dict)
        self.ids.append(id)
        
        return id
    
    def query(self, probe: Union[str, Dict], top_k: int = 5) -> List[tuple]:
        """Query for similar items."""
        if isinstance(probe, str):
            probe = json.loads(probe)
        
        probe_vec = self.encoder.encode_data(probe)
        
        # Compute similarities
        if not self.vectors:
            return []
        
        all_vecs = torch.stack(self.vectors)
        sims = torchhd.cosine_similarity(probe_vec.unsqueeze(0), all_vecs).squeeze(0)
        
        # Get top-k
        top_indices = torch.argsort(sims, descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            idx = idx.item()
            results.append((self.ids[idx], sims[idx].item(), self.data[idx]))
        
        return results
    
    def prototype(self, vectors: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        """Create prototype from vectors."""
        return self.encoder.prototype(vectors, threshold)
    
    def difference(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Compute difference between states."""
        return self.encoder.difference(before, after)
    
    def amplify(self, superposition: torch.Tensor, component: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Amplify component in superposition."""
        return self.encoder.amplify(superposition, component, strength)


# Quick test
if __name__ == "__main__":
    print("Testing TorchHD Encoder...")
    
    store = TorchHDStore(dimensions=4096)
    
    # Test structured data encoding
    store.insert({"name": "Alice", "role": "admin", "score": 95})
    store.insert({"name": "Bob", "role": "user", "score": 85})
    store.insert({"name": "Charlie", "role": "admin", "score": 90})
    
    # Query
    results = store.query({"role": "admin"}, top_k=3)
    print("\nQuery: role=admin")
    for id, sim, data in results:
        print(f"  {id}: {sim:.3f} - {data}")
    
    # Test numeric similarity (Level encoding)
    print("\nNumeric similarity test (score field):")
    enc = store.encoder
    v90 = enc._encode_numeric(90, "score")
    v91 = enc._encode_numeric(91, "score")
    v50 = enc._encode_numeric(50, "score")
    
    print(f"  sim(90, 91) = {enc.similarity(v90, v91):.3f} (should be high)")
    print(f"  sim(90, 50) = {enc.similarity(v90, v50):.3f} (should be lower)")
    
    print("\n✅ TorchHD encoder working!")
