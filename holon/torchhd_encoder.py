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
        """Get or create hypervector for a field name (role vector)."""
        if field_name not in self._field_cache:
            # Deterministic seeding based on field name
            seed = int(hashlib.md5(field_name.encode()).hexdigest()[:8], 16)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self._field_cache[field_name] = torchhd.random(
                1, self.dimensions, device=self.device, generator=generator
            ).squeeze(0)
        return self._field_cache[field_name]
    
    def _get_value_hv(self, value: str) -> torch.Tensor:
        """Get or create hypervector for a categorical value."""
        cache_key = str(value)
        if cache_key not in self._value_cache:
            seed = int(hashlib.md5(cache_key.encode()).hexdigest()[:8], 16)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self._value_cache[cache_key] = torchhd.random(
                1, self.dimensions, device=self.device, generator=generator
            ).squeeze(0)
        return self._value_cache[cache_key]
    
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
        """Encode timestamp using Circular embedding (wraps around)."""
        import datetime
        
        # Handle ISO string timestamps
        if isinstance(timestamp, str):
            try:
                # Try ISO format parsing
                if 'T' in timestamp:
                    dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.datetime.fromisoformat(timestamp)
                timestamp = dt.timestamp()
            except ValueError:
                # Fallback to hash-based encoding for invalid strings
                return self._get_value_hv(f"time:{timestamp}")
        
        if self._circular_embedding is None:
            # 24 phases for hours in a day
            self._circular_embedding = embeddings.Circular(24, self.dimensions, device=self.device)
        
        # Extract hour of day for circular encoding
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour = dt.hour + dt.minute / 60.0
        
        # Also encode day of week, month for richer time representation
        dow_emb = self._get_value_hv(f"dow:{dt.weekday()}")
        month_emb = self._get_value_hv(f"month:{dt.month}")
        hour_hv = self._circular_embedding(torch.tensor([hour], device=self.device)).squeeze(0)
        
        # Bundle time components (torchhd.bundle takes 2 args, so chain or sum)
        return hour_hv + dow_emb + month_emb
    
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
        
        # Build key-value pairs
        keys = []
        values = []
        
        for field, value in data.items():
            # Convert EDN Keyword/Symbol to string
            field_str = str(field)
            if hasattr(field, 'name'):  # EDN Keyword
                field_str = field.name
            
            if field_str.startswith("_"):  # Skip metadata fields
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
        
        return self._encode_dict(data)
    
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
        """Create prototype from list of vectors."""
        if not vectors:
            return torch.zeros(self.dimensions, device=self.device)
        # Sum all vectors to create prototype
        result = vectors[0].clone()
        for v in vectors[1:]:
            result = result + v
        return result
    
    def difference(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Compute what changed between two states."""
        return after - before
    
    def amplify(self, superposition: torch.Tensor, component: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Amplify a component in a superposition."""
        return superposition + strength * component
    
    def negate(self, superposition: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
        """Remove a component from a superposition."""
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
