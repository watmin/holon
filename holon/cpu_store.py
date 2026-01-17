import uuid
import logging
from typing import Dict, Any, List, Tuple
from .store import Store
from .atomizer import parse_data
from .vector_manager import VectorManager
from .encoder import Encoder
from .similarity import find_similar_vectors


class CPUStore(Store):
    def __init__(self, dimensions: int = 16000, backend: str = 'auto'):
        self.dimensions = dimensions

        # Auto-select backend
        if backend == 'auto':
            try:
                import cupy as cp
                cp.cuda.runtime.getDeviceCount()  # Check GPU availability
                self.backend = 'gpu'
                print("🎮 Auto-selected GPU backend")
            except (ImportError, cp.cuda.runtime.CUDARuntimeError):
                self.backend = 'cpu'
                print("💻 Auto-selected CPU backend")
        else:
            self.backend = backend

        self.vector_manager = VectorManager(dimensions, self.backend)
        self.encoder = Encoder(self.vector_manager)
        self.stored_data: Dict[str, Dict[str, Any]] = {}  # id -> original data dict
        self.stored_vectors: Dict[str, Any] = {}  # id -> encoded vector

    def insert(self, data: str, data_type: str = 'json') -> str:
        import time
        start = time.time()
        parsed = parse_data(data, data_type)
        parse_time = time.time() - start

        start = time.time()
        encoded_vector = self.encoder.encode_data(parsed)
        encode_time = time.time() - start

        data_id = str(uuid.uuid4())
        self.stored_data[data_id] = parsed
        self.stored_vectors[data_id] = encoded_vector

        # Log timing for first few inserts
        if len(self.stored_data) <= 5:
            logging.info(f"INSERT_TIMING: parse={parse_time:.4f}s, encode={encode_time:.4f}s, total={parse_time+encode_time:.4f}s")

        return data_id

    def query(self, probe: str, data_type: str = 'json', top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        parsed_probe = parse_data(probe, data_type)
        probe_vector = self.encoder.encode_data(parsed_probe)
        similar_ids_scores = find_similar_vectors(probe_vector, self.stored_vectors, top_k, threshold)
        results = []
        for data_id, score in similar_ids_scores:
            # Convert GPU vectors back to CPU for data access if needed
            data_dict = self.stored_data[data_id]
            results.append((data_id, score, data_dict))
        return results

    def get(self, data_id: str) -> Dict[str, Any]:
        if data_id not in self.stored_data:
            raise KeyError(f"Data ID {data_id} not found")
        return self.stored_data[data_id]

    def delete(self, data_id: str) -> bool:
        if data_id in self.stored_data:
            del self.stored_data[data_id]
            del self.stored_vectors[data_id]
            return True
        return False