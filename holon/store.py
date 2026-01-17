from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class Store(ABC):
    @abstractmethod
    def insert(self, data: str, data_type: str = 'json') -> str:
        """
        Insert a data blob (JSON or EDN string) into the store.
        Returns a unique ID for the inserted data.

        :param data: The data blob as a string.
        :param data_type: 'json' or 'edn'.
        :return: Unique identifier for the data.
        """
        pass

    @abstractmethod
    def query(self, probe: str, data_type: str = 'json', top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query the store with a probe data blob.
        Returns a list of (id, similarity_score, original_data) tuples for top matches.

        :param probe: The query probe as a string.
        :param data_type: 'json' or 'edn'.
        :param top_k: Number of top results to return.
        :param threshold: Minimum similarity score to include in results.
        :return: List of tuples (data_id, score, data_dict).
        """
        pass

    @abstractmethod
    def get(self, data_id: str) -> Dict[str, Any]:
        """
        Retrieve original data by ID.

        :param data_id: Unique identifier.
        :return: Original data as a dictionary.
        """
        pass

    @abstractmethod
    def delete(self, data_id: str) -> bool:
        """
        Delete data by ID.

        :param data_id: Unique identifier.
        :return: True if deleted, False otherwise.
        """
        pass