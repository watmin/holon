#!/usr/bin/env python3
"""
Metadata-Only Quote Storage System
Stores only vectors + coordinate metadata, never the actual text content.
"""

import json

# Configure logging
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataQuoteSystem:
    """Quote system that stores only vectors + coordinate metadata, no text content."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        # In a real system, this would be external metadata lookup
        self.metadata_lookup = {}  # vector_id -> coordinate_metadata

    def encode_quote_to_vector(self, quote_text: str) -> List[float]:
        """Encode quote text to vector (without storing text)."""
        # Normalize the quote text
        words = self._normalize_text(quote_text)

        # Create encoding data structure
        encode_data = {"text": {"_encode_mode": "ngram", "sequence": words}}

        # Get vector from client (simulating /encode API)
        return self.client.encode_vectors_json(encode_data)

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re

        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def store_quote_coordinates(
        self, quote_text: str, coordinates: Dict[str, Any]
    ) -> str:
        """
        Store a quote by encoding it to vector and storing only coordinates.

        Args:
            quote_text: The actual quote text (used for encoding, not stored)
            coordinates: Coordinate metadata to store with the vector

        Returns:
            vector_id: Unique ID for the stored vector
        """
        print(f"🔄 Encoding quote: '{quote_text[:50]}...'")

        # Encode quote to vector (text is not stored)
        vector = self.encode_quote_to_vector(quote_text)

        # Create storage unit with only vector + coordinates (no text)
        storage_unit = {
            "vector_data": vector,  # The encoded vector
            "coordinates": coordinates,  # Coordinate metadata
            "metadata": {
                "quote_type": "coordinate_reference",
                "word_count": len(self._normalize_text(quote_text)),
                "book_title": coordinates.get("book_title", "Unknown"),
            },
        }

        # Store in Holon (this will encode the vector_data again, but that's OK)
        # In a production system, you'd store pre-computed vectors directly
        vector_id = self.client.insert_json(storage_unit)

        # Store coordinate lookup (in real system, this would be external DB)
        self.metadata_lookup[vector_id] = coordinates

        print(
            f"✅ Stored quote coordinates at: {coordinates['chapter']} | Para {coordinates['paragraph_num']} | Page {coordinates['page_start']}"
        )
        return vector_id

    def search_quotes_by_content(
        self, query_text: str, top_k: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for quotes by content, return only coordinate metadata."""
        print(f"🔍 Searching for quote content: '{query_text}'")

        # Encode query to vector
        query_vector = self.encode_quote_to_vector(query_text)

        # Create probe with the query vector
        probe_data = {
            "vector_data": query_vector,
            "metadata": {"quote_type": "coordinate_reference"},
        }

        # Search for similar vectors
        results = self.client.search_json(
            probe_data,
            top_k=top_k,
            threshold=threshold,
        )

        print(f"📊 Found {len(results)} similar vectors")

        # Return only coordinate metadata (no text content)
        coordinate_results = []
        for result_data in results:
            vector_id = result_data["id"]
            score = result_data["score"]
            stored_data = result_data["data"]
            coordinates = self.metadata_lookup.get(
                vector_id, stored_data.get("coordinates", {})
            )

            result = {
                "vector_id": vector_id,
                "similarity_score": score,
                "coordinates": coordinates,
                "query_text": query_text,
            }
            coordinate_results.append(result)

        return coordinate_results

    def demonstrate_metadata_only_system(self):
        """Demonstrate the metadata-only quote storage system."""
        print("🏷️ Metadata-Only Quote Storage System")
        print("=" * 50)

        # Define coordinate metadata for our quotes
        quote_coordinates = [
            {
                "quote_text": "Everything depends upon relative minuteness.",
                "coordinates": {
                    "chapter": "Chapter II",
                    "paragraph_num": 1,
                    "page_start": 2,
                    "page_end": 2,
                    "word_count": 5,
                    "book_title": "Calculus Made Easy",
                    "quote_position": {"word_start": 0, "word_end": 5},
                },
            },
            {
                "quote_text": "Integration is the reverse of differentiation.",
                "coordinates": {
                    "chapter": "Chapter IV",
                    "paragraph_num": 1,
                    "page_start": 4,
                    "page_end": 4,
                    "word_count": 6,
                    "book_title": "Calculus Made Easy",
                    "quote_position": {"word_start": 0, "word_end": 6},
                },
            },
            {
                "quote_text": "dy dx is the slope of the tangent.",
                "coordinates": {
                    "chapter": "Chapter III",
                    "paragraph_num": 1,
                    "page_start": 3,
                    "page_end": 3,
                    "word_count": 7,
                    "book_title": "Calculus Made Easy",
                    "quote_position": {"word_start": 2, "word_end": 7},
                },
            },
            {
                "quote_text": "Some calculus-tricks are quite easy.",
                "coordinates": {
                    "chapter": "Conclusion",
                    "paragraph_num": 1,
                    "page_start": 5,
                    "page_end": 5,
                    "word_count": 5,
                    "book_title": "Calculus Made Easy",
                    "quote_position": {"word_start": 0, "word_end": 5},
                },
            },
        ]

        # Store quotes (vectors only + coordinates, no text)
        print("💾 Storing quotes as vectors + coordinate metadata...")
        stored_ids = []
        for quote_data in quote_coordinates:
            vector_id = self.store_quote_coordinates(
                quote_data["quote_text"], quote_data["coordinates"]
            )
            stored_ids.append(vector_id)

        print(f"✅ Stored {len(stored_ids)} quote vectors with coordinates")

        # Demonstrate searches that return only coordinates
        search_queries = [
            "Everything depends upon relative minuteness.",
            "integration is the reverse",
            "slope of the tangent",
            "calculus tricks",
        ]

        for query in search_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)

            results = self.search_quotes_by_content(query, threshold=0.0)

            if results:
                print("📍 Coordinate Results:")
                for i, result in enumerate(results[:3], 1):
                    coord = result["coordinates"]
                    print(
                        f"  {i}. 📍 {coord['chapter']} | Para {coord['paragraph_num']} | Page {coord['page_start']}"
                    )
                    print(f"      🎯 Similarity: {result.get('similarity', 0):.3f}")
                    print(f"      📖 Book: {coord['book_title']}")
                    if "quote_position" in coord:
                        pos = coord["quote_position"]
                        print(
                            f"      🎯 Position: words {pos['word_start']}-{pos['word_end']}"
                        )
                    print()
            else:
                print("❌ No matches found")

        print("\n✅ Key Achievement:")
        print("   • Only vectors + coordinate metadata stored in DB")
        print("   • No text content ever stored")
        print("   • Queries return precise coordinate locations")
        print("   • System provides exact quote positioning within documents")
        print("   • Metadata-only architecture achieved! 🌟")


def main():
    """Demonstrate metadata-only quote storage system."""
    system = MetadataQuoteSystem(dimensions=16000)
    system.demonstrate_metadata_only_system()


if __name__ == "__main__":
    main()
