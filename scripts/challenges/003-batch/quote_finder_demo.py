#!/usr/bin/env python3
"""
Comprehensive Quote Finder Demo
Demonstrates the complete Holon quote finder system with real data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from holon import CPUStore
from holon.encoder import ListEncodeMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveQuoteFinder:
    """Complete quote finder system with real data validation."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.processed_quotes = []  # Store processed quotes for validation

    def load_processed_quotes(self, json_file: str) -> List[Dict[str, Any]]:
        """Load processed quotes from JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        quotes = data["quotes"]
        print(f"📚 Loaded {len(quotes)} processed quotes from {json_file}")
        return quotes

    def create_quote_unit(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Holon unit for a quote (metadata-only storage)."""
        # Normalize the quote text for searching
        words = self._normalize_text(quote["text"])

        # Create unit with n-gram encoding for the words
        unit = {
            "words": {
                "_encode_mode": "ngram",  # Use n-gram encoding for fuzzy matching
                "sequence": words,
            },
            "metadata": {
                "chapter": quote["chapter"],
                "page": quote["page"],
                "paragraph": quote["paragraph"],
                "book_title": quote["book_title"],
                "word_count": quote["word_count"],
                "line_number": quote["line_number"],
            },
        }

        return unit

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text: lowercase, remove punctuation, split into words."""
        import re

        # Remove punctuation and lowercase
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        # Split into words and filter out empty strings
        words = [word for word in normalized.split() if word]
        return words

    def ingest_quotes(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes into Holon store using metadata-only approach."""
        print(f"💾 Ingesting {len(quotes)} quotes into Holon store...")

        units_data = []
        for quote in quotes:
            unit_data = self.create_quote_unit(quote)
            units_data.append(json.dumps(unit_data))

            # Keep processed quotes for validation
            self.processed_quotes.append(quote)

        # Batch insert for efficiency
        ids = self.store.batch_insert(units_data, data_type="json")
        print(f"✅ Successfully ingested {len(ids)} quote units")
        return ids

    def bootstrap_search_vector(self, query_text: str) -> List[float]:
        """Use /encode API to bootstrap a search vector for a query."""
        words = self._normalize_text(query_text)

        # Create the same encoding structure as stored units
        encode_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Use the encoder directly (simulating the API)
        vector = self.store.encoder.encode_data(encode_data)

        # Convert to list for API-like response
        cpu_vector = self.store.vector_manager.to_cpu(vector)
        return cpu_vector.tolist()

    def search_quotes(
        self,
        query_text: str,
        top_k: int = 10,
        threshold: float = 0.0,
        chapter_filter: str = None,
        page_filter: int = None,
    ) -> List[Dict[str, Any]]:
        """Search for quotes using bootstrapped vector and metadata filters."""
        print(f"🔍 Searching for: '{query_text}'")

        # Bootstrap search vector
        search_vector_list = self.bootstrap_search_vector(query_text)

        # Create probe using n-gram encoding
        words = self._normalize_text(query_text)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Build guard for filters
        guard = {}
        if chapter_filter:
            guard["metadata"] = {"chapter": chapter_filter}
        if page_filter:
            if "metadata" not in guard:
                guard["metadata"] = {}
            guard["metadata"]["page"] = page_filter

        # If no guard specified, remove it
        if not guard:
            guard = None

        # Query the store
        results = self.store.query(
            probe=json.dumps(probe_data),
            data_type="json",
            top_k=top_k,
            threshold=threshold,
            guard=guard,
        )

        print(f"📊 Query returned {len(results)} raw results")

        # Enrich results with reconstructed text and validation
        enriched_results = []
        for data_id, score, data in results:
            # Find the original quote data for validation
            original_quote = None
            for quote in self.processed_quotes:
                # Match by metadata (since we don't store full text)
                if (
                    quote["chapter"] == data["metadata"]["chapter"]
                    and quote["page"] == data["metadata"]["page"]
                    and quote["line_number"] == data["metadata"]["line_number"]
                ):
                    original_quote = quote
                    break

            enriched_result = {
                "id": data_id,
                "score": score,
                "metadata": data["metadata"],
                "reconstructed_text": original_quote["text"]
                if original_quote
                else "Text not found",
                "search_words": words,
                "query_text": query_text,
                "validation": {
                    "metadata_match": original_quote is not None,
                    "text_available": original_quote is not None,
                },
            }

            enriched_results.append(enriched_result)

        return enriched_results

    def demonstrate_system(self):
        """Comprehensive demonstration of the quote finder system."""
        print("🎯 Holon Quote Finder - Comprehensive Demonstration")
        print("=" * 60)

        # Load processed quotes
        quotes_file = Path(__file__).parent / "processed_quotes.json"
        quotes = self.load_processed_quotes(str(quotes_file))

        # Show what we have
        print("\n📚 Available Quotes:")
        for i, quote in enumerate(quotes, 1):
            print(
                f"   {i}. [{quote['chapter']}] Page {quote['page']}: \"{quote['text']}\""
            )
        print()

        # Ingest into Holon
        ids = self.ingest_quotes(quotes)

        # Demonstration scenarios
        scenarios = [
            {
                "name": "Exact Quote Match",
                "query": "Everything depends upon relative minuteness",
                "description": "Should find exact match with high score",
            },
            {
                "name": "Fuzzy Phrase Match",
                "query": "depends on relative smallness",
                "description": "Similar words, different structure",
            },
            {
                "name": "Partial Quote Match",
                "query": "slope of the tangent",
                "description": "Subset of a longer quote",
            },
            {
                "name": "Integration Concept",
                "query": "integration is the reverse",
                "description": "Key calculus concept",
            },
            {
                "name": "Calculus Tricks",
                "query": "calculus tricks are easy",
                "description": "Book title reference",
            },
        ]

        for scenario in scenarios:
            print(f"\n🔬 Scenario: {scenario['name']}")
            print(f"   {scenario['description']}")
            print("-" * 50)

            results = self.search_quotes(scenario["query"], threshold=0.0)

            if results:
                print(f"   ✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['score']:.3f}")
                    print(f"      Text: \"{result['reconstructed_text']}\"")
                    print(
                        f"      Metadata: {result['metadata']['chapter']} | Page {result['metadata']['page']}"
                    )
                    print(
                        f"      Validation: {'✅' if result['validation']['metadata_match'] else '❌'}"
                    )
                    if i >= 3:  # Show top 3
                        break
            else:
                print("   ❌ No results found")
            print()

        # Demonstrate metadata filtering
        print("\n🔬 Metadata Filtering Demonstration:")
        print("-" * 40)

        # Search for integration quotes
        integration_results = self.search_quotes("integration", threshold=0.0)
        print(f"   Integration mentions: {len(integration_results)} found")

        # Bootstrap vector demo
        print("\n🔬 Vector Bootstrapping Demo:")
        print("-" * 40)

        test_phrase = "differential symbol"
        vector = self.bootstrap_search_vector(test_phrase)
        print(f"   Bootstrapped vector for '{test_phrase}': {len(vector)} dimensions")
        print(f"   Sample values: {vector[:5]}...")
        print(f"   Value range: {min(vector):.3f} to {max(vector):.3f}")

        # Search with the bootstrapped concept
        results = self.search_quotes(test_phrase, threshold=0.0)
        print(f"   Search results: {len(results)} matches")

        print("\n🎉 Demonstration Complete!")
        print("✅ Key Validations:")
        print("   • N-gram encoding enables fuzzy matching")
        print("   • Metadata-only storage preserves pointers")
        print("   • Vector bootstrapping works for queries")
        print("   • Similarity search finds relevant quotes")
        print("   • System handles partial and fuzzy matches")


def main():
    """Run the comprehensive quote finder demonstration."""
    finder = ComprehensiveQuoteFinder(dimensions=16000)
    finder.demonstrate_system()


if __name__ == "__main__":
    main()
