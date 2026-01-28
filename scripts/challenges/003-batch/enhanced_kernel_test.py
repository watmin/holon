#!/usr/bin/env python3
"""
Enhanced Kernel Primitives Test - JSON interface only.
Test holon's enhanced geometric capabilities through clean JSON API.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

from holon import CPUStore, HolonClient


class EnhancedKernelQuoteFinder:
    """Quote finder using enhanced kernel primitives via JSON interface."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.quotes_data = []
        self.id_to_quote = {}

    def normalize_words(self, text: str) -> List[str]:
        """Normalize text."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def ingest_quotes_enhanced(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes using enhanced kernel primitives via JSON."""
        print(f"📝 Ingesting {len(quotes)} quotes with enhanced kernel primitives...")

        units_data = []
        for quote in quotes:
            # Enhanced N-gram with kernel-level primitives
            # User specifies config via JSON - kernel handles the complexity
            unit_data = {
                "text": {
                    "_encode_mode": "ngram",  # Use standard NGRAM mode
                    "_encode_config": {
                        "n_sizes": [1, 2],        # Individual + bigrams
                        "weights": [0.3, 0.7],  # Weight bigrams higher
                        "length_penalty": True, # Normalize for query length
                        "idf_weighting": False  # Could enable with corpus stats
                    },
                    "sequence": self.normalize_words(quote["text"])
                }
            }
            units_data.append(unit_data)
            self.quotes_data.append(quote)

        ids = self.client.insert_batch_json(units_data)

        for vector_id, quote in zip(ids, quotes):
            self.id_to_quote[vector_id] = quote

        print(f"✅ Enhanced kernel ingestion complete: {len(ids)} quotes")
        return ids

    def search_enhanced_kernel(self, query_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using enhanced kernel primitives via clean JSON interface.
        """
        words = self.normalize_words(query_phrase)

        # Enhanced query with same kernel primitives
        probe_data = {
            "text": {
                "_encode_mode": "ngram",
                "_encode_config": {
                    "n_sizes": [1, 2],
                    "weights": [0.3, 0.7],
                    "length_penalty": True,  # Critical for substring matching
                    "idf_weighting": False
                },
                "sequence": words
            }
        }

        print(f"🔍 Enhanced kernel search for: '{query_phrase}' → {words}")

        # Pure geometric search through JSON API
        results = self.client.search_json(
            probe=probe_data,
            top_k=top_k,
            threshold=0.0
        )

        # Convert to our format
        formatted_results = []
        for result in results:
            data_id = result["id"]
            vsa_score = result["score"]
            original_quote = self.id_to_quote.get(data_id)

            if original_quote:
                result_data = {
                    "id": data_id,
                    "vsa_score": vsa_score,
                    "fuzzy_score": 0.0,
                    "combined_score": vsa_score,
                    "search_method": "enhanced_kernel",
                    "metadata": original_quote,
                    "reconstructed_text": original_quote["text"],
                    "search_words": words,
                }
                formatted_results.append(result_data)

        formatted_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return formatted_results[:top_k]


def test_kernel_enhancements():
    """Test enhanced kernel primitives through JSON interface."""
    print("🧠 Testing Enhanced Kernel Primitives (JSON Interface)")
    print("=" * 60)

    finder = EnhancedKernelQuoteFinder()

    # Load quotes
    quotes_file = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "quotes.txt"

    with open(quotes_file, "r") as f:
        quotes_text = f.read()

    quotes = []
    lines = quotes_text.strip().split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if '"' in line:
            quote_match = re.search(r'"([^"]*)"', line)
            if quote_match:
                quote_text = quote_match.group(1)
                quotes.append({
                    "text": quote_text,
                    "chapter": "Unknown",
                    "page": (i + 1) * 3,
                    "paragraph": 1,
                    "book_title": "Calculus Made Easy",
                })

    # Ingest with enhanced kernel primitives (JSON config)
    finder.ingest_quotes_enhanced(quotes)

    # Test queries
    test_queries = [
        ("Everything depends upon relative minuteness", "exact match"),
        ("depends upon relative", "substring match"),
        ("calculus", "single word - should improve with length penalty"),
        ("quantum physics", "negative control"),
        ("integration is reverse", "multi-word substring"),
    ]

    print("\n🔍 Enhanced Kernel Search Results (JSON Interface):")
    print("-" * 50)

    for query, desc in test_queries:
        print(f"\n📝 Query: '{query}' ({desc})")
        results = finder.search_enhanced_kernel(query, top_k=3)

        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results):
                score = result["vsa_score"]
                quote_text = result["reconstructed_text"][:60]
                print(".3f")
        else:
            print("   ❌ No results found")


def demonstrate_kernel_primitives():
    """Demonstrate the enhanced kernel primitives available via JSON."""
    print("\n🔧 Enhanced Kernel Primitives (Userland JSON Interface)")
    print("=" * 60)

    print("JSON Interface - Clean separation of concerns:")
    print("""
    {
        "text": {
            "_encode_mode": "ngram",
            "_encode_config": {
                "n_sizes": [1, 2, 3],        // Individual + bigrams + trigrams
                "weights": [0.2, 0.6, 0.4], // Relative importance
                "length_penalty": true,     // Normalize query length
                "idf_weighting": false      // TF-IDF style weighting
            },
            "sequence": ["word1", "word2", "word3"]
        }
    }
    """)

    print("Available Enhanced Primitives:")
    print("  ✅ Configurable pattern sizes (individuals vs compositions)")
    print("  ✅ Weighted component combination")
    print("  ✅ Length normalization for fair matching")
    print("  ✅ Extensible for future enhancements")
    print()

    print("Kernel Benefits:")
    print("  ✅ Pure geometric computation")
    print("  ✅ No traditional algorithm fallbacks")
    print("  ✅ Clean JSON interface for users")
    print("  ✅ Extensible without API changes")
    print()

    print("Future Extensibility:")
    print("  ✅ Add semantic similarity primitives")
    print("  ✅ Add contextual weighting")
    print("  ✅ Add domain-specific enhancements")
    print("  ✅ All through JSON config, no userland changes")


if __name__ == "__main__":
    test_kernel_enhancements()
    demonstrate_kernel_primitives()

    print("\n🎯 CONCLUSION:")
    print("Enhanced kernel primitives push holon's geometric limits")
    print("while maintaining clean JSON interface for Clojure-like userland.")
