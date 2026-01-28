#!/usr/bin/env python3
"""
NGRAM-only Quote Finder - Test if holon can solve substring matching with NGRAM alone.
No difflib fallback - pure geometric approach.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode


class NgramOnlyQuoteFinder:
    """Quote finder using only NGRAM encoding - no traditional algorithms."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.quotes_data = []
        self.id_to_quote = {}

    def normalize_words(self, text: str) -> List[str]:
        """Normalize text: lowercase, remove punctuation, split into words."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def create_ngram_unit(self, text: str) -> Dict[str, Any]:
        """Create unit using NGRAM encoding only."""
        words = self.normalize_words(text)
        return {"words": {"_encode_mode": "ngram", "sequence": words}}

    def ingest_quotes(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes using NGRAM encoding."""
        print(f"📝 Ingesting {len(quotes)} quotes with NGRAM encoding...")

        units_data = []
        for quote in quotes:
            unit_data = self.create_ngram_unit(quote["text"])
            units_data.append(unit_data)
            self.quotes_data.append(quote)

        ids = self.client.insert_batch_json(units_data)

        for vector_id, quote in zip(ids, quotes):
            self.id_to_quote[vector_id] = quote

        print(f"✅ Ingested {len(ids)} quote units")
        return ids

    def search_quotes_ngram_only(self, query_phrase: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search using only NGRAM encoding - no traditional fallbacks.
        This tests the limits of holon's geometric substring matching.
        """
        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        print(f"🔍 Searching for: '{query_phrase}' → {words}")

        # Pure geometric search
        results = self.client.search_json(
            probe=probe_data,
            top_k=top_k,
            threshold=threshold
        )

        # Convert to our format
        formatted_results = []
        for result in results:
            data_id = result["id"]
            vsa_score = result["score"]
            original_quote = self.id_to_quote.get(data_id)

            if original_quote:
                formatted_result = {
                    "id": data_id,
                    "vsa_score": vsa_score,
                    "fuzzy_score": 0.0,  # No traditional scoring
                    "combined_score": vsa_score,
                    "search_method": "ngram_only",
                    "metadata": original_quote,
                    "reconstructed_text": original_quote["text"],
                    "search_words": words,
                }
                formatted_results.append(formatted_result)

        formatted_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return formatted_results[:top_k]

    def analyze_ngram_performance(self, query_phrase: str) -> Dict[str, Any]:
        """Analyze how well NGRAM encoding performs for different query types."""
        analysis = {
            "query": query_phrase,
            "query_words": self.normalize_words(query_phrase),
            "exact_matches": [],
            "partial_matches": [],
            "no_matches": [],
        }

        # Test with very low threshold to see all possible matches
        results = self.search_quotes_ngram_only(query_phrase, top_k=10, threshold=0.0)

        analysis["total_results"] = len(results)
        analysis["results"] = results

        # Categorize results
        for result in results:
            quote_text = result["reconstructed_text"].lower()
            query_lower = query_phrase.lower()

            if query_lower in quote_text:
                analysis["exact_matches"].append(result)
            elif any(word in quote_text for word in analysis["query_words"]):
                analysis["partial_matches"].append(result)
            else:
                analysis["no_matches"].append(result)

        return analysis


def test_ngram_substring_capabilities():
    """Test what NGRAM can and cannot do for substring matching."""
    print("🧪 Testing NGRAM-only substring capabilities")
    print("=" * 60)

    finder = NgramOnlyQuoteFinder()

    # Load quotes
    quotes_file = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "quotes.txt"

    with open(quotes_file, "r") as f:
        quotes_text = f.read()

    # Parse quotes
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

    finder.ingest_quotes(quotes)

    # Test cases
    test_queries = [
        # Exact matches (should work well)
        "Everything depends upon relative minuteness",
        "Integration is the reverse of differentiation",

        # Partial phrases (test substring capability)
        "depends upon relative",
        "reverse of differentiation",

        # Single words (minimal test)
        "calculus",
        "differentiation",

        # Non-existent (negative control)
        "quantum physics",
    ]

    print("\n🔍 Testing various query types:")
    print("-" * 40)

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        analysis = finder.analyze_ngram_performance(query)

        print(f"   Words: {analysis['query_words']}")
        print(f"   Total results: {analysis['total_results']}")
        print(f"   Exact matches: {len(analysis['exact_matches'])}")
        print(f"   Partial matches: {len(analysis['partial_matches'])}")
        print(f"   No matches: {len(analysis['no_matches'])}")

        # Show top result details
        if analysis["results"]:
            top = analysis["results"][0]
            print(".3f")
            print(f"   Found: '{top['reconstructed_text'][:50]}...'")
        else:
            print("   ❌ No results found")


def identify_missing_primitives():
    """Identify what geometric primitives holon needs for better substring matching."""
    print("\n🔧 Identifying Missing Geometric Primitives")
    print("=" * 60)

    print("CURRENT NGRAM CAPABILITIES:")
    print("  ✅ Bigram binding: (word_i ⊙ word_i+1)")
    print("  ✅ Bundling: Sum of all bigrams + singles")
    print("  ✅ Similarity search: Cosine similarity in hyperspace")
    print()

    print("WHAT WORKS WELL:")
    print("  ✅ Exact sequence matching")
    print("  ✅ Partial word overlap")
    print("  ✅ Fuzzy matching within same structure")
    print()

    print("WHAT'S MISSING FOR SUBSTRING MATCHING:")
    print("  ❌ Sliding window geometric search")
    print("  ❌ Sequence alignment algorithms")
    print("  ❌ Unbinding/extraction operations")
    print("  ❌ Cross-sequence similarity measures")
    print()

    print("POTENTIAL NEW PRIMITIVES:")
    print("  🔧 sequence_align(A, B) - Align sequences geometrically")
    print("  🔧 extract_subsequence(full, start, end) - Extract subsequences")
    print("  🔧 sliding_window_search(query, window_size) - Geometric sliding window")
    print("  🔧 cross_similarity(A, B) - Similarity across different structures")
    print()

    print("WHY NGRAM + DIFFLIB WORKS:")
    print("  - NGRAM provides geometric rough matching")
    print("  - difflib provides precise substring alignment")
    print("  - Hybrid approach leverages both strengths")


if __name__ == "__main__":
    test_ngram_substring_capabilities()
    identify_missing_primitives()

    print("\n🎯 CONCLUSION:")
    print("NGRAM alone provides geometric substring matching,")
    print("but lacks the precision of traditional algorithms.")
    print("Missing primitives would enable pure geometric solutions.")
