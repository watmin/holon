#!/usr/bin/env python3
"""
Test Enhanced N-gram Primitives - Push holon's geometric limits.
Can we achieve better substring matching with kernel-level enhancements?
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

from holon.enhanced_encoder import EnhancedEncoder, EnhancedListEncodeMode
from holon import CPUStore
from holon.vector_manager import VectorManager


class EnhancedNgramQuoteFinder:
    """Quote finder using enhanced holon primitives."""

    def __init__(self, dimensions: int = 16000):
        self.vector_manager = VectorManager(dimensions=dimensions)
        self.encoder = EnhancedEncoder(self.vector_manager)
        self.store = CPUStore(dimensions=dimensions, encoder=self.encoder)
        self.quotes_data = []
        self.id_to_quote = {}

    def normalize_words(self, text: str) -> List[str]:
        """Normalize text."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def ingest_quotes_enhanced(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes using enhanced encoding primitives."""
        print(f"📝 Ingesting {len(quotes)} quotes with enhanced N-gram primitives...")

        units_data = []
        for quote in quotes:
            # Use enhanced N-gram with configurable parameters
            unit_data = {
                "text": {
                    "_encode_mode": "ngram_weighted",
                    "_encode_config": {
                        "length_penalty": True,
                        "corpus_stats": None  # Could be pre-computed for better weighting
                    },
                    "sequence": self.normalize_words(quote["text"])
                }
            }
            units_data.append(unit_data)
            self.quotes_data.append(quote)

        ids = self.store.insert_batch_json(units_data)

        for vector_id, quote in zip(ids, quotes):
            self.id_to_quote[vector_id] = quote

        print(f"✅ Enhanced ingestion complete: {len(ids)} quotes")
        return ids

    def search_enhanced_ngram(self, query_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using enhanced geometric primitives only.
        """
        words = self.normalize_words(query_phrase)

        # Use enhanced encoding for query
        probe_data = {
            "text": {
                "_encode_mode": "ngram_weighted",
                "_encode_config": {
                    "length_penalty": True
                },
                "sequence": words
            }
        }

        print(f"🔍 Enhanced search for: '{query_phrase}' → {words}")

        # Enhanced geometric search
        results = self.store.query(
            probe=json.dumps(probe_data),
            data_type="json",
            top_k=top_k,
            threshold=0.0
        )

        # Convert to our format
        formatted_results = []
        for item_id, score, data in results:
            original_quote = self.id_to_quote.get(item_id)
            if original_quote:
                result = {
                    "id": item_id,
                    "vsa_score": score,
                    "fuzzy_score": 0.0,
                    "combined_score": score,
                    "search_method": "enhanced_ngram",
                    "metadata": original_quote,
                    "reconstructed_text": original_quote["text"],
                    "search_words": words,
                }
                formatted_results.append(result)

        # Apply enhanced similarity scoring
        for result in formatted_results:
            # Re-encode query and target for comparison
            query_vec = self.encoder.encode_data(probe_data)
            # Note: We'd need to store vectors to do enhanced similarity comparison
            # For now, keep the VSA score

        formatted_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return formatted_results[:top_k]


def test_enhanced_primitives():
    """Test enhanced geometric primitives for substring matching."""
    print("🚀 Testing Enhanced Holon Primitives")
    print("=" * 50)

    finder = EnhancedNgramQuoteFinder()

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

    # Ingest with enhanced primitives
    finder.ingest_quotes_enhanced(quotes)

    # Test queries
    test_queries = [
        ("Everything depends upon relative minuteness", "exact match"),
        ("depends upon relative", "substring match"),
        ("calculus", "single word"),
        ("quantum physics", "negative control"),
    ]

    print("\n🔍 Enhanced Geometric Search Results:")
    print("-" * 40)

    for query, desc in test_queries:
        print(f"\n📝 Query: '{query}' ({desc})")
        results = finder.search_enhanced_ngram(query, top_k=3)

        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results):
                score = result["vsa_score"]
                quote_text = result["reconstructed_text"][:50]
                print(".3f")
        else:
            print("   ❌ No results found")


def compare_encoding_modes():
    """Compare different enhanced encoding modes."""
    print("\n🔬 Comparing Enhanced Encoding Modes")
    print("=" * 50)

    vector_manager = VectorManager(dimensions=1000)
    encoder = EnhancedEncoder(vector_manager)

    test_sequence = ["the", "quick", "brown", "fox", "jumps"]

    modes_to_test = [
        ("ngram", "ngram", {}),
        ("ngram_weighted", "ngram_weighted", {"length_penalty": True}),
        ("ngram_configurable", "ngram_configurable", {"n_sizes": [1, 2, 3], "weights": [0.3, 0.7, 0.5]}),
        ("subsequence_aligned", "subsequence_aligned", {"alignment_mode": "sliding_window", "window_size": 3}),
    ]

    print(f"Test sequence: {test_sequence}")

    for mode_name, mode, config in modes_to_test:
        try:
            vector = encoder.encode_list(test_sequence, mode=mode, **config)
            mean_val = vector.mean()
            std_val = vector.std()
            print(".3f")
        except Exception as e:
            print(f"   {mode_name}: ❌ Error - {e}")


def test_similarity_enhancements():
    """Test enhanced similarity scoring."""
    print("\n📊 Testing Enhanced Similarity Scoring")
    print("=" * 50)

    vector_manager = VectorManager(dimensions=1000)
    encoder = EnhancedEncoder(vector_manager)

    # Create test vectors
    seq1 = ["quick", "brown", "fox"]
    seq2 = ["the", "quick", "brown", "fox", "jumps"]
    seq3 = ["lazy", "dog", "runs"]

    vec1 = encoder.encode_list(seq1, mode="ngram")
    vec2 = encoder.encode_list(seq2, mode="ngram")
    vec3 = encoder.encode_list(seq3, mode="ngram")

    similarity_modes = [
        "cosine",
        "length_normalized",
        "contiguous_bonus"
    ]

    print("Sequence 1: quick brown fox")
    print("Sequence 2: the quick brown fox jumps")
    print("Sequence 3: lazy dog runs")
    print()

    for seq_name, vec, desc in [("Seq1", vec1, "short related"), ("Seq2", vec2, "long related"), ("Seq3", vec3, "unrelated")]:
        print(f"{seq_name} vs Seq1 ({desc}):")
        for mode in similarity_modes:
            try:
                if mode == "contiguous_bonus":
                    sim = encoder.compute_enhanced_similarity(vec1, vec, mode, contiguous_bonus=0.1, overlap_factor=0.5)
                else:
                    sim = encoder.compute_enhanced_similarity(vec1, vec, mode)
                print(".3f")
            except Exception as e:
                print(f"     {mode}: ❌ {e}")
        print()


if __name__ == "__main__":
    test_enhanced_primitives()
    compare_encoding_modes()
    test_similarity_enhancements()

    print("\n🎯 CONCLUSION:")
    print("Enhanced primitives push holon's geometric limits further.")
    print("Can we achieve 91.7% F1 without traditional algorithms?")