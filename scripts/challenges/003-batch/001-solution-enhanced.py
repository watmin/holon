#!/usr/bin/env python3
"""
Enhanced Quote Finder with New Kernel Primitives

Demonstrates using the new VSA primitives for advanced text search:
- prototype: Learn topic signatures (calculus terms, chapter intros)
- difference: Find unique aspects of quotes
- blend: Multi-topic fuzzy search
- amplify: Boost topic signals
- negate: Exclude topics ("X but NOT Y")
"""

import json
import re
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity


def normalize_text(text: str) -> List[str]:
    """Normalize text to word list."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) > 1]


def parse_result_data(data):
    """Parse result data from string to dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return json.loads(data)
    return data


# Sample quotes organized by topic for prototype learning
TOPIC_QUOTES = {
    "differentiation": [
        "The simplest case is dx or dy which merely means a little bit of x or y",
        "dy dx is the slope of the tangent to the curve at that point",
        "The differential coefficient is the rate of change",
        "To differentiate means to find the differential coefficient",
        "The derivative shows the rate of change of y with respect to x",
    ],
    "integration": [
        "Integration is the reverse of differentiation",
        "The integral sign means the sum of all the little bits",
        "To integrate means to find the area under the curve",
        "Integration gives us the original function from its derivative",
        "The indefinite integral has an arbitrary constant",
    ],
    "limits": [
        "Everything depends upon relative minuteness",
        "The limit is the value approached as we get infinitely close",
        "As the increment approaches zero the ratio approaches the limit",
        "Infinitely small quantities approach their limit",
        "The limiting value is what we get as dx becomes negligible",
    ],
    "encouragement": [
        "What one fool can do, another can",
        "Do not be afraid of the calculus",
        "The terror that this may inspire is quite unnecessary",
        "These things are quite easy if only you have patience",
        "Being able to do so is no more than anyone can accomplish",
    ],
}


class EnhancedQuoteFinder:
    """Quote finder with advanced primitives."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.topic_prototypes = {}
        self.quote_vectors = {}

    def ingest_quotes(self):
        """Ingest sample quotes and learn topic prototypes."""
        print("\n📥 Ingesting quotes and learning topic prototypes...")

        all_quotes = []
        for topic, quotes in TOPIC_QUOTES.items():
            topic_vecs = []

            for quote in quotes:
                words = normalize_text(quote)
                unit = {
                    "unit_id": str(uuid.uuid4()),
                    "words": {"$mode": "ngram", "sequence": words},
                    "metadata": {"topic": topic, "quote": quote[:50] + "..."}
                }

                # Insert and get vector
                uid = self.client.insert_json(unit)
                vec = np.array(self.client.encode_vectors_json(unit))

                self.quote_vectors[quote[:30]] = vec
                topic_vecs.append(vec)
                all_quotes.append((quote, topic, vec))

            # Learn topic prototype
            if topic_vecs:
                prototype = self.store.prototype(topic_vecs, threshold=0.5)
                self.topic_prototypes[topic] = prototype
                print(f"   → Learned '{topic}' prototype from {len(topic_vecs)} quotes")

        print(f"\n   Total: {len(all_quotes)} quotes, {len(self.topic_prototypes)} topic prototypes")
        return all_quotes

    def classify_quote(self, quote: str) -> str:
        """Classify a quote by topic using prototypes."""
        words = normalize_text(quote)
        probe = {"words": {"$mode": "ngram", "sequence": words}}
        vec = np.array(self.client.encode_vectors_json(probe))

        best_topic = None
        best_sim = -1

        for topic, proto in self.topic_prototypes.items():
            sim = normalized_dot_similarity(vec, proto)
            if sim > best_sim:
                best_sim = sim
                best_topic = topic

        return best_topic, best_sim

    def blend_search(self, topic1: str, topic2: str, alpha: float = 0.5) -> List[tuple]:
        """Search for quotes matching a blend of two topics."""
        if topic1 not in self.topic_prototypes or topic2 not in self.topic_prototypes:
            return []

        blended = self.store.blend(
            self.topic_prototypes[topic1],
            self.topic_prototypes[topic2],
            alpha=alpha
        )

        # Score all quotes against blend
        results = []
        for topic, quotes in TOPIC_QUOTES.items():
            for quote in quotes:
                key = quote[:30]
                if key in self.quote_vectors:
                    sim = normalized_dot_similarity(self.quote_vectors[key], blended)
                    results.append((quote[:50], topic, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]

    def amplified_search(self, query: str, boost_topic: str, strength: float = 2.0) -> List[tuple]:
        """Search with amplified topic signal."""
        words = normalize_text(query)
        probe = {"words": {"$mode": "ngram", "sequence": words}}
        base_vec = np.array(self.client.encode_vectors_json(probe))

        if boost_topic not in self.topic_prototypes:
            return []

        amplified = self.store.amplify(base_vec, self.topic_prototypes[boost_topic], strength)

        # Score all quotes
        results = []
        for topic, quotes in TOPIC_QUOTES.items():
            for quote in quotes:
                key = quote[:30]
                if key in self.quote_vectors:
                    sim = normalized_dot_similarity(self.quote_vectors[key], amplified)
                    results.append((quote[:50], topic, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]

    def negated_search(self, include_topic: str, exclude_topic: str) -> List[tuple]:
        """Search for one topic but NOT another."""
        if include_topic not in self.topic_prototypes or exclude_topic not in self.topic_prototypes:
            return []

        negated = self.store.negate(
            self.topic_prototypes[include_topic],
            self.topic_prototypes[exclude_topic],
            method="subtract"
        )

        # Score all quotes
        results = []
        for topic, quotes in TOPIC_QUOTES.items():
            for quote in quotes:
                key = quote[:30]
                if key in self.quote_vectors:
                    sim = normalized_dot_similarity(self.quote_vectors[key], negated)
                    results.append((quote[:50], topic, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]

    def find_unique_aspects(self, quote: str) -> np.ndarray:
        """Find what makes a quote unique compared to the average."""
        words = normalize_text(quote)
        probe = {"words": {"$mode": "ngram", "sequence": words}}
        quote_vec = np.array(self.client.encode_vectors_json(probe))

        # Create average of all prototypes
        all_protos = list(self.topic_prototypes.values())
        avg_proto = self.store.bundle(all_protos)

        # Difference shows what's unique
        unique = self.store.difference(avg_proto, quote_vec)
        return unique


def main():
    print("=" * 70)
    print("ENHANCED QUOTE FINDER WITH NEW PRIMITIVES")
    print("=" * 70)

    finder = EnhancedQuoteFinder()
    all_quotes = finder.ingest_quotes()

    # ========================================
    # TEST 1: Topic Classification
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: AUTOMATIC TOPIC CLASSIFICATION")
    print("=" * 70)

    test_quotes = [
        "The derivative of x squared is 2x",  # Should match differentiation
        "Find the area under the curve",  # Should match integration
        "As dx approaches zero",  # Should match limits
        "Anyone can learn this",  # Should match encouragement
    ]

    correct = 0
    expected = ["differentiation", "integration", "limits", "encouragement"]

    for quote, exp in zip(test_quotes, expected):
        topic, sim = finder.classify_quote(quote)
        match = "✅" if topic == exp else "❌"
        if topic == exp:
            correct += 1
        print(f"\n   '{quote[:40]}...'")
        print(f"   → Classified as: {topic} (sim: {sim:.4f}) {match}")

    print(f"\n   Classification accuracy: {correct}/{len(test_quotes)}")

    # ========================================
    # TEST 2: Blend Search
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 2: MULTI-TOPIC BLEND SEARCH")
    print("=" * 70)

    print("\n   Searching for: differentiation + integration blend (50/50)")
    results = finder.blend_search("differentiation", "integration", alpha=0.5)

    if results:
        print("   Top matches:")
        for quote, topic, sim in results:
            print(f"     [{topic}] {quote}... (sim: {sim:.4f})")

    print("\n   Searching for: limits + encouragement blend")
    results = finder.blend_search("limits", "encouragement", alpha=0.5)

    if results:
        print("   Top matches:")
        for quote, topic, sim in results:
            print(f"     [{topic}] {quote}... (sim: {sim:.4f})")

    # ========================================
    # TEST 3: Amplified Search
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 3: AMPLIFIED TOPIC SEARCH")
    print("=" * 70)

    print("\n   Query: 'rate of change' with differentiation boost (2x)")
    results = finder.amplified_search("rate of change", "differentiation", strength=2.0)

    if results:
        print("   Top matches:")
        for quote, topic, sim in results:
            marker = "⭐" if topic == "differentiation" else ""
            print(f"     [{topic}] {quote}... (sim: {sim:.4f}) {marker}")

    # ========================================
    # TEST 4: Negated Search
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 4: NEGATED SEARCH (X but NOT Y)")
    print("=" * 70)

    print("\n   Searching for: differentiation but NOT integration")
    results = finder.negated_search("differentiation", "integration")

    if results:
        print("   Top matches (should be differentiation, not integration):")
        for quote, topic, sim in results:
            marker = "✅" if topic == "differentiation" else ("❌" if topic == "integration" else "")
            print(f"     [{topic}] {quote}... (sim: {sim:.4f}) {marker}")

    print("\n   Searching for: encouragement but NOT limits")
    results = finder.negated_search("encouragement", "limits")

    if results:
        print("   Top matches:")
        for quote, topic, sim in results:
            marker = "✅" if topic == "encouragement" else ""
            print(f"     [{topic}] {quote}... (sim: {sim:.4f}) {marker}")

    # ========================================
    # TEST 5: Unique Aspect Detection
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 5: UNIQUE ASPECT DETECTION")
    print("=" * 70)

    # Find what makes the famous quote unique
    famous = "What one fool can do, another can"
    unique_vec = finder.find_unique_aspects(famous)

    print(f"\n   Analyzing uniqueness of: '{famous}'")
    print(f"   → Computed difference vector from average prototype")

    # Find other quotes similar to this "unique" aspect
    results = []
    for topic, quotes in TOPIC_QUOTES.items():
        for quote in quotes:
            key = quote[:30]
            if key in finder.quote_vectors:
                sim = normalized_dot_similarity(finder.quote_vectors[key], unique_vec)
                results.append((quote[:50], topic, sim))

    results.sort(key=lambda x: x[2], reverse=True)
    print("   Quotes sharing unique aspects:")
    for quote, topic, sim in results[:3]:
        print(f"     [{topic}] {quote}... (sim: {sim:.4f})")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: NEW PRIMITIVES FOR TEXT SEARCH")
    print("=" * 70)
    print(f"""
    1. PROTOTYPE: Automatic topic classification
       → {correct}/{len(test_quotes)} accuracy on unseen quotes

    2. BLEND: Multi-topic fuzzy search
       → Find quotes combining two concepts

    3. AMPLIFY: Boost topic signals
       → Improve precision for specific topics

    4. NEGATE: Exclusion queries
       → "X but NOT Y" search patterns

    5. DIFFERENCE: Unique aspect detection
       → Find what makes a quote special

    These primitives enable semantic text search
    beyond simple keyword matching!
    """)


if __name__ == "__main__":
    main()
