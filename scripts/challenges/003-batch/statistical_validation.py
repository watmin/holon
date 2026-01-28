#!/usr/bin/env python3
"""
Statistical Validation for Quote Finder - Challenge 4 Style Validation
Tests precision, recall, F1 scores, negative controls, and generalization.
"""

import json
import time
from typing import Dict, List, Tuple
from pathlib import Path

from holon import CPUStore, HolonClient
import json
import logging
import re
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteFinder:
    """Fixed Quote Finder using hybrid VSA + traditional search."""

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

    def create_unit_data(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """Create unit data for Holon (words only for vector, metadata stored separately)."""
        words = self.normalize_words(quote["text"])
        return {"words": {"_encode_mode": "ngram", "sequence": words}}

    def ingest_quotes(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes into Holon store."""
        logger.info(f"Ingesting {len(quotes)} quotes...")

        units_data = []
        for quote in quotes:
            unit_data = self.create_unit_data(quote)
            units_data.append(unit_data)
            self.quotes_data.append(quote)

        ids = self.client.insert_batch_json(units_data)

        for vector_id, quote in zip(ids, quotes):
            self.id_to_quote[vector_id] = quote

        logger.info(f"Successfully ingested {len(ids)} quote units")
        return ids

    def search_quotes_hybrid(self, query_phrase: str, top_k: int = 5, vsa_threshold: float = 0.3, fuzzy_threshold: float = 0.6, chapter_filter: str = None) -> List[Dict[str, Any]]:
        """Hybrid search: VSA/HDC for exact matches + traditional fuzzy matching."""
        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        guard = None
        if chapter_filter:
            guard = {"metadata": {"chapter": chapter_filter}}

        # Phase 1: VSA/HDC search for exact/high-similarity matches
        vsa_results = self.client.search_json(
            probe_data,
            top_k=top_k,
            threshold=vsa_threshold,
            guard=guard,
        )

        # Convert VSA results
        hybrid_results = []
        seen_quotes = set()

        for result_data in vsa_results:
            data_id = result_data["id"]
            vsa_score = result_data["score"]
            original_quote = self.id_to_quote.get(data_id)
            if original_quote:
                result = {
                    "id": data_id,
                    "vsa_score": vsa_score,
                    "fuzzy_score": 1.0,
                    "combined_score": vsa_score,
                    "search_method": "vsa_exact",
                    "metadata": original_quote,
                    "reconstructed_text": original_quote["text"],
                    "search_words": words,
                }
                hybrid_results.append(result)
                seen_quotes.add(data_id)

        # Phase 2: Add fuzzy text matching if needed
        if len(hybrid_results) < top_k:
            fuzzy_candidates = self._fuzzy_text_search(query_phrase, fuzzy_threshold, seen_quotes)
            for candidate in fuzzy_candidates:
                if len(hybrid_results) >= top_k:
                    break
                combined_score = candidate["fuzzy_score"] * 0.5
                result = {
                    "id": candidate["id"],
                    "vsa_score": 0.0,
                    "fuzzy_score": candidate["fuzzy_score"],
                    "combined_score": combined_score,
                    "search_method": "fuzzy_text",
                    "metadata": candidate["metadata"],
                    "reconstructed_text": candidate["text"],
                    "search_words": words,
                }
                hybrid_results.append(result)

        hybrid_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return hybrid_results[:top_k]

    def _fuzzy_text_search(self, query_phrase: str, min_similarity: float = 0.6, exclude_ids: set = None) -> List[Dict[str, Any]]:
        """Traditional fuzzy text search using difflib."""
        if exclude_ids is None:
            exclude_ids = set()

        import difflib

        query_lower = query_phrase.lower().strip()
        candidates = []

        for quote in self.quotes_data:
            quote_id = None
            for vid, q in self.id_to_quote.items():
                if q == quote:
                    quote_id = vid
                    break

            if quote_id in exclude_ids:
                continue

            quote_text_lower = quote["text"].lower()

            # Check for substring match first
            if query_lower in quote_text_lower:
                similarity = 1.0
            else:
                # Try substring matching - find best matching window
                query_words = query_lower.split()
                quote_words = quote_text_lower.split()

                best_similarity = 0.0
                # Try windows of different sizes
                for window_size in range(len(query_words), min(len(query_words) + 3, len(quote_words) + 1)):
                    for i in range(len(quote_words) - window_size + 1):
                        window = " ".join(quote_words[i:i + window_size])
                        matcher = difflib.SequenceMatcher(None, query_lower, window)
                        similarity = matcher.ratio()
                        best_similarity = max(best_similarity, similarity)

                similarity = best_similarity

                # Fallback to full text comparison if no good substring match
                if similarity < min_similarity:
                    matcher = difflib.SequenceMatcher(None, query_lower, quote_text_lower)
                    similarity = matcher.ratio()

            if similarity >= min_similarity:
                candidates.append({
                    "id": quote_id,
                    "text": quote["text"],
                    "metadata": quote,
                    "fuzzy_score": similarity,
                })

        candidates.sort(key=lambda x: x["fuzzy_score"], reverse=True)
        return candidates


class QuoteFinderValidator:
    """Statistical validator for quote finder using Challenge 4 methodology."""

    def __init__(self):
        self.finder = None
        self.test_queries = []
        self.ground_truth = {}

    def setup_test_data(self):
        """Set up comprehensive test data like Challenge 4's statistical validation."""
        # Load quotes
        quotes_file = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "quotes.txt"

        with open(quotes_file, "r") as f:
            quotes_text = f.read()

        # Parse quotes
        quotes = self._parse_quotes(quotes_text)

        # Initialize finder with the fixed hybrid implementation
        self.finder = QuoteFinder(dimensions=16000)
        ids = self.finder.ingest_quotes(quotes)

        # Create comprehensive test queries
        self._create_test_queries(quotes)

        print(f"✅ Set up validation with {len(quotes)} quotes and {len(self.test_queries)} test queries")

    def _parse_quotes(self, quotes_text: str) -> List[Dict]:
        """Parse quotes from text file."""
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
                        "page": (i + 1) * 3,  # Spread them out
                        "paragraph": 1,
                        "book_title": "Calculus Made Easy",
                    })

        return quotes

    def _create_test_queries(self, quotes: List[Dict]):
        """Create comprehensive test queries like Challenge 4's validation."""
        # Exact matches (should have high precision/recall)
        exact_queries = [
            ("Everything depends upon relative minuteness", [3]),  # Index 3
            ("Integration is the reverse of differentiation", [5]),  # Index 5
            ("d which merely means 'a little bit of.'", [1]),  # Index 1
        ]

        # Fuzzy matches (should find related quotes)
        fuzzy_queries = [
            ("depends on relative smallness", [3]),  # Similar to quote 3
            ("differential symbol", [1]),  # Related to differential quote
            ("calculus tricks", [7]),  # Related to quote 7
        ]

        # Partial matches
        partial_queries = [
            ("differential", [5]),  # Should find differentiation quote
            ("integral", [6]),  # Should find integral quotes
            ("slope", [4]),  # Should find tangent quote
        ]

        # Negative controls (should NOT find matches)
        negative_queries = [
            ("quantum physics", []),  # No calculus content
            ("machine learning", []),  # No matching content
            ("database normalization", []),  # Technical but wrong domain
        ]

        # Combine all
        self.test_queries = exact_queries + fuzzy_queries + partial_queries + negative_queries

        # Store ground truth mapping
        for i, quote in enumerate(quotes):
            self.ground_truth[i] = quote

    def run_statistical_validation(self) -> Dict:
        """Run comprehensive statistical validation like Challenge 4."""
        print("\n📊 Running Statistical Validation (Challenge 4 Style)")
        print("=" * 60)

        results = {
            "total_queries": len(self.test_queries),
            "exact_match_tests": 0,
            "fuzzy_match_tests": 0,
            "partial_match_tests": 0,
            "negative_tests": 0,
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
            "response_times": [],
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "true_negatives": 0,
        }

        for i, (query, expected_indices) in enumerate(self.test_queries):
            print(f"\n🔍 Testing query {i+1}/{len(self.test_queries)}: '{query}'")

            # Categorize test type
            if i < 3:
                results["exact_match_tests"] += 1
                test_type = "exact"
            elif i < 6:
                results["fuzzy_match_tests"] += 1
                test_type = "fuzzy"
            elif i < 9:
                results["partial_match_tests"] += 1
                test_type = "partial"
            else:
                results["negative_tests"] += 1
                test_type = "negative"

            # Time the query
            start_time = time.time()
            search_results = self.finder.search_quotes_hybrid(query, top_k=5)
            response_time = time.time() - start_time
            results["response_times"].append(response_time)

            # Analyze results
            precision, recall, f1, tp, fp, fn, tn = self._analyze_query_results(
                query, search_results, expected_indices, test_type
            )

            results["precision_scores"].append(precision)
            results["recall_scores"].append(recall)
            results["f1_scores"].append(f1)
            results["true_positives"] += tp
            results["false_positives"] += fp
            results["false_negatives"] += fn
            results["true_negatives"] += tn

            print(f"            Precision: {precision:.3f}")
            print(f"            Recall: {recall:.3f}")
            print(f"            F1: {f1:.3f}")
            print(f"            Response time: {response_time:.4f}s")
        # Calculate aggregate statistics
        results.update(self._calculate_aggregate_stats(results))

        self._print_validation_summary(results)
        return results

    def _analyze_query_results(self, query: str, results: List[Dict], expected_indices: List[int], test_type: str) -> Tuple[float, float, float, int, int, int, int]:
        """Analyze individual query results for precision/recall/F1."""
        # For simplicity, consider top result only (like Challenge 4's winner-takes-all)
        if not results:
            if expected_indices:  # Should have found something
                return 0.0, 0.0, 0.0, 0, 0, len(expected_indices), 1  # FP=0, FN=len(expected), TN=1
            else:  # Negative test, correctly found nothing
                return 1.0, 1.0, 1.0, 0, 0, 0, 1  # Perfect negative result

        # Check if top result is correct
        top_result = results[0]
        found_correct = False

        # For negative tests, any result is wrong
        if test_type == "negative":
            return 0.0, 0.0, 0.0, 0, 1, 0, 0  # FP=1, no relevant docs

        # Check if we found an expected result
        for expected_idx in expected_indices:
            expected_quote = self.ground_truth.get(expected_idx, {})
            if expected_quote.get("text", "").lower() in top_result.get("reconstructed_text", "").lower():
                found_correct = True
                break

        if found_correct:
            return 1.0, 1.0, 1.0, 1, 0, 0, 0  # Perfect match
        else:
            return 0.0, 0.0, 0.0, 0, 1, len(expected_indices), 0  # FP=1, FN=len(expected)

    def _calculate_aggregate_stats(self, results: Dict) -> Dict:
        """Calculate aggregate statistics like Challenge 4."""
        precision_scores = [s for s in results["precision_scores"] if not (s == 0.0 and results["precision_scores"].index(s) >= 9)]  # Exclude negative tests for precision
        recall_scores = results["recall_scores"]
        f1_scores = results["f1_scores"]

        return {
            "avg_precision": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            "avg_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "avg_response_time": sum(results["response_times"]) / len(results["response_times"]),
            "total_tp": results["true_positives"],
            "total_fp": results["false_positives"],
            "total_fn": results["false_negatives"],
            "total_tn": results["true_negatives"],
            "accuracy": (results["true_positives"] + results["true_negatives"]) / len(self.test_queries),
        }

    def _print_validation_summary(self, results: Dict):
        """Print comprehensive validation summary like Challenge 4."""
        print("\n" + "="*60)
        print("🎯 STATISTICAL VALIDATION SUMMARY")
        print("="*60)

        print("📈 Performance Metrics:")
        print(f"   Average Precision: {results['avg_precision']:.1%}")
        print(f"   Average Recall: {results['avg_recall']:.1%}")
        print(f"   Average F1 Score: {results['avg_f1']:.1%}")
        print(f"   Average Response Time: {results['avg_response_time']:.4f}s")

        print("\n🔍 Confusion Matrix:")
        print(f"   True Positives:  {results['total_tp']}")
        print(f"   False Positives: {results['total_fp']}")
        print(f"   False Negatives: {results['total_fn']}")
        print(f"   True Negatives:  {results['total_tn']}")

        print("\n✅ Test Coverage:")
        print(f"   Exact Match Tests: {results['exact_match_tests']}")
        print(f"   Fuzzy Match Tests: {results['fuzzy_match_tests']}")
        print(f"   Partial Match Tests: {results['partial_match_tests']}")
        print(f"   Negative Control Tests: {results['negative_tests']}")

        # Challenge 4 style success assessment
        if results["avg_f1"] > 0.8:
            assessment = "🎉 EXCELLENT - Challenge 4 level performance!"
        elif results["avg_f1"] > 0.6:
            assessment = "✅ GOOD - Significant improvement over VSA-only"
        elif results["avg_f1"] > 0.4:
            assessment = "⚠️  FAIR - Working but needs refinement"
        else:
            assessment = "❌ POOR - Still needs work"

        print(f"\n🏆 Overall Assessment: {assessment}")
        print(f"   F1 Score: {results['avg_f1']:.3f} (Challenge 4 target: >0.7)")


def main():
    """Run the statistical validation."""
    import re

    validator = QuoteFinderValidator()
    validator.setup_test_data()
    results = validator.run_statistical_validation()

    return results


if __name__ == "__main__":
    main()
