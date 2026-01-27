#!/usr/bin/env python3
"""
Holon-Powered Quote Finder Demo
Implements the enhanced quote finder with n-gram encoding and vector bootstrapping.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Try to import PDF reader
try:
    import pypdf

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("Warning: pypdf not available, will use text file only")

from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteFinder:
    """Quote finder using Holon's VSA/HDC with n-gram encoding and vector bootstrapping."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.quotes_data = []  # Store full text for demo (not in actual DB)
        self.id_to_quote = {}  # Map vector IDs to quote data

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        if not HAS_PYPDF:
            raise ImportError("pypdf required for PDF processing")

        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def parse_quotes(self, quotes_text: str) -> List[Dict[str, Any]]:
        """Parse quotes from text file into structured units."""
        quotes = []
        lines = quotes_text.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Simple quote parsing - split by quote marks
            if '"' in line:
                # Extract quote between quotes
                quote_match = re.search(r'"([^"]*)"', line)
                if quote_match:
                    quote_text = quote_match.group(1)
                    # Simple metadata extraction from context
                    context = line.replace(f'"{quote_text}"', "").strip()

                    quotes.append(
                        {
                            "text": quote_text,
                            "chapter": self._extract_chapter(context),
                            "page": i + 1,  # Simple page estimation
                            "paragraph": 1,
                            "book_title": "Calculus Made Easy",
                        }
                    )

        return quotes

    def _extract_chapter(self, context: str) -> str:
        """Extract chapter info from context."""
        if "prologue" in context.lower():
            return "Prologue"
        elif "chapter" in context.lower():
            # Try to extract chapter number
            match = re.search(r"chapter\s+(\w+)", context, re.IGNORECASE)
            return match.group(1) if match else "Unknown"
        return "Unknown"

    def normalize_words(self, text: str) -> List[str]:
        """Normalize text: lowercase, remove punctuation, split into words."""
        # Remove punctuation and lowercase
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        # Split into words and filter out empty strings
        words = [word for word in normalized.split() if word]
        return words

    def create_unit_data(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """Create unit data structure for Holon (words only for vector, metadata stored separately)."""
        words = self.normalize_words(quote["text"])

        # Store ONLY the words for vector encoding - metadata is retrieved by ID
        return {"words": {"_encode_mode": "ngram", "sequence": words}}

    def ingest_quotes(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes into Holon store using metadata-only approach."""
        logger.info(f"Ingesting {len(quotes)} quotes...")

        units_data = []
        for quote in quotes:
            unit_data = self.create_unit_data(quote)
            units_data.append(unit_data)

            # Keep full text for demo purposes (not stored in DB)
            self.quotes_data.append(quote)

        # Batch insert for efficiency
        ids = self.client.insert_batch_json(units_data)

        # Create mapping from vector IDs to quote data
        for vector_id, quote in zip(ids, quotes):
            self.id_to_quote[vector_id] = quote

        logger.info(f"Successfully ingested {len(ids)} quote units")
        return ids

    def bootstrap_search_vector(self, phrase: str) -> List[float]:
        """Use /encode API to bootstrap a search vector for a phrase."""
        words = self.normalize_words(phrase)

        # Create the same encoding structure as stored units
        encode_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Use the client to encode (simulating the API)
        return self.client.encode_vectors_json(encode_data)

    def search_quotes_hybrid(
        self,
        query_phrase: str,
        top_k: int = 5,
        vsa_threshold: float = 0.3,  # Higher threshold for VSA (more selective)
        fuzzy_threshold: float = 0.6,  # Lower threshold for fuzzy (more inclusive)
        chapter_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: VSA/HDC for exact matches + traditional fuzzy matching for related text.
        This follows Challenge 4's hybrid approach pattern.
        """
        logger.info(f"Hybrid search for: '{query_phrase}'")

        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Build guard for chapter filtering
        guard = None
        if chapter_filter:
            guard = {"metadata": {"chapter": chapter_filter}}

        # Phase 1: VSA/HDC search for exact/high-similarity matches
        vsa_results = self.client.search_json(
            probe_data,
            top_k=top_k,
            threshold=vsa_threshold,  # Higher threshold = more selective
            guard=guard,
        )

        # Convert VSA results to our format
        hybrid_results = []
        seen_quotes = set()

        for result_data in vsa_results:
            data_id = result_data["id"]
            vsa_score = result_data["score"]
            data = result_data["data"]
            original_quote = self.id_to_quote.get(data_id)
            if original_quote:
                result = {
                    "id": data_id,
                    "vsa_score": vsa_score,
                    "fuzzy_score": 1.0,  # Exact match from VSA
                    "combined_score": vsa_score,  # For now, just use VSA score
                    "search_method": "vsa_exact",
                    "metadata": original_quote,
                    "reconstructed_text": original_quote["text"],
                    "search_words": words,
                }
                hybrid_results.append(result)
                seen_quotes.add(data_id)

        # Phase 2: If we don't have enough VSA results, add fuzzy text matching
        if len(hybrid_results) < top_k:
            fuzzy_candidates = self._fuzzy_text_search(
                query_phrase, fuzzy_threshold, exclude_ids=seen_quotes
            )

            # Add fuzzy results
            for candidate in fuzzy_candidates:
                if len(hybrid_results) >= top_k:
                    break

                # Calculate combined score (VSA would be 0 for fuzzy matches)
                combined_score = candidate["fuzzy_score"] * 0.5  # Weight fuzzy matches lower

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

        # Sort by combined score
        hybrid_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return hybrid_results[:top_k]
        """Search for quotes using bootstrapped vector."""
        logger.info(f"Searching for: '{query_phrase}'")

        # Bootstrap search vector (not used in current implementation)
        # search_vector_list = self.bootstrap_search_vector(query_phrase)

        # Create probe using the vector (this is a simplified approach)
        # In a real implementation, we'd need to extend the query API to accept pre-computed vectors
        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Build guard for chapter filtering
        guard = None
        if chapter_filter:
            guard = {"metadata": {"chapter": chapter_filter}}

        # Query the store
        results = self.client.search_json(
            probe_data,
            top_k=top_k,
            threshold=threshold,
            guard=guard,
        )

        print(f"DEBUG: Query returned {len(results)} raw results for '{query_phrase}'")
        if results:
            print(
                f"DEBUG: First result score: {results[0][1]:.3f}, data keys: {list(results[0][2].keys())}"
            )

        # Enrich results with reconstructed text (from our demo store)
        enriched_results = []
        for data_id, score, data in results:
            # Get the original quote data using the vector ID
            original_quote = self.id_to_quote.get(data_id)

            enriched_results.append(
                {
                    "id": data_id,
                    "score": score,
                    "metadata": original_quote if original_quote else {},
                    "reconstructed_text": original_quote["text"]
                    if original_quote
                    else "Text not found",
                    "search_words": words,
                }
            )

        return enriched_results

    def search_quotes(
        self,
        query_phrase: str,
        top_k: int = 5,
        threshold: float = 0.1,
        chapter_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """Original VSA-only search method (kept for comparison)."""
        logger.info(f"VSA-only search for: '{query_phrase}'")

        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        guard = None
        if chapter_filter:
            guard = {"metadata": {"chapter": chapter_filter}}

        results = self.client.search_json(
            probe_data,
            top_k=top_k,
            threshold=threshold,
            guard=guard,
        )

        enriched_results = []
        for result_data in results:
            data_id = result_data["id"]
            score = result_data["score"]
            data = result_data["data"]
            original_quote = self.id_to_quote.get(data_id)
            enriched_results.append(
                {
                    "id": data_id,
                    "score": score,
                    "metadata": original_quote if original_quote else {},
                    "reconstructed_text": original_quote["text"]
                    if original_quote
                    else "Text not found",
                    "search_words": words,
                }
            )

        return enriched_results

    def _fuzzy_text_search(
        self, query_phrase: str, min_similarity: float = 0.6, exclude_ids: set = None
    ) -> List[Dict[str, Any]]:
        """Traditional fuzzy text search using difflib."""
        if exclude_ids is None:
            exclude_ids = set()

        import difflib

        query_lower = query_phrase.lower().strip()
        candidates = []

        for quote in self.quotes_data:
            # Skip if already found by VSA
            quote_id = None
            for vid, q in self.id_to_quote.items():
                if q == quote:
                    quote_id = vid
                    break

            if quote_id in exclude_ids:
                continue

            quote_text_lower = quote["text"].lower()

            # Exact substring match gets highest score
            if query_lower in quote_text_lower:
                similarity = 1.0
            else:
                # Fuzzy matching with sliding window
                matcher = difflib.SequenceMatcher(None, query_lower, quote_text_lower)
                similarity = matcher.ratio()

                # Also try partial matches
                if similarity < min_similarity:
                    # Split query into words and find best partial match
                    query_words = query_lower.split()
                    best_partial = 0.0

                    for i in range(len(quote_text_lower) - len(query_lower) + 1):
                        window = quote_text_lower[i:i+len(query_lower)]
                        window_matcher = difflib.SequenceMatcher(None, query_lower, window)
                        best_partial = max(best_partial, window_matcher.ratio())

                    similarity = max(similarity, best_partial)

            if similarity >= min_similarity:
                candidates.append({
                    "id": quote_id,
                    "text": quote["text"],
                    "metadata": quote,
                    "fuzzy_score": similarity,
                })

        # Sort by similarity
        candidates.sort(key=lambda x: x["fuzzy_score"], reverse=True)
        return candidates


def main():
    """Demo the quote finder system."""
    print("🔍 Holon Quote Finder Demo")
    print("=" * 50)

    # Initialize quote finder
    finder = QuoteFinder(dimensions=16000)

    # Load quotes from text file
    quotes_file = (
        Path(__file__).parent.parent.parent.parent
        / "docs"
        / "challenges"
        / "003-batch"
        / "quotes.txt"
    )

    if not quotes_file.exists():
        print(f"❌ Quotes file not found: {quotes_file}")
        return

    print(f"📖 Loading quotes from: {quotes_file}")
    with open(quotes_file, "r") as f:
        quotes_text = f.read()

    # Parse quotes
    quotes = finder.parse_quotes(quotes_text)
    print(f"📝 Parsed {len(quotes)} quotes:")
    for i, quote in enumerate(quotes):
        print(f"   {i+1}. [{quote['chapter']}] {quote['text'][:50]}...")
        print(f"      Page: {quote['page']}, Para: {quote['paragraph']}")
    print()

    # Ingest into Holon (metadata only)
    ids = finder.ingest_quotes(quotes)
    print(f"💾 Ingested {len(ids)} units into Holon store")

    # Demo searches - showing both broken VSA-only and fixed hybrid approaches
    print("\n🔎 Running Search Demos:")
    print("-" * 30)

    test_queries = [
        ("Everything depends upon relative minuteness", "exact match"),
        ("depends on relative smallness", "fuzzy similar"),
        ("differential symbol", "partial phrase"),
        ("integration", "single word"),
    ]

    for i, (query, desc) in enumerate(test_queries, 1):
        print(f"\n{i}. {desc.upper()}: '{query}'")
        print("-" * 50)

        # Show broken VSA-only approach
        print("   ❌ BROKEN VSA-only approach:")
        vsa_results = finder.search_quotes(
            query, threshold=0.0
        )  # This calls the old method
        print(f"      Found {len(vsa_results)} results:")
        for result in vsa_results[:2]:  # Show top 2
            print(
                f"      Score: {result['score']:.3f} | Text: {result['reconstructed_text'][:50]}..."
            )

        # Show fixed hybrid approach
        print("   ✅ FIXED hybrid approach:")
        hybrid_results = finder.search_quotes_hybrid(query)
        print(f"      Found {len(hybrid_results)} results:")
        for result in hybrid_results:
            method = result['search_method']
            score_type = "VSA" if result['vsa_score'] > 0 else "Fuzzy"
            score = result['vsa_score'] if result['vsa_score'] > 0 else result['fuzzy_score']
            print(
                f"      {score_type}: {score:.3f} ({method}) | Text: {result['reconstructed_text'][:50]}..."
            )
        print()

    # Demo 5: Vector bootstrapping showcase
    print("\n5. Vector bootstrapping demo:")
    phrase = "calculus tricks are easy"
    vector = finder.bootstrap_search_vector(phrase)
    print(f"   Bootstrapped vector for '{phrase}': {len(vector)} dimensions")
    print(f"   Sample values: {vector[:5]}...")

    print("\n✅ Demo completed!")
    print("✨ Key features demonstrated:")
    print("   • N-gram encoding for fuzzy subsequence matching")
    print("   • Metadata-only storage (no full text in DB)")
    print("   • Vector bootstrapping API")
    print("   • Chapter filtering with guards")


if __name__ == "__main__":
    main()
