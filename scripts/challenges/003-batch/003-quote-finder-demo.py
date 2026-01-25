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

from holon import CPUStore
from holon.encoder import ListEncodeMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteFinder:
    """Quote finder using Holon's VSA/HDC with n-gram encoding and vector bootstrapping."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.quotes_data = []  # Store full text for demo (not in actual DB)

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
        """Create unit data structure for Holon (metadata only, no full text)."""
        words = self.normalize_words(quote["text"])

        return {
            "words": {"_encode_mode": "ngram", "sequence": words},
            "metadata": {
                "chapter": quote["chapter"],
                "paragraph": quote["paragraph"],
                "page": quote["page"],
                "book_title": quote["book_title"],
            },
        }

    def ingest_quotes(self, quotes: List[Dict[str, Any]]) -> List[str]:
        """Ingest quotes into Holon store using metadata-only approach."""
        logger.info(f"Ingesting {len(quotes)} quotes...")

        units_data = []
        for quote in quotes:
            unit_data = self.create_unit_data(quote)
            units_data.append(json.dumps(unit_data))

            # Keep full text for demo purposes (not stored in DB)
            self.quotes_data.append(quote)

        # Batch insert for efficiency
        ids = self.store.batch_insert(units_data, data_type="json")
        logger.info(f"Successfully ingested {len(ids)} quote units")
        return ids

    def bootstrap_search_vector(self, phrase: str) -> List[float]:
        """Use /encode API to bootstrap a search vector for a phrase."""
        words = self.normalize_words(phrase)

        # Create the same encoding structure as stored units
        encode_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Use the encoder directly (simulating the API)
        vector = self.store.encoder.encode_data(encode_data)

        # Convert to list for API-like response
        cpu_vector = self.store.vector_manager.to_cpu(vector)
        return cpu_vector.tolist()

    def search_quotes(
        self,
        query_phrase: str,
        top_k: int = 5,
        threshold: float = 0.1,
        chapter_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """Search for quotes using bootstrapped vector."""
        logger.info(f"Searching for: '{query_phrase}'")

        # Bootstrap search vector
        search_vector_list = self.bootstrap_search_vector(query_phrase)

        # Create probe using the vector (this is a simplified approach)
        # In a real implementation, we'd need to extend the query API to accept pre-computed vectors
        words = self.normalize_words(query_phrase)
        probe_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

        # Build guard for chapter filtering
        guard = None
        if chapter_filter:
            guard = {"metadata": {"chapter": chapter_filter}}

        # Query the store
        results = self.store.query(
            probe=json.dumps(probe_data),
            data_type="json",
            top_k=top_k,
            threshold=threshold,
            guard=guard,
        )

        print(f"DEBUG: Query returned {len(results)} raw results for '{query_phrase}'")
        if results:
            print(
                f"DEBUG: First result score: {results[0][1]:.3f}, metadata: {results[0][2]['metadata']}"
            )

        # Enrich results with reconstructed text (from our demo store)
        enriched_results = []
        for data_id, score, data in results:
            # Find the original quote data
            original_quote = None
            for quote in self.quotes_data:
                # Match by metadata (since we don't store text)
                if (
                    quote["chapter"] == data["metadata"]["chapter"]
                    and quote["page"] == data["metadata"]["page"]
                ):
                    original_quote = quote
                    break

            enriched_results.append(
                {
                    "id": data_id,
                    "score": score,
                    "metadata": data["metadata"],
                    "reconstructed_text": original_quote["text"]
                    if original_quote
                    else "Text not found",
                    "search_words": words,
                }
            )

        return enriched_results


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

    # Demo searches
    print("\n🔎 Running Search Demos:")
    print("-" * 30)

    # Demo 1: Exact quote search
    print("\n1. Exact quote search:")
    results = finder.search_quotes("Everything depends upon relative minuteness")
    print(f"   Found {len(results)} results:")
    for result in results:
        print(
            f"   Score: {result['score']:.3f} | Chapter: {result['metadata']['chapter']} | Page: {result['metadata']['page']}"
        )
        print(f"   Text: {result['reconstructed_text'][:80]}...")
        print(f"   Search words: {result['search_words']}")
        print()

    # Demo 2: Fuzzy search
    print("\n2. Fuzzy search (similar phrase):")
    results = finder.search_quotes(
        "depends on relative smallness", threshold=0.0
    )  # Lower threshold for fuzzy
    print(f"   Found {len(results)} results:")
    for result in results:
        print(
            f"   Score: {result['score']:.3f} | Chapter: {result['metadata']['chapter']} | Page: {result['metadata']['page']}"
        )
        print(f"   Text: {result['reconstructed_text'][:80]}...")
        print(f"   Search words: {result['search_words']}")
        print()

    # Demo 3: Partial phrase
    print("\n3. Partial phrase search:")
    results = finder.search_quotes(
        "differential symbol", threshold=0.0
    )  # Lower threshold for partial
    print(f"   Found {len(results)} results:")
    for result in results:
        print(
            f"   Score: {result['score']:.3f} | Chapter: {result['metadata']['chapter']} | Page: {result['metadata']['page']}"
        )
        print(f"   Text: {result['reconstructed_text'][:80]}...")
        print(f"   Search words: {result['search_words']}")
        print()

    # Demo 4: Vector bootstrapping showcase
    print("\n4. Vector bootstrapping demo:")
    phrase = "calculus tricks are easy"
    vector = finder.bootstrap_search_vector(phrase)
    print(f"   Bootstrapped vector for '{phrase}': {len(vector)} dimensions")
    print(f"   Sample values: {vector[:5]}...")

    # Demo 5: Chapter filtering
    print("\n5. Chapter-filtered search:")
    results = finder.search_quotes(
        "integration", threshold=0.0, chapter_filter="Unknown"
    )  # All are Unknown, search for integration
    print(f"   Found {len(results)} results:")
    for result in results:
        print(
            f"   Score: {result['score']:.3f} | Chapter: {result['metadata']['chapter']} | Page: {result['metadata']['page']}"
        )
        print(f"   Text: {result['reconstructed_text'][:80]}...")
        print(f"   Search words: {result['search_words']}")
        print()

    print("\n✅ Demo completed!")
    print("✨ Key features demonstrated:")
    print("   • N-gram encoding for fuzzy subsequence matching")
    print("   • Metadata-only storage (no full text in DB)")
    print("   • Vector bootstrapping API")
    print("   • Chapter filtering with guards")


if __name__ == "__main__":
    main()
