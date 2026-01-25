#!/usr/bin/env python3
"""
Quote Finder App - Ingest PDFs and locate quotes with coordinates.
"""

import json

# Configure logging
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf_content_indexer import PDFContentIndexer, PDFQuoteLocator

from holon import CPUStore
from holon.encoder import ListEncodeMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuoteFinderApp:
    """Application for indexing PDFs and finding quote locations."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.indexed_content = []
        self.locator = None

    def ingest_pdf(self, pdf_path: str) -> bool:
        """Ingest a PDF and index its content with coordinates."""
        print(f"📚 Ingesting PDF: {pdf_path}")

        chunks = []
        try:
            # Index the PDF content
            indexer = PDFContentIndexer(pdf_path)
            chunks = indexer.index_pdf_content()

            if not chunks:
                print("❌ No content extracted from PDF")
                return False

            # Store chunks in Holon for vector search
            print(f"💾 Storing {len(chunks)} content chunks...")
            chunk_ids = []

            for chunk in chunks:
                unit_data = indexer.create_chunk_unit(chunk)
                chunk_id = self.store.insert(json.dumps(unit_data), "json")
                chunk_ids.append(chunk_id)

                # Keep track of content for location lookup
                chunk["stored_id"] = chunk_id
                self.indexed_content.append(chunk)

            # Create locator for quote finding
            self.locator = PDFQuoteLocator(self.indexed_content)

            print(
                f"✅ Successfully ingested PDF with {len(self.indexed_content)} chunks"
            )
            return True

        except Exception as e:
            print(f"❌ Failed to ingest PDF: {e}")
            return False

    def find_quote_coordinates(
        self, quote_text: str, search_similar: bool = True, min_similarity: float = 0.7
    ) -> Dict[str, Any]:
        """
        Find coordinates where a quote appears in the indexed PDF.

        Args:
            quote_text: The quote to find
            search_similar: Whether to use vector similarity search
            min_similarity: Minimum similarity for fuzzy matches

        Returns:
            Dict with locations and recommendations
        """
        if not self.locator:
            return {
                "error": "No PDF ingested yet. Use ingest_pdf() first.",
                "locations": [],
            }

        print(f"🔍 Finding quote: '{quote_text[:60]}...'")

        # First try exact location finding
        exact_locations = self.locator.find_quote_locations(
            quote_text, min_similarity=1.0
        )

        if exact_locations:
            print(f"🎯 Found {len(exact_locations)} exact matches")
            return {
                "quote": quote_text,
                "match_type": "exact",
                "locations": exact_locations,
                "recommendations": self._generate_recommendations(
                    exact_locations, quote_text
                ),
            }

        # If no exact matches, try fuzzy matching
        if search_similar:
            print("🔄 No exact matches, trying fuzzy search...")
            fuzzy_locations = self.locator.find_quote_locations(
                quote_text, min_similarity=min_similarity
            )

            if fuzzy_locations:
                print(f"📊 Found {len(fuzzy_locations)} fuzzy matches")
                return {
                    "quote": quote_text,
                    "match_type": "fuzzy",
                    "locations": fuzzy_locations,
                    "recommendations": self._generate_recommendations(
                        fuzzy_locations, quote_text
                    ),
                }

        # If still no matches, try vector similarity search
        print("🔄 Trying vector similarity search...")
        vector_locations = self._vector_similarity_search(quote_text)

        return {
            "quote": quote_text,
            "match_type": "vector_similarity",
            "locations": vector_locations,
            "recommendations": self._generate_recommendations(
                vector_locations, quote_text
            ),
        }

    def _vector_similarity_search(
        self, query_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Use vector similarity to find related content chunks."""
        # Encode query
        words = self._normalize_text(query_text)
        probe_data = {"text": {"_encode_mode": "ngram", "sequence": words}}

        # Search for similar chunks
        results = self.store.query(
            probe=json.dumps(probe_data), data_type="json", top_k=top_k, threshold=0.0
        )

        # Convert to location format
        locations = []
        for vector_id, score, stored_data in results:
            # Find the corresponding chunk
            chunk = None
            for c in self.indexed_content:
                if c.get("stored_id") == vector_id:
                    chunk = c
                    break

            if chunk:
                location = {
                    "coordinates": chunk["coordinates"],
                    "similarity": score,
                    "match_type": "vector_similarity",
                    "chunk_content": chunk["content"],
                    "chunk_id": chunk["id"],
                }
                locations.append(location)

        return locations

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re

        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def _generate_recommendations(
        self, locations: List[Dict[str, Any]], quote: str
    ) -> List[str]:
        """Generate recommendations for manual verification."""
        if not locations:
            return ["No locations found to check"]

        recommendations = []

        # Group by page
        page_groups = {}
        for loc in locations:
            page = loc["coordinates"]["page"]
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(loc)

        # Generate recommendations
        for page, page_locs in sorted(page_groups.items()):
            best_match = max(page_locs, key=lambda x: x.get("similarity", 0))
            sim = best_match.get("similarity", 0)

            if sim >= 0.9:
                confidence = "High confidence"
            elif sim >= 0.7:
                confidence = "Medium confidence"
            else:
                confidence = "Low confidence - manual check recommended"

            recommendations.append(
                f"Check Page {page}, Chunk {best_match['coordinates']['chunk_num']} "
                f"({confidence} - similarity: {sim:.2f})"
            )

        return recommendations[:5]  # Top 5 recommendations

    def get_pdf_summary(self) -> Dict[str, Any]:
        """Get summary of indexed PDF content."""
        if not self.indexed_content:
            return {"error": "No PDF indexed yet"}

        pages = set(c["coordinates"]["page"] for c in self.indexed_content)
        total_words = sum(c["coordinates"]["word_count"] for c in self.indexed_content)

        return {
            "total_chunks": len(self.indexed_content),
            "pages_indexed": len(pages),
            "page_range": f"{min(pages)} - {max(pages)}",
            "total_words": total_words,
            "avg_chunk_size": total_words // len(self.indexed_content)
            if self.indexed_content
            else 0,
        }


def demo_quote_finder():
    """Demonstrate the quote finder application."""
    print("🎯 PDF Quote Finder Application Demo")
    print("=" * 50)

    app = QuoteFinderApp()

    # Find the PDF
    project_root = Path(__file__).parent.parent.parent.parent
    pdf_path = (
        project_root / "docs" / "challenges" / "003-batch" / "calculus-made-easy.pdf"
    )

    if not pdf_path.exists():
        print(f"❌ PDF not found at: {pdf_path}")
        print("Cannot run demo without PDF file")
        return

    # Ingest the PDF
    print("📖 Ingesting PDF content...")
    success = app.ingest_pdf(str(pdf_path))

    if not success:
        print("❌ Failed to ingest PDF")
        return

    # Show PDF summary
    summary = app.get_pdf_summary()
    print("\n📊 PDF Summary:")
    print(f"   Chunks indexed: {summary['total_chunks']}")
    print(f"   Pages covered: {summary['pages_indexed']} ({summary['page_range']})")
    print(f"   Total words: {summary['total_words']}")
    print(f"   Avg chunk size: {summary['avg_chunk_size']} words")

    # Test quote finding
    test_quotes = [
        "calculus made easy",
        "differential calculus",
        "function",
        "mathematical methods",
        "rate of change",
    ]

    print("\n🎯 Quote Location Testing:")
    for quote in test_quotes:
        print(f"\n🔍 Searching for: '{quote}'")
        print("-" * 40)

        result = app.find_quote_coordinates(
            quote, search_similar=True, min_similarity=0.6
        )

        if "error" in result:
            print(f"   ❌ {result['error']}")
            continue

        print(f"   🎯 Match type: {result['match_type']}")
        print(f"   📍 Found {len(result['locations'])} locations")

        # Show top recommendation
        if result["recommendations"]:
            print(f"   💡 Top recommendation: {result['recommendations'][0]}")

        # Show top location details
        if result["locations"]:
            top_loc = result["locations"][0]
            coord = top_loc["coordinates"]
            print(
                f"   📄 Best match at: Page {coord['page']}, Chunk {coord['chunk_num']}"
            )
            print(f"   🎯 Similarity: {top_loc.get('similarity', 0):.3f}")
            print(
                f"   📖 Content preview: {top_loc.get('chunk_content', 'N/A')[:100]}..."
            )

        print()

    print("✅ Quote Finder Demo Complete!")
    print("\n🎉 Users can now:")
    print("   • Ingest any PDF document")
    print("   • Query for quotes or concepts")
    print("   • Get exact coordinates for manual verification")
    print("   • Use similarity search for related content")


if __name__ == "__main__":
    demo_quote_finder()
