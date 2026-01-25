#!/usr/bin/env python3
"""
Paragraph-Based Quote Finder
Uses coordinate system where paragraphs are base units and quotes are located within them.
"""

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from holon import CPUStore
from holon.encoder import ListEncodeMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParagraphQuoteFinder:
    """Quote finder using paragraph-based coordinate system."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.paragraphs = []  # Store paragraph data for coordinate lookup

    def load_processed_paragraphs(self, json_file: str) -> List[Dict[str, Any]]:
        """Load processed paragraphs with coordinates."""
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paragraphs = data["paragraphs"]
        print(f"📚 Loaded {len(paragraphs)} paragraphs with coordinate system")
        return paragraphs

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re

        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def create_paragraph_unit(self, paragraph: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Holon unit for a paragraph with coordinate metadata."""
        words = self._normalize_text(paragraph["text"])

        unit = {
            "text": {
                "_encode_mode": "ngram",  # N-gram encoding for fuzzy text matching
                "sequence": words,
            },
            "coordinates": paragraph["coordinates"],
            "metadata": {
                "book_title": paragraph["book_title"],
                "id": paragraph["id"],
                "word_count": len(words),
            },
        }
        return unit

    def ingest_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> List[str]:
        """Ingest paragraphs into Holon store with coordinate system."""
        print(f"💾 Ingesting {len(paragraphs)} paragraphs into coordinate system...")

        units_data = []
        for paragraph in paragraphs:
            unit_data = self.create_paragraph_unit(paragraph)
            units_data.append(json.dumps(unit_data))

            # Keep paragraph data for coordinate lookup
            self.paragraphs.append(paragraph)

        # Batch insert for efficiency
        ids = self.store.batch_insert(units_data, data_type="json")
        print(f"✅ Successfully ingested {len(ids)} paragraph units")
        return ids

    def find_quote_in_paragraphs(self, quote_text: str) -> List[Dict[str, Any]]:
        """Find which paragraphs contain a given quote and return coordinates."""
        quote_lower = quote_text.lower().strip()
        matches = []

        for paragraph in self.paragraphs:
            para_text_lower = paragraph["text"].lower()

            # Check if quote appears in paragraph
            if quote_lower in para_text_lower:
                # Calculate similarity score using difflib for fuzzy matching
                matcher = SequenceMatcher(None, quote_lower, para_text_lower)
                similarity = matcher.ratio()

                # Find the position in the paragraph
                start_pos = para_text_lower.find(quote_lower)
                if start_pos != -1:
                    # Calculate word position in paragraph
                    words_before = len(para_text_lower[:start_pos].split())
                    quote_words = len(quote_lower.split())

                    match = {
                        "paragraph_id": paragraph["id"],
                        "coordinates": paragraph["coordinates"],
                        "similarity": similarity,
                        "quote_position": {
                            "word_start": words_before,
                            "word_end": words_before + quote_words,
                            "char_start": start_pos,
                            "char_end": start_pos + len(quote_lower),
                        },
                        "paragraph_text": paragraph["text"],
                        "matched_quote": quote_text,
                    }
                    matches.append(match)

        return sorted(matches, key=lambda x: x["similarity"], reverse=True)

    def search_quotes_by_content(
        self, query_text: str, top_k: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for quotes by content, returning paragraph coordinates."""
        print(f"🔍 Searching for quote content: '{query_text}'")

        # First, try exact quote matching in our paragraph database
        exact_matches = self.find_quote_in_paragraphs(query_text)

        if exact_matches:
            print(f"📍 Found {len(exact_matches)} exact matches in paragraphs")
            return exact_matches[:top_k]

        # If no exact matches, use vector similarity search on paragraphs
        print("🔄 No exact matches, using vector similarity search...")

        words = self._normalize_text(query_text)
        probe_data = {"text": {"_encode_mode": "ngram", "sequence": words}}

        # Query the store
        results = self.store.query(
            probe=json.dumps(probe_data),
            data_type="json",
            top_k=top_k,
            threshold=threshold,
        )

        print(f"📊 Vector search returned {len(results)} paragraph matches")

        # Enrich results with paragraph data and coordinate information
        enriched_results = []
        for data_id, score, data in results:
            # Find the corresponding paragraph
            paragraph = None
            for p in self.paragraphs:
                if p["id"] == data["metadata"]["id"]:
                    paragraph = p
                    break

            if paragraph:
                # Check if the query appears in this paragraph
                quote_matches = self.find_quote_in_paragraphs(query_text)
                relevant_matches = [
                    m for m in quote_matches if m["paragraph_id"] == paragraph["id"]
                ]

                result = {
                    "paragraph_id": paragraph["id"],
                    "coordinates": data["coordinates"],
                    "similarity_score": score,
                    "paragraph_text": paragraph["text"],
                    "quote_matches": relevant_matches,
                    "search_query": query_text,
                }
                enriched_results.append(result)

        return enriched_results

    def demonstrate_coordinate_system(self):
        """Demonstrate the paragraph-based coordinate system."""
        print("🏗️ Paragraph-Based Quote Finder - Coordinate System Demo")
        print("=" * 60)

        # Load processed paragraphs
        paragraphs_file = Path(__file__).parent / "processed_paragraphs.json"
        paragraphs = self.load_processed_paragraphs(str(paragraphs_file))

        # Show coordinate system
        print("\n📍 Coordinate System Examples:")
        for i, p in enumerate(paragraphs[:5]):
            coord = p["coordinates"]
            print(
                f"   {p['id']}: {coord['chapter']} | Para {coord['paragraph_num']} | Pages {coord['page_start']}-{coord['page_end']}"
            )
            print(f"      Text: {p['text'][:80]}...")
            if p["contains_quotes"]:
                print(f"      Contains quotes: {p['contains_quotes']}")
            print()

        # Ingest paragraphs
        ids = self.ingest_paragraphs(paragraphs)

        # Test coordinate-based quote finding
        test_quotes = [
            "Everything depends upon relative minuteness.",
            "Integration is the reverse of differentiation.",
            "dy dx is the slope of the tangent.",
            "some calculus tricks",  # Partial/fuzzy match
        ]

        for quote in test_quotes:
            print(f"\n🔍 Finding Quote: '{quote}'")
            print("-" * 50)

            matches = self.search_quotes_by_content(quote, threshold=0.0)

            if matches:
                print(f"   ✅ Found in {len(matches)} paragraph(s):")
                for i, match in enumerate(matches[:3]):  # Show top 3
                    coord = match["coordinates"]
                    print(f"   {i+1}. {match['paragraph_id']}")
                    print(
                        f"      📍 Coordinate: {coord['chapter']} | Para {coord['paragraph_num']} | Page {coord['page_start']}"
                    )
                    print(".3f")
                    print(f"      📄 Paragraph: {match['paragraph_text'][:100]}...")

                    if match.get("quote_matches"):
                        for qm in match["quote_matches"]:
                            print(
                                f"      🎯 Quote found at word positions {qm['quote_position']['word_start']}-{qm['quote_position']['word_end']}"
                            )
                    print()
            else:
                print("   ❌ No matches found")
        print("\n🎯 Coordinate System Benefits:")
        print("   • Each quote has a definite coordinate (chapter.paragraph.page)")
        print("   • Paragraphs are the stable base units for vector encoding")
        print("   • Quotes are located within paragraph coordinate space")
        print("   • Supports fuzzy matching and partial quote finding")
        print("   • More realistic document structure modeling")


def main():
    """Run the paragraph-based quote finder demonstration."""
    finder = ParagraphQuoteFinder(dimensions=16000)
    finder.demonstrate_coordinate_system()


if __name__ == "__main__":
    main()
