#!/usr/bin/env python3
"""
PDF Substring Matching Demo - Ingest paragraphs, query phrases, find locations.
Demonstrates multi-resolution n-gram encoding for PDF content analysis.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

from holon import CPUStore, HolonClient


class PDFContentAnalyzer:
    """Analyze PDF content with substring matching and location tracking."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.content_data = []
        self.id_to_content = {}

    def normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def ingest_pdf_paragraph(self, paragraph_data: Dict[str, Any]) -> str:
        """
        Ingest a PDF paragraph with location metadata.

        Args:
            paragraph_data: {
                "text": "Full paragraph text...",
                "chapter": "Chapter 1",
                "page": 5,
                "paragraph_num": 3,
                "bounding_box": [x1, y1, x2, y2]
            }
        """
        # Multi-resolution encoding for robust substring matching
        unit_data = {
            "content": {
                "_encode_mode": "ngram",
                "_encode_config": {
                    "n_sizes": [1, 2, 3],       # Individual + bigrams + trigrams
                    "weights": [0.2, 0.6, 0.4], # Weight local patterns higher
                    "length_penalty": True,     # Normalize for query fairness
                    "term_weighting": True,     # Weight important terms
                    "positional_weighting": True, # Earlier words more important
                    "discrimination_boost": True  # Enhance distinctive features
                },
                "sequence": self.normalize_text(paragraph_data["text"])
            },
            "metadata": {
                "chapter": paragraph_data["chapter"],
                "page": paragraph_data["page"],
                "paragraph_num": paragraph_data["paragraph_num"],
                "bounding_box": paragraph_data.get("bounding_box"),
                "full_text": paragraph_data["text"]
            }
        }

        content_id = self.client.insert_json(unit_data)
        self.content_data.append(paragraph_data)
        self.id_to_content[content_id] = paragraph_data

        return content_id

    def find_phrase_locations(self, query_phrase: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find where a phrase appears in the PDF content.

        Args:
            query_phrase: The phrase to search for
            top_k: Number of top matches to return

        Returns:
            List of location results with similarity scores and metadata
        """
        # Use same multi-resolution encoding for query
        probe_data = {
            "content": {
                "_encode_mode": "ngram",
                "_encode_config": {
                    "n_sizes": [1, 2, 3],       # Match document encoding
                    "weights": [0.2, 0.6, 0.4], # Same weighting
                    "length_penalty": True,     # Fair query matching
                    "term_weighting": True,
                    "positional_weighting": True,
                    "discrimination_boost": True
                },
                "sequence": self.normalize_text(query_phrase)
            }
        }

        # Geometric search finds paragraphs with similar multi-resolution patterns
        results = self.client.search_json(
            probe=probe_data,
            top_k=top_k,
            threshold=0.0  # Show all potential matches
        )

        # Format results with location metadata
        locations = []
        for result in results:
            content_id = result["id"]
            similarity = result["score"]
            original_content = self.id_to_content.get(content_id)

            if original_content:
                location = {
                    "similarity": similarity,
                    "chapter": original_content["chapter"],
                    "page": original_content["page"],
                    "paragraph_num": original_content["paragraph_num"],
                    "bounding_box": original_content.get("bounding_box"),
                    "matched_text": original_content["text"][:100] + "..." if len(original_content["text"]) > 100 else original_content["text"],
                    "query_phrase": query_phrase,
                    "confidence": self._calculate_confidence(similarity, query_phrase, original_content["text"])
                }
                locations.append(location)

        return sorted(locations, key=lambda x: x["similarity"], reverse=True)

    def _calculate_confidence(self, similarity: float, query: str, text: str) -> str:
        """Calculate human-readable confidence level."""
        query_lower = query.lower()
        text_lower = text.lower()

        # Exact substring match = highest confidence
        if query_lower in text_lower:
            return "High (exact match)"

        # High similarity with shared words = medium confidence
        query_words = set(self.normalize_text(query))
        text_words = set(self.normalize_text(text))
        overlap = len(query_words.intersection(text_words))
        overlap_ratio = overlap / len(query_words) if query_words else 0

        if similarity > 0.8 and overlap_ratio > 0.5:
            return "Medium (strong pattern match)"
        elif similarity > 0.6:
            return "Low (weak pattern match)"
        else:
            return "Very Low (minimal similarity)"


def demo_pdf_substring_matching():
    """Demonstrate PDF paragraph ingestion and substring querying."""
    print("📄 PDF Substring Matching Demo")
    print("=" * 50)

    analyzer = PDFContentAnalyzer()

    # Sample PDF paragraphs with location metadata
    sample_paragraphs = [
        {
            "text": "Calculus made easy is a classic mathematics textbook that explains differential and integral calculus using simple language and practical examples.",
            "chapter": "Introduction",
            "page": 1,
            "paragraph_num": 1,
            "bounding_box": [50, 100, 500, 150]
        },
        {
            "text": "The fundamental theorem of calculus connects differentiation and integration, showing that they are inverse operations of each other.",
            "chapter": "Chapter 1",
            "page": 15,
            "paragraph_num": 2,
            "bounding_box": [50, 200, 500, 250]
        },
        {
            "text": "Integration is the reverse of differentiation, meaning that finding the integral of a function is like undoing the derivative operation.",
            "chapter": "Chapter 2",
            "page": 45,
            "paragraph_num": 1,
            "bounding_box": [50, 300, 500, 350]
        },
        {
            "text": "When solving calculus problems, it's important to understand the relationship between the slope of a tangent line and the derivative of a function.",
            "chapter": "Chapter 3",
            "page": 78,
            "paragraph_num": 3,
            "bounding_box": [50, 400, 500, 450]
        }
    ]

    # Ingest paragraphs
    print("📝 Ingesting PDF paragraphs with multi-resolution encoding...")
    for i, para in enumerate(sample_paragraphs):
        content_id = analyzer.ingest_pdf_paragraph(para)
        print(f"  ✅ Paragraph {i+1}: {para['chapter']} - Page {para['page']} → {content_id}")

    # Test various queries
    test_queries = [
        "calculus made easy",       # Exact phrase from intro
        "reverse of differentiation", # Partial phrase match
        "fundamental theorem",      # Two-word phrase
        "slope of tangent",         # Related concept
        "quantum physics",          # Unrelated (should not match)
    ]

    print("\n🔍 Querying for phrases and finding locations:")
    print("-" * 50)

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        locations = analyzer.find_phrase_locations(query, top_k=3)

        if locations:
            for i, loc in enumerate(locations[:3], 1):
                print(f"  {i}. 📍 {loc['chapter']} - Page {loc['page']} - Para {loc['paragraph_num']}")
                print(".3f")
                print(f"     💡 {loc['confidence']}")
        else:
            print("  ❌ No matches found")

    print("\n🎯 Multi-Resolution Magic:")
    print("  ✅ n=1: Individual word matching")
    print("  ✅ n=2: Bigram pattern recognition")
    print("  ✅ n=3: Trigram context understanding")
    print("  ✅ Combined: Robust substring matching across different scales!")

    print("\n📊 Why This Works for PDFs:")
    print("  • Ingest full paragraphs with location metadata")
    print("  • Query partial phrases at any granularity")
    print("  • Get precise locations (chapter, page, paragraph, coordinates)")
    print("  • Multi-resolution encoding handles fuzzy/partial matches")
    print("  • Pure geometric - no traditional text search fallbacks!")


if __name__ == "__main__":
    demo_pdf_substring_matching()
