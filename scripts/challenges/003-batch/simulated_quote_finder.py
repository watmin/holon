#!/usr/bin/env python3
"""
Simulated Quote Finder - Demonstrates PDF content indexing and quote location.
Uses synthetic PDF-like content to show the complete workflow.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from holon import CPUStore
from holon.encoder import ListEncodeMode

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulatedPDFIndexer:
    """Simulate indexing PDF content with realistic calculus book content."""

    def __init__(self):
        self.calculus_content = self._generate_synthetic_pdf_content()

    def _generate_synthetic_pdf_content(self) -> List[Dict[str, Any]]:
        """Generate synthetic PDF-like content from a calculus book."""
        # Simulate pages from "Calculus Made Easy" with realistic content
        pages_content = [
            # Page 1 - Title and intro
            "Calculus Made Easy by Silvanus Thompson. Being a very-simplest introduction to those beautiful methods of reckoning which are generally called by the terrifying names of the Differential Calculus and the Integral Calculus.",

            # Page 2 - Introduction
            "Calculus Made Easy. To those who have never heard of it, or who have a vague idea that it is something difficult. What one fool can do, another can. This book will show you how.",

            # Page 3 - Differential symbols
            "In mathematics we write dx for a little bit of x. dy for a little bit of y. d which merely means 'a little bit of.' The differential symbol d.",

            # Page 4 - Integration symbol
            "The integral sign Z which is merely a long S. The sum of. Integration is the reverse of differentiation. Z represents accumulation.",

            # Page 5 - Fundamental theorem
            "dy dx is the slope of the tangent. The derivative measures rate of change. The fundamental theorem connects differentiation and integration.",

            # Page 6 - Functions
            "A function of x. Mathematical functions. Rate of change. Calculus deals with quantities that change. How fast they change.",

            # Page 7 - Limits
            "Everything depends upon relative minuteness. The concept of limits. Infinitesimals. The foundation of calculus.",

            # Page 8 - Maxima and minima
            "Maxima and minima. Finding maximum and minimum values. Critical points. First and second derivatives.",

            # Page 9 - Applications
            "Applications of calculus. Physics problems. Velocity and acceleration. Areas and volumes. Optimization problems.",

            # Page 10 - Complex functions
            "Functions of several variables. Partial derivatives. Chain rule. Gradient. Directional derivatives."
        ]

        # Convert to chunks with coordinates
        chunks = []
        chunk_id = 0

        for page_num, page_text in enumerate(pages_content, 1):
            # Split page into sentence chunks
            sentences = page_text.split('. ')
            current_chunk = ""
            chunk_sentences = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                sentence = sentence.strip() + '.'
                current_chunk += sentence + ' '
                chunk_sentences.append(sentence)

                # Create chunk every 2-3 sentences or when we hit a good breakpoint
                if len(chunk_sentences) >= 2 or len(current_chunk.split()) > 15:
                    chunk_text = current_chunk.strip()

                    chunk = {
                        'id': f"page_{page_num}_chunk_{chunk_id}",
                        'content': chunk_text,
                        'coordinates': {
                            'page': page_num,
                            'chunk_num': chunk_id % 3,  # 3 chunks per page
                            'char_start': 0,  # Simplified
                            'char_end': len(chunk_text),
                            'word_start': 0,
                            'word_count': len(chunk_text.split())
                        },
                        'metadata': {
                            'source': 'simulated_pdf',
                            'chunk_type': 'sentence_group',
                            'sentence_count': len(chunk_sentences)
                        }
                    }

                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk = ""
                    chunk_sentences = []

            # Add remaining content as final chunk
            if current_chunk.strip():
                chunk_text = current_chunk.strip()
                chunk = {
                    'id': f"page_{page_num}_chunk_{chunk_id}",
                    'content': chunk_text,
                    'coordinates': {
                        'page': page_num,
                        'chunk_num': chunk_id % 3,
                        'char_start': 0,
                        'char_end': len(chunk_text),
                        'word_start': 0,
                        'word_count': len(chunk_text.split())
                    },
                    'metadata': {
                        'source': 'simulated_pdf',
                        'chunk_type': 'sentence_group',
                        'sentence_count': len(chunk_sentences) if chunk_sentences else 1
                    }
                }
                chunks.append(chunk)
                chunk_id += 1

        return chunks

    def index_content(self) -> List[Dict[str, Any]]:
        """Return the indexed content chunks."""
        return self.calculus_content

    def create_chunk_unit(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Holon unit for a content chunk."""
        words = self._normalize_text(chunk['content'])

        unit = {
            'text': {
                '_encode_mode': 'ngram',
                'sequence': words
            },
            'coordinates': chunk['coordinates'],
            'metadata': {
                'chunk_id': chunk['id'],
                'word_count': len(words),
                'source': 'simulated_pdf'
            }
        }
        return unit

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        words = [word for word in normalized.split() if word]
        return words


class SimulatedQuoteLocator:
    """Locate quotes within indexed content."""

    def __init__(self, indexed_chunks: List[Dict[str, Any]]):
        self.chunks = indexed_chunks

    def find_quote_locations(self, quote_text: str, min_similarity: float = 0.8) -> List[Dict[str, Any]]:
        """Find locations where a quote appears."""
        quote_lower = quote_text.lower().strip()
        locations = []

        for chunk in self.chunks:
            chunk_text_lower = chunk['content'].lower()

            # Exact match
            if quote_lower in chunk_text_lower:
                start_pos = chunk_text_lower.find(quote_lower)
                word_start = len(chunk_text_lower[:start_pos].split())

                location = {
                    'coordinates': chunk['coordinates'],
                    'similarity': 1.0,
                    'match_type': 'exact',
                    'chunk_content': chunk['content'],
                    'quote_found': quote_text,
                    'position_in_chunk': {
                        'word_start': word_start,
                        'word_end': word_start + len(quote_lower.split()),
                        'char_start': start_pos,
                        'char_end': start_pos + len(quote_lower)
                    }
                }
                locations.append(location)

            # Fuzzy match
            elif min_similarity < 1.0:
                import difflib
                matcher = difflib.SequenceMatcher(None, quote_lower, chunk_text_lower)
                similarity = matcher.ratio()

                if similarity >= min_similarity:
                    location = {
                        'coordinates': chunk['coordinates'],
                        'similarity': similarity,
                        'match_type': 'fuzzy',
                        'chunk_content': chunk['content'],
                        'best_match': self._find_best_match(quote_lower, chunk_text_lower)
                    }
                    locations.append(location)

        return sorted(locations, key=lambda x: (-x['similarity'], x['coordinates']['page']))

    def _find_best_match(self, quote: str, text: str) -> str:
        """Find best substring match."""
        import difflib
        quote_words = quote.split()
        text_words = text.split()

        best_match = ""
        best_ratio = 0

        for window_size in [len(quote_words), len(quote_words)+1, len(quote_words)-1]:
            if window_size <= 0 or window_size > len(text_words):
                continue

            for i in range(len(text_words) - window_size + 1):
                window = ' '.join(text_words[i:i+window_size])
                ratio = difflib.SequenceMatcher(None, quote, window.lower()).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window

        return best_match if best_ratio > 0.5 else "No good match"


class SimulatedQuoteFinderApp:
    """Application for indexing content and finding quote locations."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.indexed_content = []
        self.locator = None

    def ingest_content(self) -> bool:
        """Ingest simulated PDF content."""
        print("📚 Ingesting simulated PDF content...")

        try:
            indexer = SimulatedPDFIndexer()
            chunks = indexer.index_content()

            print(f"💾 Storing {len(chunks)} content chunks...")
            for chunk in chunks:
                unit_data = indexer.create_chunk_unit(chunk)
                chunk_id = self.store.insert(json.dumps(unit_data), 'json')
                chunk['stored_id'] = chunk_id
                self.indexed_content.append(chunk)

            self.locator = SimulatedQuoteLocator(self.indexed_content)

            print(f"✅ Successfully ingested content with {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Failed to ingest content: {e}")
            return False

    def find_quote_coordinates(self, quote_text: str) -> Dict[str, Any]:
        """Find coordinates where a quote appears."""
        if not self.locator:
            return {'error': 'No content ingested yet'}

        print(f"🔍 Finding quote: '{quote_text[:50]}...'")

        # Try exact matching first
        exact_locations = self.locator.find_quote_locations(quote_text, min_similarity=1.0)

        if exact_locations:
            return {
                'quote': quote_text,
                'match_type': 'exact',
                'locations': exact_locations,
                'recommendations': self._generate_recommendations(exact_locations)
            }

        # Try fuzzy matching
        fuzzy_locations = self.locator.find_quote_locations(quote_text, min_similarity=0.6)

        if fuzzy_locations:
            return {
                'quote': quote_text,
                'match_type': 'fuzzy',
                'locations': fuzzy_locations,
                'recommendations': self._generate_recommendations(fuzzy_locations)
            }

        # Vector similarity search
        vector_locations = self._vector_similarity_search(quote_text)

        return {
            'quote': quote_text,
            'match_type': 'vector_similarity',
            'locations': vector_locations,
            'recommendations': self._generate_recommendations(vector_locations)
        }

    def _vector_similarity_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        import re
        words = [w for w in re.sub(r'[^\w\s]', '', query_text.lower()).split() if w]

        probe_data = {
            'text': {
                '_encode_mode': 'ngram',
                'sequence': words
            }
        }

        results = self.store.query(
            probe=json.dumps(probe_data),
            data_type='json',
            top_k=top_k,
            threshold=0.0
        )

        locations = []
        for vector_id, score, stored_data in results:
            chunk = next((c for c in self.indexed_content if c.get('stored_id') == vector_id), None)
            if chunk:
                locations.append({
                    'coordinates': chunk['coordinates'],
                    'similarity': score,
                    'match_type': 'vector_similarity',
                    'chunk_content': chunk['content']
                })

        return locations

    def _generate_recommendations(self, locations: List[Dict[str, Any]]) -> List[str]:
        """Generate manual verification recommendations."""
        if not locations:
            return ["No locations found to check"]

        recommendations = []
        page_groups = {}

        for loc in locations:
            page = loc['coordinates']['page']
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(loc)

        for page, page_locs in sorted(page_groups.items()):
            best_match = max(page_locs, key=lambda x: x.get('similarity', 0))
            sim = best_match.get('similarity', 0)

            confidence = "High" if sim >= 0.9 else "Medium" if sim >= 0.7 else "Low"
            recommendations.append(
                f"Check Page {page}, Chunk {best_match['coordinates']['chunk_num']} "
                f"({confidence} confidence - similarity: {sim:.2f})"
            )

        return recommendations[:3]


def demo_pdf_quote_finder():
    """Demonstrate the PDF quote finder application."""
    print("🎯 PDF Quote Finder - Content Indexing Demo")
    print("=" * 55)

    app = SimulatedQuoteFinderApp()

    # Ingest simulated PDF content
    print("📖 Ingesting simulated calculus book content...")
    success = app.ingest_content()

    if not success:
        print("❌ Failed to ingest content")
        return

    # Show content summary
    print("\n📊 Content Summary:")
    print(f"   Chunks indexed: {len(app.indexed_content)}")
    pages = set(c['coordinates']['page'] for c in app.indexed_content)
    print(f"   Pages covered: {len(pages)} ({min(pages)} - {max(pages)})")

    # Test quote finding
    test_quotes = [
        "calculus made easy",
        "Everything depends upon relative minuteness",
        "dy dx is the slope",
        "differential calculus",
        "integration is the reverse"
    ]

    print("\n🎯 Quote Location Testing:")
    for quote in test_quotes:
        print(f"\n🔍 Query: '{quote}'")
        print("-" * 45)

        result = app.find_quote_coordinates(quote)

        if 'error' in result:
            print(f"   ❌ {result['error']}")
            continue

        print(f"   🎯 Match type: {result['match_type']}")
        print(f"   📍 Found {len(result['locations'])} locations")

        # Show top recommendation
        if result['recommendations']:
            print(f"   💡 {result['recommendations'][0]}")

        # Show top location
        if result['locations']:
            top_loc = result['locations'][0]
            coord = top_loc['coordinates']
            print(f"   📄 Best location: Page {coord['page']}, Chunk {coord['chunk_num']}")
            print(".3f")
            print(f"   📖 Content: {top_loc.get('chunk_content', 'N/A')[:100]}...")

            if top_loc.get('quote_found'):
                print(f"   🎯 Exact match: '{top_loc['quote_found']}'")
        print()

    print("✅ PDF Quote Finder Demo Complete!")
    print("\n🎉 This demonstrates the full workflow:")
    print("   1. 📖 Ingest PDF content into coordinate-addressable chunks")
    print("   2. 🔍 Query for quotes or concepts")
    print("   3. 📍 Get exact coordinates for manual verification")
    print("   4. ✅ Users can check the actual PDF at those locations!")


if __name__ == "__main__":
    demo_pdf_quote_finder()