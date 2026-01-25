#!/usr/bin/env python3
"""
Improved Quote Finder - Intelligent quote location with context awareness.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from difflib import SequenceMatcher

from holon import CPUStore
from holon.encoder import ListEncodeMode

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedQuoteLocator:
    """Intelligent quote location with context awareness."""

    def __init__(self, indexed_chunks: List[Dict[str, Any]]):
        self.chunks = indexed_chunks
        # Group chunks by page for better deduplication
        self.chunks_by_page = {}
        for chunk in indexed_chunks:
            page = chunk['coordinates']['page']
            if page not in self.chunks_by_page:
                self.chunks_by_page[page] = []
            self.chunks_by_page[page].append(chunk)

    def find_quote_locations(self, quote_text: str, min_similarity: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find quote locations with intelligent deduplication and context awareness.

        Args:
            quote_text: The quote to find
            min_similarity: Minimum similarity for fuzzy matches

        Returns:
            List of location results, deduplicated by page
        """
        print(f"🔍 Intelligently locating quote: '{quote_text[:50]}...'")

        quote_lower = quote_text.lower().strip()
        page_locations = {}  # page -> best_location

        # Search each page
        for page_num, page_chunks in self.chunks_by_page.items():
            page_results = []

            for chunk in page_chunks:
                chunk_text_lower = chunk['content'].lower()

                # Check for exact substring match
                if quote_lower in chunk_text_lower:
                    # Calculate context score (how well the quote fits as a unit)
                    context_score = self._calculate_context_score(quote_lower, chunk_text_lower)

                    location = {
                        'coordinates': chunk['coordinates'],
                        'similarity': 1.0,
                        'context_score': context_score,
                        'match_type': 'exact_substring',
                        'chunk_content': chunk['content'],
                        'quote_found': quote_text,
                        'page_rank': self._calculate_page_importance(page_num)
                    }
                    page_results.append(location)

                # Check for fuzzy matches
                elif min_similarity < 1.0:
                    # Use sequence matcher for better fuzzy matching
                    matcher = SequenceMatcher(None, quote_lower, chunk_text_lower)
                    similarity = matcher.ratio()

                    if similarity >= min_similarity:
                        # Find best matching substring
                        best_match = self._find_best_substring_match(quote_lower, chunk_text_lower, similarity)

                        location = {
                            'coordinates': chunk['coordinates'],
                            'similarity': similarity,
                            'context_score': self._calculate_context_score(best_match, chunk_text_lower),
                            'match_type': 'fuzzy',
                            'chunk_content': chunk['content'],
                            'best_match': best_match,
                            'page_rank': self._calculate_page_importance(page_num)
                        }
                        page_results.append(location)

            # For each page, keep only the best result
            if page_results:
                # Sort by context_score, then similarity, then page_rank
                page_results.sort(key=lambda x: (-x.get('context_score', 0), -x['similarity'], -x.get('page_rank', 0)))
                page_locations[page_num] = page_results[0]

        # Convert to final results list, sorted by quality
        results = list(page_locations.values())
        results.sort(key=lambda x: (-x.get('context_score', 0), -x['similarity'], -x.get('page_rank', 0)))

        print(f"✅ Found {len(results)} unique page locations for quote")
        return results

    def _calculate_context_score(self, quote: str, context: str) -> float:
        """
        Calculate how well the quote fits as a coherent unit in context.
        Higher scores for quotes that appear as complete thoughts.
        """
        score = 0.0

        # Check if quote appears as a complete phrase (bounded by punctuation or spaces)
        quote_pos = context.find(quote)
        if quote_pos >= 0:
            # Check preceding character
            if quote_pos == 0 or context[quote_pos - 1] in ' .,!?;:()[]{}':
                score += 0.3

            # Check following character
            end_pos = quote_pos + len(quote)
            if end_pos == len(context) or context[end_pos] in ' .,!?;:()[]{}':
                score += 0.3

            # Bonus for quotes that are substantial portion of chunk
            quote_ratio = len(quote) / len(context)
            if quote_ratio > 0.1:  # Quote is >10% of chunk
                score += 0.2

            # Bonus for quotes near beginning of chunk (more prominent)
            position_ratio = quote_pos / len(context)
            if position_ratio < 0.3:  # In first 30% of chunk
                score += 0.2

        return min(1.0, score)  # Cap at 1.0

    def _calculate_page_importance(self, page_num: int) -> float:
        """
        Calculate page importance score.
        Earlier pages (introduction, main content) are more important.
        """
        if page_num <= 5:
            return 1.0  # Introduction/title pages
        elif page_num <= 20:
            return 0.8  # Main content
        elif page_num <= 50:
            return 0.6  # Core content
        else:
            return 0.4  # Later content (appendices, etc.)

    def _find_best_substring_match(self, quote: str, text: str, min_similarity: float) -> str:
        """Find the best substring match above similarity threshold."""
        quote_words = quote.split()
        text_words = text.split()

        best_match = ""
        best_score = 0

        # Try different window sizes
        for window_size in range(len(quote_words), len(quote_words) + 3):
            if window_size > len(text_words):
                continue

            for i in range(len(text_words) - window_size + 1):
                window = ' '.join(text_words[i:i+window_size])
                similarity = SequenceMatcher(None, quote, window.lower()).ratio()

                if similarity > best_score and similarity >= min_similarity:
                    best_score = similarity
                    best_match = window

        return best_match if best_score >= min_similarity else "No good match found"


class IntelligentQuoteFinder:
    """Complete intelligent quote finding system."""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.locator = None

    def ingest_pdf_content(self, pdf_index_file: str) -> bool:
        """Ingest pre-indexed PDF content."""
        try:
            with open(pdf_index_file, 'r') as f:
                data = json.load(f)

            indexed_chunks = data['chunks']
            print(f"📚 Loading {len(indexed_chunks)} indexed chunks")

            # Store chunks in vector database
            for chunk in indexed_chunks:
                unit_data = self._create_chunk_unit(chunk)
                chunk_id = self.store.insert(json.dumps(unit_data), 'json')
                chunk['stored_id'] = chunk_id

            # Initialize locator
            self.locator = ImprovedQuoteLocator(indexed_chunks)

            print(f"✅ Successfully loaded PDF content with {len(indexed_chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Failed to load PDF content: {e}")
            return False

    def find_quote(self, quote_text: str, search_strategy: str = 'intelligent') -> Dict[str, Any]:
        """
        Find a quote using specified strategy.

        Args:
            quote_text: The quote to find
            search_strategy: 'intelligent', 'exact', or 'fuzzy'

        Returns:
            Results dictionary with locations and metadata
        """
        if not self.locator:
            return {'error': 'No PDF content loaded. Use ingest_pdf_content() first.'}

        if search_strategy == 'intelligent':
            locations = self.locator.find_quote_locations(quote_text, min_similarity=0.7)
        elif search_strategy == 'exact':
            locations = self.locator.find_quote_locations(quote_text, min_similarity=1.0)
        else:  # fuzzy
            locations = self.locator.find_quote_locations(quote_text, min_similarity=0.6)

        # Generate recommendations
        recommendations = self._generate_recommendations(locations, quote_text)

        return {
            'quote': quote_text,
            'search_strategy': search_strategy,
            'locations_found': len(locations),
            'locations': locations[:10],  # Top 10 results
            'recommendations': recommendations,
            'summary': f"Found {len(locations)} unique page locations for '{quote_text[:30]}...'"
        }

    def _create_chunk_unit(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
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
                'source': 'pdf_index'
            }
        }
        return unit

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def _generate_recommendations(self, locations: List[Dict[str, Any]], quote: str) -> List[str]:
        """Generate human-readable recommendations."""
        if not locations:
            return ["No quote locations found. Try different wording or check spelling."]

        recommendations = []
        top_locations = locations[:5]  # Top 5

        for i, loc in enumerate(top_locations, 1):
            coord = loc['coordinates']
            context_score = loc.get('context_score', 0)
            similarity = loc.get('similarity', 0)

            confidence = "High" if context_score > 0.5 else "Medium" if similarity > 0.8 else "Low"

            rec = f"Check Page {coord['page']}, Chunk {coord['chunk_num']} ({confidence} confidence)"
            recommendations.append(rec)

        if len(locations) > 5:
            recommendations.append(f"... and {len(locations) - 5} more locations found")

        return recommendations


def demonstrate_intelligent_quote_finding():
    """Demonstrate the improved quote finding system."""
    print("🎯 Intelligent Quote Finder - Context-Aware Location")
    print("=" * 60)

    finder = IntelligentQuoteFinder()

    # Load the indexed PDF content
    index_file = Path(__file__).parent / "pdf_content_index.json"
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        print("Run pdf_content_indexer.py first to create the index")
        return

    success = finder.ingest_pdf_content(str(index_file))
    if not success:
        return

    # Test various quote finding scenarios
    test_quotes = [
        {
            'text': 'calculus made easy',
            'description': 'Book title - appears in many places'
        },
        {
            'text': 'Everything depends upon relative minuteness',
            'description': 'Famous quote from the book'
        },
        {
            'text': 'differential calculus',
            'description': 'Technical term'
        },
        {
            'text': 'slope of the tangent',
            'description': 'Mathematical concept'
        }
    ]

    for test_case in test_quotes:
        print(f"\n🔍 Finding: '{test_case['text']}'")
        print(f"   {test_case['description']}")
        print("-" * 50)

        result = finder.find_quote(test_case['text'], search_strategy='intelligent')

        if 'error' in result:
            print(f"   ❌ {result['error']}")
            continue

        print(f"   📊 Strategy: {result['search_strategy']}")
        print(f"   📍 Locations found: {result['locations_found']}")

        # Show top recommendations
        if result['recommendations']:
            print("   💡 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"      • {rec}")

        # Show detailed results for top match
        if result['locations']:
            top_loc = result['locations'][0]
            coord = top_loc['coordinates']
            print("\n   🏆 Top Result:")
            print(f"      Page: {coord['page']}, Chunk: {coord['chunk_num']}")
            print(".3f")
            print(".3f")
            print(f"      Match type: {top_loc.get('match_type', 'unknown')}")

            if 'quote_found' in top_loc:
                print(f"      Found: '{top_loc['quote_found']}'")
            elif 'best_match' in top_loc:
                print(f"      Best match: '{top_loc['best_match'][:50]}...'")

        print()

    print("🎉 Intelligent Quote Finding Complete!")
    print("\n✨ Key Improvements:")
    print("   • Context-aware scoring (phrase boundaries, position)")
    print("   • Page-level deduplication (one result per page)")
    print("   • Intelligent ranking (context > similarity > page importance)")
    print("   • Meaningful recommendations for manual verification")
    print("   • Handles book titles, quotes, and technical terms appropriately")


if __name__ == "__main__":
    demonstrate_intelligent_quote_finding()