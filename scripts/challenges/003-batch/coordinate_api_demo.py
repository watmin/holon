#!/usr/bin/env python3
"""
Coordinate API Demo
Demonstrates the concrete response structure for coordinate-based quote queries.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# Import our paragraph finder
from paragraph_quote_finder import ParagraphQuoteFinder


def demonstrate_concrete_response_structure():
    """Show the exact response structure for coordinate-based queries."""

    print("🎯 Coordinate-Based Quote Finder - Concrete Response Demo")
    print("=" * 65)

    # Initialize the finder
    finder = ParagraphQuoteFinder(dimensions=16000)

    # Load and ingest paragraphs
    paragraphs_file = Path(__file__).parent / "processed_paragraphs.json"
    paragraphs = finder.load_processed_paragraphs(str(paragraphs_file))
    finder.ingest_paragraphs(paragraphs)

    # Test different query types
    queries = [
        {
            "name": "Exact Quote Lookup",
            "query": "Everything depends upon relative minuteness.",
            "description": "Find exact quote with coordinate",
        },
        {
            "name": "Fuzzy Concept Search",
            "query": "slope of the tangent",
            "description": "Find related mathematical concepts",
        },
        {
            "name": "Partial Quote Match",
            "query": "integration is the reverse",
            "description": "Partial phrase matching",
        },
    ]

    for query_info in queries:
        print(f"\n🔍 {query_info['name']}")
        print(f"   {query_info['description']}")
        print("-" * 50)

        # Perform the search
        results = finder.search_quotes_by_content(query_info["query"], threshold=0.0)

        if results:
            print("   ✅ Query Response Structure:")
            print("   {")
            print("     'results': [")

            for i, result in enumerate(results[:2]):  # Show first 2 results
                comma = "," if i < len(results[:2]) - 1 else ""
                print("       {")
                print(f"         'paragraph_id': '{result['paragraph_id']}',")
                print("         'coordinates': {")
                coord = result["coordinates"]
                print(f"           'chapter': '{coord['chapter']}',")
                print(f"           'paragraph_num': {coord['paragraph_num']},")
                print(f"           'page_start': {coord['page_start']},")
                print(f"           'page_end': {coord['page_end']},")
                print(f"           'word_count': {coord['word_count']}")
                print("         },")
                print(".3f")
                print(
                    f"         'paragraph_text': '{result['paragraph_text'][:60]}...',"
                )

                if result.get("quote_matches"):
                    print("         'quote_matches': [")
                    for j, qm in enumerate(result["quote_matches"][:1]):
                        print("           {")
                        print(f"             'similarity': {qm['similarity']:.3f},")
                        print("             'quote_position': {")
                        pos = qm["quote_position"]
                        print(f"               'word_start': {pos['word_start']},")
                        print(f"               'word_end': {pos['word_end']},")
                        print(f"               'char_start': {pos['char_start']},")
                        print(f"               'char_end': {pos['char_end']}")
                        print("             },")
                        print(f"             'matched_quote': '{qm['matched_quote']}'")
                        print("           }")
                    print("         ],")
                else:
                    print("         'quote_matches': []")

                print(f"         'search_query': '{query_info['query']}'")
                print(f"       }}{comma}")

            print("     ],")
            print("     'query': {")
            print(f"       'text': '{query_info['query']}',")
            print(f"       'type': '{query_info['name']}',")
            print("       'timestamp': '2024-01-01T12:00:00Z'")
            print("     }")
            print("   }")

            print("\n📊 Summary:")
            print(f"   • Found {len(results)} paragraph matches")
            print(
                f"   • Top result coordinate: {results[0]['coordinates']['chapter']} | Para {results[0]['coordinates']['paragraph_num']} | Page {results[0]['coordinates']['page_start']}"
            )
            print(".3f")
        else:
            print("   ❌ No results found")

    print("\n🌐 HTTP API Structure:")
    print("   POST /query")
    print("   {")
    print(
        '     \'probe\': \'{"text": {"_encode_mode": "ngram", "sequence": ["integration", "is", "the", "reverse"]}}\','
    )
    print("     'data_type': 'json',")
    print("     'top_k': 10,")
    print("     'threshold': 0.0")
    print("   }")
    print()
    print("   Response structure shown above would be returned as JSON.")


def demonstrate_http_api_structure():
    """Show how this would work over HTTP API."""

    print("\n🔌 HTTP API Integration Status:")
    print("-" * 40)

    # Check if server is running (would need to start it separately)
    try:
        response = requests.get("http://localhost:8000/health", timeout=1)
        server_running = response.status_code == 200
    except:
        server_running = False

    if server_running:
        print("   ✅ Holon server is running on localhost:8000")

        # Example HTTP request structure
        print("\n   📡 Example HTTP Request:")
        print("   POST http://localhost:8000/query")
        print("   Content-Type: application/json")
        print("   {")
        print('     "probe": "{... ngram-encoded search query ...}",')
        print('     "data_type": "json",')
        print('     "top_k": 5,')
        print('     "threshold": 0.0')
        print("   }")

        print("\n   📨 HTTP Response Structure:")
        print("   {")
        print('     "results": [')
        print("       {")
        print('         "id": "paragraph_uuid",')
        print('         "score": 0.85,')
        print('         "data": {')
        print('           "coordinates": {')
        print('             "chapter": "Chapter IV",')
        print('             "paragraph_num": 1,')
        print('             "page_start": 4,')
        print('             "page_end": 4')
        print("           },")
        print('           "text": {')
        print('             "_encode_mode": "ngram",')
        print(
            '             "sequence": ["integration", "is", "the", "reverse", "of", "differentiation"]'
        )
        print("           },")
        print('           "metadata": {')
        print('             "book_title": "Calculus Made Easy",')
        print('             "id": "Chapter IV.1"')
        print("           }")
        print("         }")
        print("       }")
        print("     ]")
        print("   }")

    else:
        print("   ⚠️  Holon server not running")
        print("   💡 To test HTTP API:")
        print(
            "      1. ./scripts/run_with_venv.sh python scripts/server/holon_server.py"
        )
        print("      2. Use coordinate_api_demo.py with HTTP requests")

    print("\n🔧 API Extension Needed:")
    print("   The current /query endpoint returns basic results.")
    print("   For coordinate system, we could add:")
    print("   • /quote-search endpoint with coordinate-enriched responses")
    print("   • /paragraph-lookup endpoint for coordinate-based queries")
    print("   • Enhanced /query with coordinate post-processing")


if __name__ == "__main__":
    demonstrate_concrete_response_structure()
    demonstrate_http_api_structure()
