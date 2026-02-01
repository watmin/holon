#!/usr/bin/env python3
"""
Holon-Powered Quote Finder via HTTP API

Demonstrates the quote finder challenge using only HTTP endpoints.
Features:
- Batch ingestion via /api/v1/items/batch
- Vector bootstrapping via /api/v1/vectors/encode
- Search via /api/v1/search
"""

import json
import re
import time
import uuid
import requests
from pathlib import Path
from typing import Dict, List, Optional

BASE_URL = "http://localhost:8000"
API = "/api/v1"


def health_check() -> bool:
    """Check if server is running."""
    try:
        r = requests.get(f"{BASE_URL}{API}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def clear_store():
    """Clear the store for fresh start."""
    try:
        r = requests.post(f"{BASE_URL}{API}/store/clear", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def insert_batch(items: List[Dict]) -> List[str]:
    """Insert multiple items via HTTP."""
    json_strings = [json.dumps(item) for item in items]
    r = requests.post(f"{BASE_URL}{API}/items/batch", json={
        "items": json_strings,
        "data_type": "json"
    }, timeout=60)
    r.raise_for_status()
    return r.json()["ids"]


def encode(data: Dict) -> List[float]:
    """Encode data to vector via HTTP."""
    r = requests.post(f"{BASE_URL}{API}/vectors/encode", json={
        "data": json.dumps(data),
        "data_type": "json"
    }, timeout=10)
    r.raise_for_status()
    return r.json()["vector"]


def search(probe: Dict, limit: int = 10, threshold: float = 0.0) -> List[Dict]:
    """Search via HTTP."""
    r = requests.post(f"{BASE_URL}{API}/search", json={
        "probe": json.dumps(probe),
        "data_type": "json",
        "top_k": limit,
        "threshold": threshold
    }, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


def search_by_vector(vector: List[float], limit: int = 10) -> List[Dict]:
    """Search using bootstrapped vector."""
    r = requests.post(f"{BASE_URL}{API}/search/by-vector", json={
        "vector": vector,
        "top_k": limit,
        "threshold": 0.0
    }, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


def parse_result_data(data):
    """Parse result data from string to dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return json.loads(data)
    return data


class HTTPQuoteFinder:
    """Quote finder using HTTP API."""

    def __init__(self, book_path: Optional[Path] = None):
        self.book_path = book_path or Path(__file__).parent.parent.parent.parent / "docs/challenges/003-batch/calculus-made-easy.pdf"

    def normalize_text(self, text: str) -> List[str]:
        """Normalize text to word list."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if len(w) > 1]

    def extract_text(self) -> str:
        """Extract text from PDF or use fallback."""
        try:
            import PyPDF2
            with open(self.book_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:60]:  # First 60 pages
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"⚠️  PDF extraction failed: {e}")
            # Fallback to quotes file
            quotes_path = self.book_path.parent / "quotes.txt"
            if quotes_path.exists():
                return quotes_path.read_text()
            return ""

    def segment_into_units(self, text: str, max_words: int = 50) -> List[Dict]:
        """Segment text into searchable units."""
        units = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        page_estimate = 1
        chapter = 1

        for i, sentence in enumerate(sentences):
            words = self.normalize_text(sentence)
            if len(words) < 3:
                continue

            # Create unit with metadata
            unit = {
                "unit_id": str(uuid.uuid4()),
                "words": {
                    "$mode": "ngram",
                    "sequence": words[:max_words]
                },
                "metadata": {
                    "chapter": chapter,
                    "page": page_estimate,
                    "sentence_index": i
                }
            }
            units.append(unit)

            # Estimate page breaks
            if i > 0 and i % 50 == 0:
                page_estimate += 1

        return units

    def ingest_book(self, batch_size: int = 100) -> tuple:
        """Ingest book content via HTTP."""
        print("📖 Extracting book text...")
        text = self.extract_text()

        if not text:
            print("❌ No text to process")
            return 0, 0

        print("🔍 Segmenting into units...")
        units = self.segment_into_units(text)
        print(f"📊 Processing {len(units)} units...")

        start_time = time.time()
        total_ingested = 0

        # Batch insert
        for i in range(0, len(units), batch_size):
            batch = units[i:i+batch_size]
            ids = insert_batch(batch)
            total_ingested += len(ids)

            if total_ingested % 500 == 0:
                elapsed = time.time() - start_time
                rate = total_ingested / elapsed if elapsed > 0 else 0
                print(f"   → Ingested {total_ingested}/{len(units)} ({rate:.0f}/sec)...")

        elapsed = time.time() - start_time
        rate = total_ingested / elapsed if elapsed > 0 else 0

        print(f"   → Ingestion complete: {total_ingested} units in {elapsed:.1f}s ({rate:.0f}/sec)")
        return total_ingested, rate

    def search_quotes(self, phrase: str, limit: int = 5) -> List[Dict]:
        """Search for quotes matching a phrase."""
        words = self.normalize_text(phrase)
        if not words:
            return []

        probe = {
            "words": {
                "$mode": "ngram",
                "sequence": words
            }
        }

        return search(probe, limit=limit, threshold=0.05)

    def bootstrap_search(self, phrase: str, limit: int = 5) -> List[Dict]:
        """Search using vector bootstrapping."""
        words = self.normalize_text(phrase)
        if not words:
            return []

        # Bootstrap the search vector
        probe = {
            "words": {
                "$mode": "ngram",
                "sequence": words
            }
        }
        vector = encode(probe)

        # Search with the vector
        return search_by_vector(vector, limit=limit)


def main():
    print("=" * 70)
    print("QUOTE FINDER VIA HTTP API")
    print("=" * 70)

    # Check server
    if not health_check():
        print(f"❌ Server not running at {BASE_URL}")
        print("Start with: ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
        return

    print(f"✅ Connected to {BASE_URL}")

    # Clear store for fresh start
    print("\n🧹 Clearing store...")
    clear_store()

    # Initialize finder
    finder = HTTPQuoteFinder()

    # Ingest book
    print("\n📥 Ingesting book content via HTTP...")
    units, rate = finder.ingest_book()

    if units == 0:
        print("❌ No units ingested")
        return

    # Test searches
    test_quotes = [
        "What one fool can do, another can",
        "dy dx is the slope of the tangent",
        "Everything depends upon relative minuteness",
        "differential calculus",
        "integral",
    ]

    print("\n🔍 Testing quote searches via HTTP...")
    for quote in test_quotes:
        print(f"\n   Searching: '{quote}'")
        results = finder.search_quotes(quote, limit=3)

        if results:
            print(f"   Found {len(results)} matches:")
            for i, result in enumerate(results):
                data = parse_result_data(result["data"])
                metadata = data["metadata"]
                score = result["score"]
                print(f"     {i+1}. Ch.{metadata['chapter']} Pg.{metadata['page']} (score: {score:.4f})")
        else:
            print("   No matches found")

    # Vector bootstrapping demo
    print("\n🧪 Vector Bootstrapping via HTTP...")
    bootstrap_phrase = "slope of the tangent"
    print(f"   Bootstrapping vector for: '{bootstrap_phrase}'")

    words = finder.normalize_text(bootstrap_phrase)
    probe = {"words": {"$mode": "ngram", "sequence": words}}
    vector = encode(probe)
    print(f"   → Generated {len(vector)}D vector")

    # Search with bootstrapped vector
    print(f"   Searching with bootstrapped vector...")
    results = search_by_vector(vector, limit=3)

    if results:
        print(f"   Found {len(results)} matches:")
        for i, result in enumerate(results):
            data = parse_result_data(result.get("data", {}))
            if isinstance(data, dict) and "metadata" in data:
                metadata = data["metadata"]
                score = result.get("score", 0)
                print(f"     {i+1}. Ch.{metadata['chapter']} Pg.{metadata['page']} (score: {score:.4f})")
            else:
                print(f"     {i+1}. (score: {result.get('score', 0):.4f})")

    # Performance test
    print("\n⚡ Performance test...")
    test_phrases = ["differential", "integral", "calculus", "slope", "tangent"]

    start = time.time()
    for phrase in test_phrases * 4:  # 20 searches
        finder.search_quotes(phrase, limit=3)
    elapsed = time.time() - start

    print(f"   → 20 searches in {elapsed:.2f}s ({20/elapsed:.1f} queries/sec)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    ✅ Quote finder works via HTTP API

    Endpoints used:
    - POST /api/v1/items/batch - Batch ingestion
    - POST /api/v1/vectors/encode - Vector bootstrapping
    - POST /api/v1/search - Similarity search
    - POST /api/v1/search/by-vector - Search with bootstrapped vector

    Results:
    - Ingested {units} units at {rate:.0f}/sec
    - Search: {20/elapsed:.1f} queries/sec
    - Vector bootstrapping enables O(1) search vector computation

    The HTTP API provides full quote finder functionality!
    """)


if __name__ == "__main__":
    main()
