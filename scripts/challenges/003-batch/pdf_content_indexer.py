#!/usr/bin/env python3
"""
PDF Content Indexer - Index full PDF content with coordinates for quote finding.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pypdf

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("❌ pypdf required for PDF processing")
    exit(1)


class PDFContentIndexer:
    """Index full PDF content with coordinate tracking for quote location."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def index_pdf_content(self) -> List[Dict[str, Any]]:
        """Index full PDF content into coordinate-addressable chunks."""
        print(f"📖 Indexing PDF content: {self.pdf_path}")

        with open(self.pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            indexed_chunks = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                except AttributeError:
                    text = page.extractText()

                # Clean and process text
                text = re.sub(r"\s+", " ", text).strip()

                if not text:
                    continue

                # Split page into logical chunks (sentences/paragraphs)
                chunks = self._split_into_chunks(text, page_num)

                indexed_chunks.extend(chunks)

                print(
                    f"📄 Page {page_num}: {len(chunks)} chunks, {len(text.split())} words"
                )

        print(f"✅ Indexed {len(indexed_chunks)} content chunks from PDF")
        return indexed_chunks

    def _split_into_chunks(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Split page text into coordinate-addressable chunks."""
        chunks = []

        # Split into sentences (basic chunking)
        sentences = re.split(r"(?<=[.!?])\s+", page_text.strip())

        chunk_size = 3  # sentences per chunk
        chunk_num = 0

        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i : i + chunk_size]
            chunk_text = " ".join(chunk_sentences).strip()

            if len(chunk_text.split()) < 5:  # Skip very small chunks
                continue

            # Calculate character positions in page
            start_char = page_text.find(chunk_sentences[0])
            if start_char == -1:
                start_char = 0

            end_char = start_char + len(chunk_text)
            if end_char > len(page_text):
                end_char = len(page_text)

            # Calculate word positions
            words_before = len(page_text[:start_char].split())

            chunk = {
                "id": f"page_{page_num}_chunk_{chunk_num}",
                "content": chunk_text,
                "coordinates": {
                    "page": page_num,
                    "chunk_num": chunk_num,
                    "char_start": start_char,
                    "char_end": end_char,
                    "word_start": words_before,
                    "word_count": len(chunk_text.split()),
                },
                "metadata": {
                    "source": "pdf_content_index",
                    "chunk_type": "sentence_group",
                    "sentence_count": len(chunk_sentences),
                },
            }

            chunks.append(chunk)
            chunk_num += 1

        return chunks

    def save_indexed_content(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save indexed content to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "source_pdf": str(self.pdf_path),
                        "indexing_method": "coordinate_chunking",
                        "total_chunks": len(chunks),
                        "pages_indexed": len(
                            set(c["coordinates"]["page"] for c in chunks)
                        ),
                    },
                    "chunks": chunks,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"💾 Saved {len(chunks)} indexed chunks to {output_file}")

    def create_chunk_unit(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Holon unit for a content chunk with coordinates."""
        # Normalize chunk text
        words = self._normalize_text(chunk["content"])

        unit = {
            "text": {
                "_encode_mode": "ngram",  # N-gram encoding for fuzzy text matching
                "sequence": words,
            },
            "coordinates": chunk["coordinates"],
            "metadata": {
                "chunk_id": chunk["id"],
                "word_count": len(words),
                "source": "pdf_index",
            },
        }
        return unit

    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text for encoding."""
        import re

        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = [word for word in normalized.split() if word]
        return words

    def index_and_store(self, output_json_path: str) -> List[Dict[str, Any]]:
        """Complete PDF indexing and storage pipeline."""
        print("🚀 Starting PDF content indexing...")

        # Index PDF content
        chunks = self.index_pdf_content()

        # Save indexed content
        self.save_indexed_content(chunks, output_json_path)

        print("✅ PDF content indexing complete!")
        return chunks


class PDFQuoteLocator:
    """Locate quotes within indexed PDF content."""

    def __init__(self, indexed_chunks: List[Dict[str, Any]]):
        self.chunks = indexed_chunks
        self.chunk_lookup = {chunk["id"]: chunk for chunk in chunks}

    def find_quote_locations(
        self, quote_text: str, min_similarity: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find locations where a quote appears in the indexed content."""
        print(f"🔍 Locating quote: '{quote_text[:50]}...'")

        # Normalize quote for comparison
        quote_lower = quote_text.lower().strip()
        locations = []

        for chunk in self.chunks:
            chunk_text_lower = chunk["content"].lower()

            # Exact substring match
            if quote_lower in chunk_text_lower:
                # Find exact position
                start_pos = chunk_text_lower.find(quote_lower)
                word_start = len(chunk_text_lower[:start_pos].split())

                location = {
                    "coordinates": chunk["coordinates"],
                    "similarity": 1.0,  # Exact match
                    "match_type": "exact",
                    "chunk_content": chunk["content"],
                    "quote_found": quote_text,
                    "position_in_chunk": {
                        "word_start": word_start,
                        "word_end": word_start + len(quote_lower.split()),
                        "char_start": start_pos,
                        "char_end": start_pos + len(quote_lower),
                    },
                }
                locations.append(location)
                print(
                    f"   🎯 Exact match at: Page {chunk['coordinates']['page']}, Chunk {chunk['coordinates']['chunk_num']}"
                )

            # Fuzzy matching for partial matches
            elif min_similarity < 1.0:
                # Simple fuzzy match using sequence matcher
                import difflib

                matcher = difflib.SequenceMatcher(None, quote_lower, chunk_text_lower)
                similarity = matcher.ratio()

                if similarity >= min_similarity:
                    location = {
                        "coordinates": chunk["coordinates"],
                        "similarity": similarity,
                        "match_type": "fuzzy",
                        "chunk_content": chunk["content"],
                        "best_match": self._find_best_substring_match(
                            quote_lower, chunk_text_lower
                        ),
                        "position_in_chunk": None,  # Would need more complex analysis
                    }
                    locations.append(location)

        # Sort by similarity (exact matches first)
        locations.sort(
            key=lambda x: (
                -x["similarity"],
                x["coordinates"]["page"],
                x["coordinates"]["chunk_num"],
            )
        )

        print(f"✅ Found {len(locations)} quote locations")
        return locations

    def _find_best_substring_match(self, quote: str, text: str) -> str:
        """Find the best substring match in text."""
        import difflib

        # Find all substrings of similar length
        quote_words = quote.split()
        text_words = text.split()

        best_match = ""
        best_ratio = 0

        # Try different window sizes
        for window_size in [
            len(quote_words),
            len(quote_words) + 1,
            len(quote_words) - 1,
        ]:
            if window_size <= 0:
                continue

            for i in range(len(text_words) - window_size + 1):
                window = " ".join(text_words[i : i + window_size])
                ratio = difflib.SequenceMatcher(None, quote, window.lower()).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window

        return best_match if best_ratio > 0.6 else "No good match found"


def main():
    """Index PDF content and demonstrate quote location."""
    project_root = Path(__file__).parent.parent.parent.parent

    # Check for PDF
    pdf_file = (
        project_root / "docs" / "challenges" / "003-batch" / "calculus-made-easy.pdf"
    )
    index_output = Path(__file__).parent / "pdf_content_index.json"

    if not pdf_file.exists():
        print(f"❌ PDF not found at: {pdf_file}")
        print("Please ensure calculus-made-easy.pdf exists")
        return

    try:
        # Index the PDF content
        indexer = PDFContentIndexer(str(pdf_file))
        chunks = indexer.index_and_store(str(index_output))

        print("\n📊 Indexing Summary:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Pages indexed: {len(set(c['coordinates']['page'] for c in chunks))}")
        print(
            f"   Avg words per chunk: {sum(c['coordinates']['word_count'] for c in chunks) // len(chunks)}"
        )

        # Demonstrate quote location
        locator = PDFQuoteLocator(chunks)

        # Test quotes (these should appear in the PDF content)
        test_quotes = [
            "calculus made easy",
            "differential calculus",
            "integral calculus",
            "function of x",
        ]

        print("\n🎯 Quote Location Demo:")
        for quote in test_quotes:
            print(f"\n🔍 Searching for: '{quote}'")
            print("-" * 40)
            locations = locator.find_quote_locations(quote, min_similarity=0.7)

            if locations:
                print("   📍 Found locations:")
                for i, loc in enumerate(locations[:3], 1):  # Top 3
                    coord = loc["coordinates"]
                    print(".3f")
                    print(f"      📄 Chunk preview: {loc['chunk_content'][:80]}...")
                    if loc.get("quote_found"):
                        print(f"      🎯 Exact match: '{loc['quote_found']}'")
                    print()
            else:
                print("   ❌ No locations found")

        print("\n✅ PDF Content Indexing Complete!")
        print(
            "🎯 Users can now query for quotes and get exact PDF coordinates for manual verification!"
        )

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return


if __name__ == "__main__":
    main()
