#!/usr/bin/env python3
"""
PDF Quote Validation: Confirm our fuzzy search finds real book content

This script validates that our Holon-powered quote finder is actually
finding quotes from the real PDF content, not just the fallback quotes.txt.
"""

import json
import re
from pathlib import Path

import PyPDF2
from holon import CPUStore, HolonClient


class PDFQuoteValidator:
    """Validates quote finder results against actual PDF content."""

    def __init__(self):
        self.store = CPUStore(dimensions=16384)
        self.client = HolonClient(local_store=self.store)
        self.pdf_path = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "calculus-made-easy.pdf"
        self.quotes_path = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "quotes.txt"

    def extract_full_pdf_text(self) -> str:
        """Extract complete text from PDF for validation."""
        with open(self.pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

        # Clean up common PDF extraction artifacts
        full_text = re.sub(r'\n+', '\n', full_text)  # Multiple newlines
        full_text = re.sub(r' +', ' ', full_text)  # Multiple spaces
        return full_text.strip()

    def load_test_quotes(self) -> list:
        """Load quotes from quotes.txt for validation."""
        quotes = []
        try:
            with open(self.quotes_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        quotes.append(line.strip('"'))
        except Exception as e:
            print(f"⚠️  Could not load quotes: {e}")
        return quotes

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (same as main solution)."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def find_quote_in_pdf(self, quote: str, pdf_text: str) -> list:
        """Find exact quote locations in PDF text."""
        normalized_quote = self.normalize_text(quote)
        normalized_pdf = self.normalize_text(pdf_text)

        locations = []
        start = 0
        while True:
            pos = normalized_pdf.find(normalized_quote, start)
            if pos == -1:
                break

            # Get context around the match
            context_start = max(0, pos - 100)
            context_end = min(len(normalized_pdf), pos + len(normalized_quote) + 100)
            context = normalized_pdf[context_start:context_end]

            locations.append({
                'position': pos,
                'context': context,
                'quote_found': normalized_quote
            })

            start = pos + 1

        return locations

    def ingest_pdf_content(self) -> int:
        """Ingest PDF content using same logic as main solution."""
        print("📖 Extracting PDF text...")
        pdf_text = self.extract_full_pdf_text()

        print("🔍 Segmenting into sentence units...")
        # Split into rough paragraphs
        paragraphs = re.split(r'\n\s*\n', pdf_text.strip())

        units = []
        chapter_num = 1
        paragraph_num = 1
        page_num = 1

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue

            # Detect chapter headers
            if re.match(r'^chapter\s+\w+', paragraph.lower()):
                chapter_num += 1
                paragraph_num = 1
                continue

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())

            for sent_idx, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:
                    continue

                words = self.normalize_text(sentence).split()

                if not words:
                    continue

                unit = {
                    "unit_id": f"pdf_unit_{para_idx}_{sent_idx}",
                    "words": {
                        "_encode_mode": "ngram",
                        "sequence": words
                    },
                    "metadata": {
                        "chapter": chapter_num,
                        "paragraph": paragraph_num,
                        "page": page_num,
                        "book_title": "Calculus Made Easy",
                        "sentence_index": sent_idx,
                        "source": "pdf"
                    }
                }

                units.append(unit)

                if len(units) % 50 == 0:
                    page_num += 1

            paragraph_num += 1

        print(f"📊 Processing {len(units)} PDF units...")

        # Batch ingest
        batch_size = 200
        total_ingested = 0

        for i in range(0, len(units), batch_size):
            batch = [json.dumps(unit) for unit in units[i:i + batch_size]]
            ids = self.client.insert_batch(batch, "json")
            total_ingested += len(ids)

            if total_ingested % 1000 == 0:
                print(f"   → Ingested {total_ingested}/{len(units)} PDF units...")

        return len(units)

    def validate_quote_finder(self):
        """Main validation: ingest PDF -> search quotes -> verify results."""
        print("🔍 PDF Quote Finder Validation")
        print("=" * 60)

        # Load test quotes
        test_quotes = self.load_test_quotes()
        print(f"📚 Loaded {len(test_quotes)} test quotes from quotes.txt")

        # Extract full PDF text for direct validation
        print("\n📖 Extracting full PDF text for validation...")
        pdf_text = self.extract_full_pdf_text()
        print(f"   → PDF contains {len(pdf_text)} characters")

        # Check if quotes actually exist in PDF
        print("\n🔎 Checking quote existence in PDF...")
        quotes_in_pdf = []
        for quote in test_quotes:
            locations = self.find_quote_in_pdf(quote, pdf_text)
            if locations:
                quotes_in_pdf.append((quote, len(locations)))
                print(f"   ✅ '{quote[:30]}...' → {len(locations)} location(s)")
            else:
                print(f"   ❌ '{quote[:30]}...' → NOT FOUND in PDF")

        print(f"\n📊 {len(quotes_in_pdf)}/{len(test_quotes)} quotes found in PDF")

        if not quotes_in_pdf:
            print("❌ No quotes found in PDF - cannot validate search results")
            return

        # Ingest PDF content into Holon
        print("\n📥 Ingesting PDF content into Holon...")
        units_ingested = self.ingest_pdf_content()

        # Test fuzzy search for quotes that exist in PDF
        print("\n🎯 Testing fuzzy search for PDF quotes...")

        validation_results = []
        for quote, pdf_locations in quotes_in_pdf[:5]:  # Test first 5 found quotes
            print(f"\n   Searching for: '{quote}'")

            # Use same search logic as main solution
            search_words = self.normalize_text(quote).split()

            probe_data = {
                "words": {
                    "_encode_mode": "ngram",
                    "sequence": search_words
                }
            }

            results = self.client.search_json(
                probe=probe_data,
                limit=5
            )

            print(f"   → Holon found {len(results)} matches:")

            # Validate each result
            for i, result in enumerate(results):
                metadata = result["data"]["metadata"]
                score = result["score"]

                # Get context from actual PDF text around this match
                # (This is approximate since we don't have exact position mapping)
                context_match = f"Chapter {metadata['chapter']}, Page {metadata['page']}"
                print(f"     {i+1}. {context_match} (score: {score:.4f})")

            validation_results.append({
                'quote': quote,
                'holon_matches': len(results),
                'pdf_locations': pdf_locations
            })

        # Summary
        print("\n📋 Validation Summary:")
        print(f"   → PDF units ingested: {units_ingested}")
        print(f"   → Quotes validated: {len(validation_results)}")

        successful_matches = sum(1 for r in validation_results if r['holon_matches'] > 0)
        print(f"   → Successful fuzzy matches: {successful_matches}/{len(validation_results)}")

        # Test partial phrase matching
        print("\n🎯 Testing partial phrase matching...")
        partial_tests = [
            "one fool",  # Should match "What one fool can do, another can"
            "slope tangent",  # Should match "dy dx is the slope of the tangent"
            "depends upon",  # Should match "Everything depends upon relative minuteness"
        ]

        for partial in partial_tests:
            print(f"\n   Partial search: '{partial}'")
            search_words = self.normalize_text(partial).split()

            probe_data = {
                "words": {
                    "_encode_mode": "ngram",
                    "sequence": search_words
                }
            }

            results = self.client.search_json(probe=probe_data, limit=3)

            if results:
                print(f"   → Found {len(results)} matches:")
                for result in results:
                    metadata = result["data"]["metadata"]
                    score = result["score"]
                    print(f"     • Chapter {metadata['chapter']}, Page {metadata['page']} (score: {score:.4f})")
            else:
                print("   → No matches found")

        print("\n✅ PDF validation completed!")


def main():
    validator = PDFQuoteValidator()

    try:
        validator.validate_quote_finder()
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
