#!/usr/bin/env python3
"""
Quote Processor for Holon Quote Finder
Processes quotes from text files with proper metadata extraction.
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class QuoteProcessor:
    """Process quotes from text files with metadata extraction"""

    def __init__(self, text_file_path: str):
        self.text_file_path = Path(text_file_path)
        if not self.text_file_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_file_path}")

    def extract_quotes_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract quotes from text file with metadata."""
        print(f"📖 Reading quotes from: {self.text_file_path}")

        with open(self.text_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into lines and process
        lines = content.strip().split('\n')
        quotes = []
        current_page = 1

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Look for quoted text
            quote_match = re.search(r'"([^"]*)"', line)
            if quote_match:
                quote_text = quote_match.group(1)

                # Skip very short quotes
                if len(quote_text.split()) < 3:
                    continue

                # Extract context (the whole line minus the quote)
                context = line.replace(f'"{quote_text}"', '').strip()

                # Extract chapter from context
                chapter = self._extract_chapter_from_context(context)

                # Simulate page based on line number (rough approximation)
                page = ((line_num - 1) // 5) + 1  # ~5 lines per page

                quote_data = {
                    'text': quote_text,
                    'chapter': chapter,
                    'page': page,
                    'paragraph': 1,
                    'book_title': 'Calculus Made Easy',
                    'context': context,
                    'word_count': len(quote_text.split()),
                    'line_number': line_num
                }

                quotes.append(quote_data)
                print(f"📝 Quote {len(quotes)}: [{chapter}] Page {page} | {quote_text[:50]}...")

        print(f"✅ Extracted {len(quotes)} quotes from text file")
        return quotes

    def parse_quotes_from_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse quotes from page text data."""
        quotes = []

        for page_data in pages_data:
            page_num = page_data['page']
            text = page_data['text']

            # Look for quoted text (between quotation marks)
            quote_pattern = r'"([^"]*?)"'
            matches = re.findall(quote_pattern, text)

            for i, quote_text in enumerate(matches):
                # Skip very short quotes (likely not meaningful)
                if len(quote_text.split()) < 3:
                    continue

                # Extract context around the quote
                context_start = max(0, text.find(f'"{quote_text}"') - 200)
                context_end = min(len(text), text.find(f'"{quote_text}"') + len(quote_text) + 200)
                context = text[context_start:context_end]

                # Try to extract chapter information from context
                chapter = self._extract_chapter_from_context(context, page_num)

                # Create structured quote data
                quote_data = {
                    'text': quote_text,
                    'chapter': chapter,
                    'page': page_num,
                    'paragraph': 1,  # Simplified - could parse paragraphs
                    'book_title': 'Calculus Made Easy',
                    'context': context.replace(f'"{quote_text}"', '[QUOTE]'),
                    'word_count': len(quote_text.split())
                }

                quotes.append(quote_data)
                print(f"📝 Quote {len(quotes)}: [{chapter}] Page {page_num} | {quote_text[:60]}...")

        print(f"✅ Parsed {len(quotes)} quotes from PDF")
        return quotes

    def _extract_chapter_from_context(self, context: str, page_num: int) -> str:
        """Extract chapter information from surrounding context."""

        # Look for chapter headers in context
        chapter_patterns = [
            r'chapter\s+(\w+)',
            r'Chapter\s+(\w+)',
            r'CHAPTER\s+(\w+)'
        ]

        for pattern in chapter_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                chapter = match.group(1)
                # Convert roman numerals or numbers
                if chapter.upper() in ['I', 'II', 'III', 'IV', 'V']:
                    return f"Chapter {chapter.upper()}"
                elif chapter.isdigit():
                    return f"Chapter {chapter}"
                else:
                    return f"Chapter {chapter.title()}"

        # Fallback based on page number (rough approximation)
        if page_num <= 5:
            return "Prologue"
        elif page_num <= 15:
            return "Chapter I"
        elif page_num <= 25:
            return "Chapter II"
        elif page_num <= 35:
            return "Chapter III"
        else:
            return "Later Chapters"

    def save_quotes_to_json(self, quotes: List[Dict[str, Any]], output_path: str):
        """Save extracted quotes to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'source': str(self.pdf_path),
                    'extraction_method': 'pypdf_quote_extraction',
                    'total_quotes': len(quotes)
                },
                'quotes': quotes
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(quotes)} quotes to {output_file}")

    def extract_all_quotes(self, output_json_path: str) -> List[Dict[str, Any]]:
        """Complete extraction pipeline."""
        print("🚀 Starting PDF quote extraction...")

        # Extract text from PDF
        pages_data = self.extract_text_with_pages()

        # Parse quotes from pages
        quotes = self.parse_quotes_from_pages(pages_data)

        # Save to JSON
        self.save_quotes_to_json(quotes, output_json_path)

        print("✅ PDF extraction complete!")
        return quotes


def main():
    """Extract quotes from calculus-made-easy.pdf"""
    pdf_path = Path(__file__).parent.parent.parent.parent / "docs" / "challenges" / "003-batch" / "calculus-made-easy.pdf"
    output_json = Path(__file__).parent / "extracted_quotes.json"

    if not pdf_path.exists():
        print(f"❌ PDF not found at: {pdf_path}")
        print("Please ensure calculus-made-easy.pdf is in docs/challenges/003-batch/")
        return

    extractor = PDFQuoteExtractor(str(pdf_path))
    quotes = extractor.extract_all_quotes(str(output_json))

    print("\n📊 Extraction Summary:")
    print(f"   Total quotes: {len(quotes)}")
    print(f"   Chapters found: {set(q['chapter'] for q in quotes)}")
    print(f"   Page range: {min(q['page'] for q in quotes)} - {max(q['page'] for q in quotes)}")
    print(f"   Output saved to: {output_json}")


if __name__ == "__main__":
    main()