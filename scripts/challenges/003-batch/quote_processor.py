#!/usr/bin/env python3
"""
Quote Processor for Holon Quote Finder
Processes quotes from text files with proper metadata extraction.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import json


class QuoteProcessor:
    """Process quotes from text files or PDF files with metadata extraction"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.is_pdf = self.file_path.suffix.lower() == '.pdf'

    def extract_quotes_with_metadata(self) -> List[Dict[str, Any]]:
        """Extract quotes from file (PDF or text) with metadata."""
        print(f"📖 Reading {'PDF' if self.is_pdf else 'text'} from: {self.file_path}")

        if self.is_pdf:
            return self._extract_from_pdf()
        else:
            return self._extract_from_text()

    def _extract_from_text(self) -> List[Dict[str, Any]]:
        """Extract quotes from text file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into lines and process
        lines = content.strip().split('\n')
        quotes = []

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

    def _extract_from_pdf(self) -> List[Dict[str, Any]]:
        """Extract quotes from PDF file."""
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError("PDF processing requires pypdf or PyPDF2. Install with: pip install pypdf")

        # Extract text from PDF
        with open(self.file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)

            pages_data = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                except AttributeError:
                    # Handle PyPDF2 API difference
                    text = page.extractText()

                # Clean up text
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()

                pages_data.append({
                    'page': page_num,
                    'text': text,
                    'word_count': len(text.split())
                })

                print(f"📄 Processed page {page_num}: {len(text.split())} words")

        # Extract quotes from all pages
        quotes = []
        for page_data in pages_data:
            page_quotes = self._extract_quotes_from_page_text(page_data['text'], page_data['page'])
            quotes.extend(page_quotes)

        print(f"✅ Extracted {len(quotes)} quotes from PDF")
        return quotes

    def _extract_quotes_from_page_text(self, page_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract quotes from a single page of text."""
        quotes = []

        # Look for quoted text (between quotation marks)
        quote_pattern = r'"([^"]*?)"'
        matches = re.findall(quote_pattern, page_text)

        for quote_text in matches:
            # Skip very short quotes
            if len(quote_text.split()) < 3:
                continue

            # Find context around the quote
            context_start = max(0, page_text.find(f'"{quote_text}"') - 200)
            context_end = min(len(page_text), page_text.find(f'"{quote_text}"') + len(quote_text) + 200)
            context = page_text[context_start:context_end].replace(f'"{quote_text}"', '[QUOTE]').strip()

            # Extract chapter from context
            chapter = self._extract_chapter_from_context(context)

            quote_data = {
                'text': quote_text,
                'chapter': chapter,
                'page': page_num,
                'paragraph': 1,
                'book_title': 'Calculus Made Easy',
                'context': context,
                'word_count': len(quote_text.split()),
                'line_number': 0  # Not applicable for PDF
            }

            quotes.append(quote_data)
            print(f"📝 Quote {len(quotes)}: [{chapter}] Page {page_num} | {quote_text[:50]}...")

        return quotes

    def _extract_chapter_from_context(self, context: str) -> str:
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

        # Default chapter extraction based on context keywords
        context_lower = context.lower()
        if 'prologue' in context_lower or 'epigraph' in context_lower:
            return "Prologue"
        elif 'chapter i' in context_lower or 'chapter 1' in context_lower:
            return "Chapter I"
        elif 'integration' in context_lower:
            return "Integration Chapter"
        else:
            return "Various Chapters"

    def save_quotes_to_json(self, quotes: List[Dict[str, Any]], output_path: str):
        """Save extracted quotes to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'source': str(self.file_path),
                    'extraction_method': 'pdf_quote_extraction' if self.is_pdf else 'text_quote_extraction',
                    'total_quotes': len(quotes)
                },
                'quotes': quotes
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(quotes)} quotes to {output_file}")

    def process_quotes(self, output_json_path: str) -> List[Dict[str, Any]]:
        """Complete processing pipeline."""
        print("🚀 Starting quote processing...")

        # Extract quotes with metadata
        quotes = self.extract_quotes_with_metadata()

        # Save to JSON
        self.save_quotes_to_json(quotes, output_json_path)

        print("✅ Quote processing complete!")
        return quotes


def main():
    """Process quotes from PDF or text file"""
    project_root = Path(__file__).parent.parent.parent.parent

    # Try PDF first
    pdf_file = project_root / "docs" / "challenges" / "003-batch" / "calculus-made-easy.pdf"
    text_file = project_root / "docs" / "challenges" / "003-batch" / "quotes.txt"
    output_json = Path(__file__).parent / "processed_quotes.json"

    if pdf_file.exists():
        print("🎯 Found PDF file - processing with PDF extraction")
        input_file = pdf_file
    elif text_file.exists():
        print("📄 PDF not found, using text file fallback")
        input_file = text_file
    else:
        print(f"❌ Neither PDF nor text file found at: {pdf_file} or {text_file}")
        print("Please ensure either calculus-made-easy.pdf or quotes.txt is in docs/challenges/003-batch/")
        return

    try:
        processor = QuoteProcessor(str(input_file))
        quotes = processor.process_quotes(str(output_json))

        print("\n📊 Processing Summary:")
        print(f"   File type: {'PDF' if processor.is_pdf else 'Text'}")
        print(f"   Total quotes: {len(quotes)}")
        print(f"   Chapters found: {set(q['chapter'] for q in quotes)}")
        print(f"   Page range: {min(q['page'] for q in quotes)} - {max(q['page'] for q in quotes)}")
        print(f"   Word count range: {min(q['word_count'] for q in quotes)} - {max(q['word_count'] for q in quotes)}")
        print(f"   Output saved to: {output_json}")

        if processor.is_pdf and len(quotes) == 0:
            print("⚠️  No quotes found in PDF. The PDF might not contain properly formatted quotes.")
            print("   Try using the text file (quotes.txt) for testing.")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        if "PDF" in str(e):
            print("💡 PDF processing failed. Try using quotes.txt instead.")
        return


if __name__ == "__main__":
    main()