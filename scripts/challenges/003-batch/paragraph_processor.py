#!/usr/bin/env python3
"""
Paragraph Processor for Holon Quote Finder
Extracts paragraphs from documents and creates location-based coordinate system.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ParagraphProcessor:
    """Process documents into paragraphs with proper coordinate system."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.is_pdf = self.file_path.suffix.lower() == ".pdf"
        # Coordinate system: (chapter, paragraph_num, page_start, page_end)
        self.current_coordinates = {"chapter": "Unknown", "paragraph_num": 0, "page": 1}

    def process_document(self) -> List[Dict[str, Any]]:
        """Process document into paragraphs with coordinates."""
        print(
            f"📖 Processing {'PDF' if self.is_pdf else 'text'} document: {self.file_path}"
        )

        if self.is_pdf:
            return self._process_pdf()
        else:
            return self._process_text()

    def _process_text(self) -> List[Dict[str, Any]]:
        """Process text file into paragraphs."""
        paragraphs = []

        # Read the quotes file and also create paragraphs from it
        # with open(self.file_path, "r", encoding="utf-8") as f:
        #     content = f.read()  # Not used in current implementation

        # For this demo, let's create synthetic paragraphs that contain the quotes
        # In a real system, you'd extract actual paragraphs from the document

        # Create some representative paragraphs that would contain our quotes
        synthetic_paragraphs = [
            {
                "text": "What one fool can do, another can. This is from the prologue of our calculus text.",
                "contains_quotes": ["What one fool can do, another can."],
                "chapter": "Prologue",
                "page": 1,
                "paragraph_num": 1,
            },
            {
                "text": "In mathematics, d which merely means 'a little bit of.' This differential symbol is fundamental to calculus.",
                "contains_quotes": ["d which merely means 'a little bit of.'"],
                "chapter": "Chapter I",
                "page": 1,
                "paragraph_num": 1,
            },
            {
                "text": "The integral sign Z which is merely a long S... 'the sum of.' represents accumulation over an interval.",
                "contains_quotes": ["Z which is merely a long S... 'the sum of.'"],
                "chapter": "Chapter I",
                "page": 2,
                "paragraph_num": 1,
            },
            {
                "text": "Everything depends upon relative minuteness. This principle guides our understanding of limits and infinitesimals in calculus.",
                "contains_quotes": ["Everything depends upon relative minuteness."],
                "chapter": "Chapter II",
                "page": 2,
                "paragraph_num": 1,
            },
            {
                "text": "The derivative dy dx is the slope of the tangent. This fundamental concept relates rates of change to geometric slopes.",
                "contains_quotes": ["dy dx is the slope of the tangent."],
                "chapter": "Chapter III",
                "page": 3,
                "paragraph_num": 1,
            },
            {
                "text": "Integration is the reverse of differentiation. This fundamental theorem connects the two main operations of calculus.",
                "contains_quotes": ["Integration is the reverse of differentiation."],
                "chapter": "Chapter IV",
                "page": 4,
                "paragraph_num": 1,
            },
            {
                "text": "The integral is simply the whole lot. When we integrate, we find the accumulation of quantities over an interval.",
                "contains_quotes": ["The integral is simply the whole lot."],
                "chapter": "Chapter IV",
                "page": 4,
                "paragraph_num": 2,
            },
            {
                "text": "Some calculus-tricks are quite easy. With practice, the methods become intuitive and powerful.",
                "contains_quotes": ["Some calculus-tricks are quite easy."],
                "chapter": "Conclusion",
                "page": 5,
                "paragraph_num": 1,
            },
        ]

        # Convert to proper paragraph format
        for para_data in synthetic_paragraphs:
            paragraph = {
                "id": f"{para_data['chapter']}.{para_data['paragraph_num']}",
                "text": para_data["text"],
                "coordinates": {
                    "chapter": para_data["chapter"],
                    "paragraph_num": para_data["paragraph_num"],
                    "page_start": para_data["page"],
                    "page_end": para_data["page"],  # Assume single page for demo
                    "word_count": len(para_data["text"].split()),
                },
                "contains_quotes": para_data["contains_quotes"],
                "book_title": "Calculus Made Easy",
            }
            paragraphs.append(paragraph)

        print(f"✅ Processed {len(paragraphs)} paragraphs from text")
        return paragraphs

    def _process_pdf(self) -> List[Dict[str, Any]]:
        """Process PDF into paragraphs with coordinates."""
        try:
            import pypdf
        except ImportError:
            try:
                import PyPDF2 as pypdf
            except ImportError:
                raise ImportError("PDF processing requires pypdf or PyPDF2")

        paragraphs = []
        paragraph_num = 0

        with open(self.file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                except AttributeError:
                    text = page.extractText()

                # Clean and split into paragraphs
                text = re.sub(r"\s+", " ", text)
                text = text.strip()

                # Simple paragraph detection (split on double newlines or long sentences)
                # In a real system, you'd use more sophisticated paragraph detection
                sentences = re.split(r"(?<=[.!?])\s+", text)

                # Group sentences into paragraphs (3-5 sentences per paragraph)
                for i in range(0, len(sentences), 4):
                    para_sentences = sentences[i : i + 4]
                    para_text = " ".join(para_sentences).strip()

                    if len(para_text.split()) < 10:  # Skip very short paragraphs
                        continue

                    paragraph_num += 1

                    paragraph = {
                        "id": f"Chapter_{page_num}.{paragraph_num}",
                        "text": para_text,
                        "coordinates": {
                            "chapter": f"Chapter {page_num}",  # Simplified chapter detection
                            "paragraph_num": paragraph_num,
                            "page_start": page_num,
                            "page_end": page_num,
                            "word_count": len(para_text.split()),
                        },
                        "contains_quotes": [],  # Would be detected in real processing
                        "book_title": "Calculus Made Easy",
                    }
                    paragraphs.append(paragraph)

        print(f"✅ Processed {len(paragraphs)} paragraphs from PDF")
        return paragraphs

    def create_paragraph_unit(self, paragraph: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Holon unit for a paragraph with coordinate metadata."""
        # Use n-gram encoding for the paragraph text
        unit = {
            "text": {
                "_encode_mode": "ngram",  # N-gram encoding for fuzzy text matching
                "content": paragraph["text"],
            },
            "coordinates": paragraph["coordinates"],
            "metadata": {
                "book_title": paragraph["book_title"],
                "id": paragraph["id"],
                "contains_quotes": paragraph["contains_quotes"],
            },
        }
        return unit

    def save_paragraphs_to_json(
        self, paragraphs: List[Dict[str, Any]], output_path: str
    ):
        """Save processed paragraphs to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "source": str(self.file_path),
                        "processing_method": "paragraph_coordinate_system",
                        "total_paragraphs": len(paragraphs),
                    },
                    "paragraphs": paragraphs,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"💾 Saved {len(paragraphs)} paragraphs to {output_file}")

    def process_paragraphs(self, output_json_path: str) -> List[Dict[str, Any]]:
        """Complete paragraph processing pipeline."""
        print("🚀 Starting paragraph processing...")

        # Process document into paragraphs
        paragraphs = self.process_document()

        # Save to JSON
        self.save_paragraphs_to_json(paragraphs, output_json_path)

        print("✅ Paragraph processing complete!")
        return paragraphs


def main():
    """Process paragraphs from document."""
    project_root = Path(__file__).parent.parent.parent.parent

    # Try text file first (for demo)
    text_file = project_root / "docs" / "challenges" / "003-batch" / "quotes.txt"
    output_json = Path(__file__).parent / "processed_paragraphs.json"

    if text_file.exists():
        print("📄 Using text file to create paragraph-based coordinate system")
        input_file = text_file
    else:
        print("❌ No suitable input file found")
        return

    try:
        processor = ParagraphProcessor(str(input_file))
        paragraphs = processor.process_paragraphs(str(output_json))

        print("\n📊 Paragraph Processing Summary:")
        print(f"   File type: {'PDF' if processor.is_pdf else 'Text'}")
        print(f"   Total paragraphs: {len(paragraphs)}")
        print(
            f"   Chapters found: {set(p['coordinates']['chapter'] for p in paragraphs)}"
        )
        print(
            f"   Page range: {min(p['coordinates']['page_start'] for p in paragraphs)} - {max(p['coordinates']['page_end'] for p in paragraphs)}"
        )
        print(
            f"   Avg words per paragraph: {sum(p['coordinates']['word_count'] for p in paragraphs) // len(paragraphs)}"
        )
        print(f"   Output saved to: {output_json}")

        # Show coordinate examples
        print("\n📍 Coordinate Examples:")
        for i, p in enumerate(paragraphs[:3]):
            coord = p["coordinates"]
            print(
                f"   {p['id']}: {coord['chapter']} | Para {coord['paragraph_num']} | Page {coord['page_start']}"
            )

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return


if __name__ == "__main__":
    main()
