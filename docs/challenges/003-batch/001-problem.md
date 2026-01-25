# Holon-Powered Quote Finder for Books

Goal: Build a "quote finder" using Holon's VSA/HDC for efficient, fuzzy indexing and search of book content. The system should ingest a PDF (or text file) of a book, extract structured metadata (chapters, paragraphs, page numbers), group text into sentence-level units, normalize words (lowercase, remove punctuation/stops), and index positional word placements. Queries should check if a phrase (sequence of words) appears (exactly or fuzzily), and return locations with metadata.

This leverages traditional VSA/HDC strengths: Encode words as atomic HVs, sequences via permutations (for positions), bind metadata (chapter/page/para), and use similarity search for fuzzy matching. No backtracking needed—queries are geometric alignments in hyperspace.

### Input Data Handling
- Support PDF input (use a library like PyPDF2 if available, or assume pre-extracted text; for demo, hardcode or load a sample text file).
- Extract structure: Assume simple heuristics (e.g., "Chapter X" lines for chapters, blank lines for paragraphs, page breaks if detectable).
- Example book: Use a public domain text like "Pride and Prejudice" (provide a snippet or link to full text for ingestion).
- Normalization: Lowercase, remove punctuation (use regex/string ops), optionally stem or lemmatize for fuzziness.

Processed Unit Structure (EDN/JSON for Holon):
- {:unit-id (uuid or int)
  :text (string – full sentence/para)
  :words (vector of strings – normalized words in order)
  :metadata {:chapter (string/int), :paragraph (int), :page (int or range e.g. [5 6]), :book-title (string)}
  :context (optional set of keywords from surrounding text)}

### Requirements
1. Ingest and Process:
   - Load PDF/text file.
   - Parse into units (sentences or short paras, ~50-200 words max to avoid dilution).
   - Normalize words: e.g., "The quick brown fox." → ["the", "quick", "brown", "fox"]
   - Assign metadata (simulate if not in PDF: increment chapter on headers, para on newlines, page on estimates).
   - Ingest 100–500 units from a sample book.

2. Encoding Strategy (VSA/HDC via Holon):
   - Words: Random HVs for each unique normalized word.
   - Positions: Use permutations ρ^k(word_HV) for k-th position in sequence.
   - Unit: Bundle sequence = ∑ ρ^k(word_k), then bind metadata (e.g., unit = sequence ⊙ chapter_HV ⊙ para_HV ⊙ page_HV).
   - Store in Holon for fuzzy similarity search.

3. Query Support:
   - "Does phrase ['quick', 'brown', 'fox'] appear?" → Yes/No + locations (top matches via similarity > threshold).
   - "Where does similar to 'pride prejudice' appear?" (fuzzy, with wildcards/negations e.g. "pride * prejudice NOT vanity").
   - Return ranked results: unit text snippet, metadata, similarity score.
   - Guards: e.g., chapter = 1, page > 50.
   - Handle sequences: Probe with permuted phrase bundle, unbind to locate positions.

4. Demo:
   - Ingest sample book text.
   - Run 3–5 queries: exact phrase, fuzzy variant, with metadata filters.
   - Show how VSA handles noise (e.g., minor word changes still match via angle similarity).

Show full code: imports (include any text/PDF parsers), Holon setup (high dims e.g. 10000), encoding/ingestion, query functions with examples. Discuss any limitations (e.g., long sequences dilute; suggest chunking).
