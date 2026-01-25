# Holon-Powered Quote Finder for Books (with Vector Bootstrapping)

Goal: Build a "quote finder" using Holon's VSA/HDC for efficient, fuzzy indexing and search of book content. The system should ingest a PDF (or text file) of a book, extract structured metadata (chapters, paragraphs, page numbers), group text into sentence-level units, normalize words (lowercase, remove punctuation/stops), and index positional word placements. Queries should check if a phrase (sequence of words) appears (exactly or fuzzily), and return locations with metadata.

**Key Innovation**: Instead of storing full text in the database, store only metadata pointers. Use n-gram encoding for word sequences and provide a vector bootstrapping API where users can submit data blobs to get vectors for follow-up queries.

This leverages VSA/HDC strengths: Encode words as atomic HVs, sequences via n-gram bindings (for fuzzy subsequence matching), bind metadata (chapter/page/para), and use similarity search for geometric alignments. Enhanced with vector bootstrapping for user-driven encoding.

### Input Data Handling
- Support PDF input (use a library like PyPDF2 if available, or assume pre-extracted text; for demo, hardcode or load a sample text file).
- Extract structure: Assume simple heuristics (e.g., "Chapter X" lines for chapters, blank lines for paragraphs, page breaks if detectable).
- Example book: Use a public domain text like "Pride and Prejudice" (provide a snippet or link to full text for ingestion).
- Normalization: Lowercase, remove punctuation (use regex/string ops), optionally stem or lemmatize for fuzziness.

Processed Unit Structure (EDN/JSON for Holon):
- {:unit-id (uuid or int)
  :words {"_encode_mode": "ngram", "sequence": ["normalized", "words", "in", "order"]}
  :metadata {:chapter (string/int), :paragraph (int), :page (int or range e.g. [5 6]), :book-title (string)}
  :bootstrap_vector (optional pre-computed vector for the word sequence)}
- **Note**: Full text is NOT stored in DB—only metadata pointers for reconstruction

### Requirements
1. Ingest and Process:
   - Load PDF/text file.
   - Parse into units (sentences or short paras, ~50-200 words max to avoid dilution).
   - Normalize words: e.g., "The quick brown fox." → ["the", "quick", "brown", "fox"]
   - Assign metadata (simulate if not in PDF: increment chapter on headers, para on newlines, page on estimates).
   - Ingest 100–500 units from a sample book.

2. Encoding Strategy (VSA/HDC via Holon with Enhanced Modes):
   - Words: Random HVs for each unique normalized word.
   - Sequences: Use n-gram encoding (pairs + singles) for fuzzy subsequence matching without absolute positions.
   - Unit: Bind n-gram encoded words with metadata (unit = words_HV ⊙ chapter_HV ⊙ para_HV ⊙ page_HV).
   - Store only metadata pointers—no full text in database.
   - Vector Bootstrapping: New /encode API returns vectors for data blobs without storage.

3. Vector Bootstrapping API:
   - POST /encode: Submit data blob, get back encoded vector for follow-up queries.
   - Allows users to pre-compute vectors for search terms or phrases.
   - Example: Submit {"words": {"_encode_mode": "ngram", "sequence": ["quick", "brown", "fox"]}} → get vector for fuzzy phrase matching.

4. Query Support with Bootstrapping:
   - "Does phrase ['quick', 'brown', 'fox'] appear?": Use bootstrapped vector from /encode API.
   - Fuzzy matching: Similar phrases match via n-gram overlap in hyperspace.
   - Return ranked results: metadata pointers + similarity scores (no stored text).
   - Guards: chapter = 1, page > 50, etc.
   - API Integration: Bootstrap search vectors, then query against stored metadata units.

5. Demo with New Features:
   - Ingest calculus quotes with metadata pointers (no full text storage).
   - Use /encode API to bootstrap vectors for search phrases.
   - Run queries: exact quotes, fuzzy variants, with metadata filters.
   - Show n-gram encoding handles mid-sentence matches and fuzzy similarity.

Show full code: imports (PDF parser), Holon setup (16k dims), n-gram encoding, /encode API usage, query examples. Discuss advantages: no text storage, vector bootstrapping, fuzzy subsequence matching.
