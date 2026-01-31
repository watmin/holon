#!/usr/bin/env python3
"""
Challenge 007-003: Hierarchical Document Retrieval

Demonstrates navigating deeply nested legal/technical documents with cross-references.
Features:
- Section hierarchy (parent/child relationships)
- Cross-references between sections
- Amendment history
- Negation filters (NOT in section X)

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/003-hierarchical-docs-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/003-hierarchical-docs-solution.py --http
"""

import argparse
import json
import time
import uuid
from typing import Any, Dict, List

from holon import CPUStore, HolonClient


def generate_contract_sections() -> List[Dict[str, Any]]:
    """Generate sample contract sections."""
    sections = [
        {
            "document": "Contract-2024-001",
            "section": "1",
            "parent_sections": [],
            "clause_type": "definitions",
            "title": "Definitions",
            "text": "Terms used in this agreement shall have the meanings specified herein",
            "references": [],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "2",
            "parent_sections": [],
            "clause_type": "scope",
            "title": "Scope of Work",
            "text": "The contractor shall provide services as described in Appendix A",
            "references": ["Appendix A"],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "3",
            "parent_sections": [],
            "clause_type": "payment",
            "title": "Payment Terms",
            "text": "Payment shall be made within 30 days of invoice receipt",
            "references": ["Section 5.1"],
            "amendments": [{"date": "2024-02", "change": "Extended to 45 days"}],
        },
        {
            "document": "Contract-2024-001",
            "section": "3.1",
            "parent_sections": ["3"],
            "clause_type": "payment",
            "title": "Invoice Requirements",
            "text": "Invoices must include detailed itemization and project codes",
            "references": [],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "5",
            "parent_sections": [],
            "clause_type": "liability",
            "title": "Liability and Indemnification",
            "text": "The contractor shall indemnify the client against all claims",
            "references": ["Section 3.1"],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "5.1",
            "parent_sections": ["5"],
            "clause_type": "liability",
            "title": "Limitation of Liability",
            "text": "Total liability shall not exceed the total contract value",
            "references": [],
            "amendments": [{"date": "2024-03", "change": "Added liability cap"}],
        },
        {
            "document": "Contract-2024-001",
            "section": "5.2",
            "parent_sections": ["5"],
            "clause_type": "indemnification",
            "title": "Indemnification Scope",
            "text": "The party shall indemnify for damages arising from negligence",
            "references": ["Section 3.1", "Appendix A"],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "5.2.1",
            "parent_sections": ["5", "5.2"],
            "clause_type": "indemnification",
            "title": "Third Party Claims",
            "text": "Indemnification covers third party intellectual property claims",
            "references": [],
            "amendments": [],
        },
        {
            "document": "Contract-2024-001",
            "section": "6",
            "parent_sections": [],
            "clause_type": "termination",
            "title": "Termination",
            "text": "Either party may terminate with 90 days written notice",
            "references": ["Section 3"],
            "amendments": [],
        },
        {
            "document": "Contract-2024-002",
            "section": "1",
            "parent_sections": [],
            "clause_type": "definitions",
            "title": "Definitions",
            "text": "Terms and definitions applicable to this service agreement",
            "references": [],
            "amendments": [],
        },
        {
            "document": "Contract-2024-002",
            "section": "4",
            "parent_sections": [],
            "clause_type": "indemnification",
            "title": "Indemnification Provisions",
            "text": "Contractor indemnifies client for direct damages only",
            "references": ["Appendix A"],
            "amendments": [],
        },
    ]

    # Add unique IDs
    for section in sections:
        section["section_id"] = str(uuid.uuid4())

    return sections


class DocumentIndex:
    """Index for hierarchical document search."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.sections = {}

    def ingest_sections(self, sections: List[Dict[str, Any]]):
        """Ingest document sections."""
        print(f"📥 Ingesting {len(sections)} sections...")
        start = time.time()

        for section in sections:
            section_id = self.client.insert_json(section)
            self.sections[section["section_id"]] = section

        elapsed = time.time() - start
        rate = len(sections) / elapsed if elapsed > 0 else 0
        print(f"   ✅ Ingested in {elapsed:.2f}s ({rate:.0f}/sec)")

    def search_by_clause_type(
        self, clause_type: str, limit: int = 10
    ) -> List[Dict]:
        """Search for clauses of a specific type."""
        return self.client.search_json(
            probe={"clause_type": clause_type}, limit=limit
        )

    def search_excluding_section(
        self, clause_type: str, exclude_section: str, limit: int = 10
    ) -> List[Dict]:
        """Search for clauses NOT in a specific section."""
        # Use guard with negation
        results = []
        all_matches = self.client.search_json(
            probe={"clause_type": clause_type}, limit=100
        )

        for result in all_matches:
            section = result["data"]["section"]
            # Exclude if section starts with the excluded section
            if not section.startswith(exclude_section):
                results.append(result)

        return results[:limit]

    def search_with_references(
        self, reference: str, limit: int = 10
    ) -> List[Dict]:
        """Find all clauses that reference a specific item."""
        # This searches for sections that contain the reference
        results = self.client.search_json(probe={"references": [reference]}, limit=limit)
        return results

    def search_with_amendments(self, limit: int = 10) -> List[Dict]:
        """Find all clauses with amendments."""
        # Search for sections that have non-empty amendments
        all_sections = self.client.search_json(probe={}, limit=100)
        results = [
            r for r in all_sections if r["data"].get("amendments")
        ]
        return results[:limit]

    def search_by_hierarchy(
        self, parent_section: str, limit: int = 10
    ) -> List[Dict]:
        """Find all child sections of a parent."""
        results = self.client.search_json(
            probe={"parent_sections": [parent_section]}, limit=limit
        )
        return results


def demo_basic_search(index: DocumentIndex):
    """Demo 1: Basic clause search."""
    print("\n" + "=" * 70)
    print("DEMO 1: Search by Clause Type")
    print("=" * 70)

    print("\n🔍 Searching for indemnification clauses...")
    results = index.search_by_clause_type("indemnification", limit=10)
    print(f"   Found {len(results)} indemnification clauses:")
    for r in results:
        data = r["data"]
        print(f"   - §{data['section']}: {data['title']} (score: {r['score']:.3f})")


def demo_exclusion_search(index: DocumentIndex):
    """Demo 2: Search with negation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Search with Exclusion (NOT in Section 5)")
    print("=" * 70)

    print("\n🔍 Searching for indemnification clauses NOT in Section 5...")
    results = index.search_excluding_section("indemnification", "5", limit=10)
    print(f"   Found {len(results)} clauses:")
    for r in results:
        data = r["data"]
        print(f"   - §{data['section']}: {data['title']} (score: {r['score']:.3f})")


def demo_reference_search(index: DocumentIndex):
    """Demo 3: Search by references."""
    print("\n" + "=" * 70)
    print("DEMO 3: Find Clauses Referencing Appendix A")
    print("=" * 70)

    print("\n🔍 Searching for clauses that reference Appendix A...")
    results = index.search_with_references("Appendix A", limit=10)
    print(f"   Found {len(results)} clauses:")
    for r in results:
        data = r["data"]
        refs = ", ".join(data.get("references", []))
        print(f"   - §{data['section']}: {data['title']} → {refs} (score: {r['score']:.3f})")


def demo_amendment_search(index: DocumentIndex):
    """Demo 4: Find amended clauses."""
    print("\n" + "=" * 70)
    print("DEMO 4: Find Clauses with Amendments")
    print("=" * 70)

    print("\n🔍 Searching for clauses with amendments...")
    results = index.search_with_amendments(limit=10)
    print(f"   Found {len(results)} amended clauses:")
    for r in results:
        data = r["data"]
        amendments = data.get("amendments", [])
        if amendments:
            latest = amendments[0]
            print(
                f"   - §{data['section']}: {latest['date']} - {latest['change']} (score: {r['score']:.3f})"
            )


def demo_hierarchy_search(index: DocumentIndex):
    """Demo 5: Search by hierarchy."""
    print("\n" + "=" * 70)
    print("DEMO 5: Find Child Sections")
    print("=" * 70)

    print("\n🔍 Searching for subsections of Section 5...")
    results = index.search_by_hierarchy("5", limit=10)
    print(f"   Found {len(results)} subsections:")
    for r in results:
        data = r["data"]
        parents = " → ".join(data.get("parent_sections", []))
        print(f"   - §{data['section']} (parent: {parents}) (score: {r['score']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Document Retrieval")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print("=" * 70)
    print("HIERARCHICAL DOCUMENT RETRIEVAL")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    start_time = time.time()

    # Create index
    index = DocumentIndex(use_http=args.http, base_url=args.url)

    # Generate and ingest sections
    sections = generate_contract_sections()
    index.ingest_sections(sections)

    # Run demos
    demo_basic_search(index)
    demo_exclusion_search(index)
    demo_reference_search(index)
    demo_amendment_search(index)
    demo_hierarchy_search(index)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Sections Indexed: {len(sections)}

    ✅ Hierarchical document retrieval demonstrates:
       - Clause type search
       - Negation filters (NOT in section X)
       - Cross-reference tracking
       - Amendment history search
       - Parent-child hierarchy navigation

    This enables sophisticated legal/technical document search
    with structural awareness and relationship tracking!
    """
    )


if __name__ == "__main__":
    main()
