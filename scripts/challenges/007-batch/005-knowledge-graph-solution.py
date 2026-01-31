#!/usr/bin/env python3
"""
Challenge 007-005: Knowledge Graph Fragment Matching

Demonstrates matching partial subgraph patterns against a knowledge base.
Features:
- Entity types and relations
- Neighbor relationships
- Relational structure encoding
- Partial pattern matching

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/005-knowledge-graph-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/005-knowledge-graph-solution.py --http
"""

import argparse
import time
import uuid
from typing import Any, Dict, List

from holon import CPUStore, HolonClient


def generate_programming_language_graph() -> List[Dict[str, Any]]:
    """Generate knowledge graph about programming languages."""
    entities = [
        {
            "entity": "Python",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "Guido van Rossum",
                "influenced_by": ["ABC", "C", "Lisp"],
                "used_for": ["web", "ml", "scripting"],
                "paradigm": ["object-oriented", "imperative", "functional"],
            },
            "neighbors": {
                "similar_to": ["Ruby", "JavaScript"],
                "competes_with": ["Java", "Go"],
            },
            "metadata": {"first_appeared": 1991, "typing": "dynamic"},
        },
        {
            "entity": "Java",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "James Gosling",
                "influenced_by": ["C++", "Smalltalk"],
                "used_for": ["enterprise", "android", "web"],
                "paradigm": ["object-oriented", "imperative"],
            },
            "neighbors": {
                "similar_to": ["C#", "Kotlin"],
                "competes_with": ["Python", "Go"],
            },
            "metadata": {"first_appeared": 1995, "typing": "static"},
        },
        {
            "entity": "JavaScript",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "Brendan Eich",
                "influenced_by": ["Self", "Scheme", "Java"],
                "used_for": ["web", "frontend", "backend"],
                "paradigm": ["object-oriented", "imperative", "functional"],
            },
            "neighbors": {
                "similar_to": ["Python", "TypeScript"],
                "competes_with": ["Dart", "CoffeeScript"],
            },
            "metadata": {"first_appeared": 1995, "typing": "dynamic"},
        },
        {
            "entity": "Ruby",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "Yukihiro Matsumoto",
                "influenced_by": ["Perl", "Smalltalk", "Lisp"],
                "used_for": ["web", "scripting"],
                "paradigm": ["object-oriented", "imperative", "functional"],
            },
            "neighbors": {
                "similar_to": ["Python", "Perl"],
                "competes_with": ["PHP", "Python"],
            },
            "metadata": {"first_appeared": 1995, "typing": "dynamic"},
        },
        {
            "entity": "Go",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "Google",
                "influenced_by": ["C", "Pascal", "CSP"],
                "used_for": ["systems", "web", "cloud"],
                "paradigm": ["imperative", "concurrent"],
            },
            "neighbors": {
                "similar_to": ["Rust", "C"],
                "competes_with": ["Python", "Java"],
            },
            "metadata": {"first_appeared": 2009, "typing": "static"},
        },
        {
            "entity": "Rust",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "Mozilla",
                "influenced_by": ["C++", "ML", "Haskell"],
                "used_for": ["systems", "embedded", "web"],
                "paradigm": ["imperative", "functional"],
            },
            "neighbors": {
                "similar_to": ["C++", "Go"],
                "competes_with": ["C++", "D"],
            },
            "metadata": {"first_appeared": 2010, "typing": "static"},
        },
        {
            "entity": "Lisp",
            "type": "ProgrammingLanguage",
            "relations": {
                "created_by": "John McCarthy",
                "influenced_by": [],
                "used_for": ["ai", "research", "scripting"],
                "paradigm": ["functional"],
            },
            "neighbors": {
                "similar_to": ["Scheme", "Clojure"],
                "competes_with": [],
            },
            "metadata": {"first_appeared": 1958, "typing": "dynamic"},
        },
    ]

    # Add unique IDs
    for entity in entities:
        entity["entity_id"] = str(uuid.uuid4())

    return entities


class KnowledgeGraphIndex:
    """Index for knowledge graph search."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.entities = {}

    def ingest_entities(self, entities: List[Dict[str, Any]]):
        """Ingest entities into the knowledge graph."""
        print(f"📥 Ingesting {len(entities)} entities...")
        start = time.time()

        for entity in entities:
            entity_id = self.client.insert_json(entity)
            self.entities[entity["entity_id"]] = entity

        elapsed = time.time() - start
        rate = len(entities) / elapsed if elapsed > 0 else 0
        print(f"   ✅ Ingested in {elapsed:.2f}s ({rate:.0f}/sec)")

    def search_by_type(self, entity_type: str, limit: int = 10) -> List[Dict]:
        """Search for entities of a specific type."""
        return self.client.search_json(probe={"type": entity_type}, limit=limit)

    def search_by_relation(
        self, relation_key: str, relation_value: str, limit: int = 10
    ) -> List[Dict]:
        """Search for entities with a specific relation."""
        probe = {"relations": {relation_key: relation_value}}
        return self.client.search_json(probe=probe, limit=limit)

    def search_influenced_by(self, language: str, limit: int = 10) -> List[Dict]:
        """Find languages influenced by a specific language."""
        probe = {"relations": {"influenced_by": [language]}}
        return self.client.search_json(probe=probe, limit=limit)

    def search_similar_languages(
        self, language: str, limit: int = 10
    ) -> List[Dict]:
        """Find languages similar to a specific language."""
        probe = {"neighbors": {"similar_to": [language]}}
        return self.client.search_json(probe=probe, limit=limit)

    def search_by_paradigm(self, paradigm: str, limit: int = 10) -> List[Dict]:
        """Search for languages supporting a paradigm."""
        probe = {"relations": {"paradigm": [paradigm]}}
        return self.client.search_json(probe=probe, limit=limit)

    def search_by_use_case(self, use_case: str, limit: int = 10) -> List[Dict]:
        """Search for languages used for a specific purpose."""
        probe = {"relations": {"used_for": [use_case]}}
        return self.client.search_json(probe=probe, limit=limit)


def demo_type_search(index: KnowledgeGraphIndex):
    """Demo 1: Search by entity type."""
    print("\n" + "=" * 70)
    print("DEMO 1: Search by Entity Type")
    print("=" * 70)

    print("\n🔍 Searching for programming languages...")
    results = index.search_by_type("ProgrammingLanguage", limit=10)
    print(f"   Found {len(results)} programming languages:")
    for r in results[:5]:
        data = r["data"]
        print(f"   - {data['entity']} (created: {data['metadata']['first_appeared']})")


def demo_influence_search(index: KnowledgeGraphIndex):
    """Demo 2: Search by influence."""
    print("\n" + "=" * 70)
    print("DEMO 2: Find Languages Influenced by Lisp")
    print("=" * 70)

    print("\n🔍 Searching for languages influenced by Lisp...")
    results = index.search_influenced_by("Lisp", limit=10)
    print(f"   Found {len(results)} languages:")
    for r in results:
        data = r["data"]
        influences = data["relations"].get("influenced_by", [])
        print(
            f"   - {data['entity']}: influenced by {', '.join(influences)} (score: {r['score']:.3f})"
        )


def demo_similarity_search(index: KnowledgeGraphIndex):
    """Demo 3: Search by similarity."""
    print("\n" + "=" * 70)
    print("DEMO 3: Find Languages Similar to Ruby")
    print("=" * 70)

    print("\n🔍 Searching for languages similar to Ruby...")
    results = index.search_similar_languages("Ruby", limit=10)
    print(f"   Found {len(results)} similar languages:")
    for r in results:
        data = r["data"]
        similar = data["neighbors"].get("similar_to", [])
        print(
            f"   - {data['entity']}: similar to {', '.join(similar)} (score: {r['score']:.3f})"
        )


def demo_paradigm_search(index: KnowledgeGraphIndex):
    """Demo 4: Search by paradigm."""
    print("\n" + "=" * 70)
    print("DEMO 4: Find Functional Programming Languages")
    print("=" * 70)

    print("\n🔍 Searching for functional languages...")
    results = index.search_by_paradigm("functional", limit=10)
    print(f"   Found {len(results)} functional languages:")
    for r in results:
        data = r["data"]
        paradigms = data["relations"].get("paradigm", [])
        print(
            f"   - {data['entity']}: {', '.join(paradigms)} (score: {r['score']:.3f})"
        )


def demo_use_case_search(index: KnowledgeGraphIndex):
    """Demo 5: Search by use case."""
    print("\n" + "=" * 70)
    print("DEMO 5: Find Languages for Web Development")
    print("=" * 70)

    print("\n🔍 Searching for web development languages...")
    results = index.search_by_use_case("web", limit=10)
    print(f"   Found {len(results)} web languages:")
    for r in results:
        data = r["data"]
        uses = data["relations"].get("used_for", [])
        print(f"   - {data['entity']}: used for {', '.join(uses)} (score: {r['score']:.3f})")


def demo_complex_search(index: KnowledgeGraphIndex):
    """Demo 6: Complex pattern search."""
    print("\n" + "=" * 70)
    print("DEMO 6: Complex Pattern - Dynamic + Functional + Web")
    print("=" * 70)

    print("\n🔍 Searching for dynamic, functional languages used for web...")
    # This is a fuzzy search that will match languages with these characteristics
    probe = {
        "metadata": {"typing": "dynamic"},
        "relations": {
            "paradigm": ["functional"],
            "used_for": ["web"],
        },
    }
    results = index.client.search_json(probe=probe, limit=10)

    print(f"   Found {len(results)} matching languages:")
    for r in results:
        data = r["data"]
        print(
            f"   - {data['entity']}: {data['metadata']['typing']} typing (score: {r['score']:.3f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Fragment Matching")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print("=" * 70)
    print("KNOWLEDGE GRAPH FRAGMENT MATCHING")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    start_time = time.time()

    # Create index
    index = KnowledgeGraphIndex(use_http=args.http, base_url=args.url)

    # Generate and ingest entities
    entities = generate_programming_language_graph()
    index.ingest_entities(entities)

    # Run demos
    demo_type_search(index)
    demo_influence_search(index)
    demo_similarity_search(index)
    demo_paradigm_search(index)
    demo_use_case_search(index)
    demo_complex_search(index)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Entities Indexed: {len(entities)}

    ✅ Knowledge graph fragment matching demonstrates:
       - Entity type search
       - Relation-based queries (influenced_by, used_for)
       - Neighbor similarity search
       - Paradigm and use case filtering
       - Complex multi-faceted pattern matching

    This enables querying graph structures without
    explicit graph traversal or query languages!
    """
    )


if __name__ == "__main__":
    main()
