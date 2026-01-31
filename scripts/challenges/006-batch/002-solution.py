#!/usr/bin/env python3
"""
Challenge 006-002: Hypothesis Garden - Parallel Thought Superposition

Demonstrates Holon's ability to maintain multiple competing hypotheses
in parallel, enabling structured exploration without prompt bloat.

This is an IDEAL use case for Holon:
- Track multiple competing ideas with pros/cons/evidence
- Fuzzy retrieval by concept similarity
- Filter by status, confidence, tags
- Prototype-based synthesis of related hypotheses

Run: ./scripts/run_with_venv.sh python scripts/challenges/006-batch/002-solution.py
"""

import json
import time
from datetime import datetime
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from holon import HolonClient
from holon.cpu_store import CPUStore


def create_hypothesis(
    hypothesis_id: str,
    description: str,
    pros: List[str],
    cons: List[str],
    evidence: Optional[List[str]] = None,
    confidence: float = 0.5,
    tags: Optional[List[str]] = None,
    status: str = "active"
) -> dict:
    """Create a structured hypothesis record."""
    return {
        "id": hypothesis_id,
        "description": description,
        "pros": pros,
        "cons": cons,
        "evidence": evidence or [],
        "confidence": confidence,
        "tags": tags or [],
        "status": status,  # active, discarded, chosen, merged
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def parse_result_data(data) -> dict:
    """Parse result data which may be a string or dict."""
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {"description": str(data)}


def seed_hypothesis_garden(client: HolonClient):
    """Seed the garden with various hypotheses about a design decision."""
    print("\n" + "="*60)
    print("SEEDING HYPOTHESIS GARDEN")
    print("Topic: How to handle vector similarity computation")
    print("="*60)

    hypotheses = [
        create_hypothesis(
            hypothesis_id="hyp_001",
            description="Use cosine similarity for all comparisons",
            pros=[
                "Normalized, range [-1, 1]",
                "Well understood in ML",
                "Handles varying vector magnitudes"
            ],
            cons=[
                "Requires normalization step",
                "May lose magnitude information"
            ],
            evidence=["Works well in prototype classification demo"],
            confidence=0.75,
            tags=["similarity", "cosine", "normalization"],
            status="active"
        ),
        create_hypothesis(
            hypothesis_id="hyp_002",
            description="Use dot product for raw similarity scores",
            pros=[
                "Faster - no normalization needed",
                "Preserves magnitude information",
                "VSA theory uses raw dot products"
            ],
            cons=[
                "Unbounded range",
                "Hard to interpret across different vector sizes"
            ],
            evidence=["VSA literature recommends this"],
            confidence=0.70,
            tags=["similarity", "dot-product", "performance"],
            status="active"
        ),
        create_hypothesis(
            hypothesis_id="hyp_003",
            description="Use Euclidean distance for spatial similarity",
            pros=[
                "Intuitive geometric interpretation",
                "Good for clustering"
            ],
            cons=[
                "Inverse relationship to similarity",
                "Not ideal for high dimensions (curse of dimensionality)"
            ],
            evidence=["Works for low-dimensional embeddings"],
            confidence=0.40,
            tags=["similarity", "euclidean", "distance"],
            status="discarded"
        ),
        create_hypothesis(
            hypothesis_id="hyp_004",
            description="Hybrid: normalized dot product with magnitude weighting",
            pros=[
                "Best of both worlds",
                "Normalized but preserves strength signal",
                "Already implemented in Holon"
            ],
            cons=[
                "More complex",
                "Extra hyperparameter (weight factor)"
            ],
            evidence=["Current implementation uses this", "Good results in testing"],
            confidence=0.85,
            tags=["similarity", "hybrid", "implementation"],
            status="active"
        ),
        create_hypothesis(
            hypothesis_id="hyp_005",
            description="Use learned similarity function",
            pros=[
                "Could adapt to domain",
                "Potentially optimal"
            ],
            cons=[
                "Requires training data",
                "Adds complexity",
                "May not generalize"
            ],
            evidence=[],
            confidence=0.30,
            tags=["similarity", "learned", "ml", "future"],
            status="active"
        ),
        # Different topic: encoding strategies
        create_hypothesis(
            hypothesis_id="hyp_010",
            description="N-gram encoding for fuzzy text matching",
            pros=[
                "Handles typos and variations",
                "Works for substrings",
                "Proven in quote finder demo"
            ],
            cons=[
                "Loses word order for long texts",
                "N-gram size tuning needed"
            ],
            evidence=["100% accuracy in quote classification", "Works for fuzzy search"],
            confidence=0.90,
            tags=["encoding", "ngram", "text", "fuzzy"],
            status="chosen"
        ),
        create_hypothesis(
            hypothesis_id="hyp_011",
            description="Semantic encoding with transformer embeddings",
            pros=[
                "Captures meaning, not just surface form",
                "Good for paraphrase detection"
            ],
            cons=[
                "Requires external model",
                "Slower",
                "Black box"
            ],
            evidence=["Industry standard for NLP"],
            confidence=0.65,
            tags=["encoding", "semantic", "transformers", "future"],
            status="active"
        ),
        # Different topic: constraint handling
        create_hypothesis(
            hypothesis_id="hyp_020",
            description="VSA can solve constraint satisfaction geometrically",
            pros=[
                "Would be elegant",
                "Fast parallel solution"
            ],
            cons=[
                "Fundamentally impossible",
                "Local similarity != global coherence",
                "Proven false with Sudoku"
            ],
            evidence=["32+ approaches failed", "Theoretical analysis confirms limits"],
            confidence=0.0,
            tags=["constraints", "csp", "np-hard", "limitations"],
            status="discarded"
        ),
        create_hypothesis(
            hypothesis_id="hyp_021",
            description="Use VSA for heuristic guidance in traditional solvers",
            pros=[
                "Plays to VSA strengths (similarity)",
                "Traditional solver handles constraints",
                "Best of both worlds"
            ],
            cons=[
                "More complex architecture",
                "Needs integration work"
            ],
            evidence=["Proposed approach for constraint problems"],
            confidence=0.70,
            tags=["constraints", "hybrid", "solver", "future"],
            status="active"
        ),
    ]

    print(f"\nInserting {len(hypotheses)} hypotheses...")
    for hyp in hypotheses:
        client.insert_json(hyp)
        status_icon = {"active": "?", "discarded": "X", "chosen": "V", "merged": "~"}.get(hyp['status'], " ")
        print(f"  [{status_icon}] {hyp['id']}: {hyp['description'][:50]}...")

    print(f"\nGarden seeded with {len(hypotheses)} hypotheses.")
    return hypotheses


def explore_garden(client: HolonClient):
    """Demonstrate various query patterns on the hypothesis garden."""
    print("\n" + "="*60)
    print("EXPLORING HYPOTHESIS GARDEN")
    print("="*60)

    # Query 1: High-confidence active hypotheses
    print("\n--- Query: High-confidence active hypotheses ---")
    results = client.search_json(
        probe={"status": "active", "confidence": 0.7},
        limit=5
    )

    print(f"Found {len(results)} high-confidence active hypotheses:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  [{data.get('confidence', 0):.0%}] {data.get('description', '')[:55]}...")

    # Query 2: Hypotheses about similarity
    print("\n--- Query: Similarity-related hypotheses ---")
    results = client.search_json(
        probe={"tags": ["similarity"]},
        limit=10
    )

    print(f"Found {len(results)} similarity hypotheses:")
    for r in results:
        data = parse_result_data(r['data'])
        status = data.get('status', 'unknown')
        icon = {"active": "?", "discarded": "X", "chosen": "V"}.get(status, " ")
        print(f"  [{icon}] {data.get('description', '')[:60]}...")

    # Query 3: Discarded hypotheses (lessons learned)
    print("\n--- Query: Discarded hypotheses (lessons learned) ---")
    results = client.search_json(
        probe={"status": "discarded"},
        limit=5
    )

    print(f"Found {len(results)} discarded hypotheses:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  X {data.get('description', '')[:60]}...")
        cons = data.get('cons', [])
        if cons:
            print(f"    Reason: {cons[0]}")

    # Query 4: Chosen hypotheses (settled decisions)
    print("\n--- Query: Chosen hypotheses (settled decisions) ---")
    results = client.search_json(
        probe={"status": "chosen"},
        limit=5
    )

    print(f"Found {len(results)} chosen hypotheses:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  V {data.get('description', '')[:60]}...")
        evidence = data.get('evidence', [])
        if evidence:
            print(f"    Evidence: {evidence[0]}")

    # Query 5: Future/speculative hypotheses
    print("\n--- Query: Future/speculative ideas ---")
    results = client.search_json(
        probe={"tags": ["future"]},
        limit=5
    )

    print(f"Found {len(results)} future ideas:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  ? {data.get('description', '')[:60]}...")

    # Query 6: Hybrid approaches
    print("\n--- Query: Hybrid/combination approaches ---")
    results = client.search_json(
        probe={"tags": ["hybrid"]},
        limit=5
    )

    print(f"Found {len(results)} hybrid approaches:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  {data.get('description', '')[:65]}...")


def synthesize_insights(client: HolonClient):
    """Demonstrate synthesis across related hypotheses."""
    print("\n" + "="*60)
    print("SYNTHESIZING INSIGHTS")
    print("="*60)

    # Get all encoding-related hypotheses
    print("\n--- Synthesis: Encoding Strategy Summary ---")
    results = client.search_json(
        probe={"tags": ["encoding"]},
        limit=10
    )

    active_encodings = []
    chosen_encodings = []

    for r in results:
        data = parse_result_data(r['data'])
        if data.get('status') == 'chosen':
            chosen_encodings.append(data)
        elif data.get('status') == 'active':
            active_encodings.append(data)

    print("\nChosen encoding strategies:")
    for h in chosen_encodings:
        print(f"  V {h.get('description', '')}")
        for pro in h.get('pros', [])[:2]:
            print(f"    + {pro}")

    print("\nActive alternatives:")
    for h in active_encodings:
        print(f"  ? {h.get('description', '')} (confidence: {h.get('confidence', 0):.0%})")

    # Get constraint-related lessons
    print("\n--- Synthesis: Constraint Handling Lessons ---")
    results = client.search_json(
        probe={"tags": ["constraints"]},
        limit=10
    )

    print("\nKey learnings about constraints:")
    for r in results:
        data = parse_result_data(r['data'])
        status = data.get('status', '')
        if status == 'discarded':
            print(f"  X FAILED: {data.get('description', '')}")
        elif status == 'active':
            print(f"  ? TRY: {data.get('description', '')}")


def main():
    print("="*60)
    print("Challenge 006-002: Hypothesis Garden")
    print("Parallel Thought Superposition with Holon")
    print("="*60)

    # Initialize client with local store
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Seed the garden
    seed_hypothesis_garden(client)

    # Explore with various queries
    explore_garden(client)

    # Synthesize insights
    synthesize_insights(client)

    # Summary
    print("\n" + "="*60)
    print("CHALLENGE 006-002: COMPLETE")
    print("="*60)
    print("""
Key Demonstrations:
1. Structured hypothesis tracking (id, pros, cons, evidence, confidence)
2. Status management (active, discarded, chosen, merged)
3. Tag-based retrieval for topic exploration
4. Multi-criteria queries (status + confidence)
5. Cross-hypothesis synthesis

Advantages over linear CoT:
- Hold 20+ hypotheses without prompt bloat
- Instant recall: "what were the similarity arguments?"
- Status tracking prevents repeating discarded ideas
- Evidence accumulation across sessions
- Visual tree of thought possible via export
""")


if __name__ == "__main__":
    main()
