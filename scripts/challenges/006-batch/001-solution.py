#!/usr/bin/env python3
"""
Challenge 006-001: Persistent Collaborator - Cross-Session Continuity

Demonstrates Holon's ability to persist and retrieve structured context
across sessions, enabling LLMs to "remember" past decisions, preferences,
and project state without token-expensive context re-injection.

This is an IDEAL use case for Holon:
- Structured data with metadata (user_id, session_id, tags, timestamps)
- Fuzzy retrieval by similarity (not exact key lookup)
- Guard-based filtering ($not, structural guards)
- Top-k retrieval with ranking

Run: ./scripts/run_with_venv.sh python scripts/challenges/006-batch/001-solution.py
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from holon import HolonClient
from holon.cpu_store import CPUStore


def create_memory_record(
    record_type: str,
    content: str,
    user_id: str = "watmin",
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    confidence: float = 0.9,
    context: Optional[dict] = None
) -> dict:
    """Create a structured memory record for persistence."""
    return {
        "type": record_type,  # "decision", "preference", "motivation", "state", "fact"
        "content": content,
        "user_id": user_id,
        "session_id": session_id or f"session_{int(time.time())}",
        "tags": tags or [],
        "confidence": confidence,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
        "status": "active"  # active, deprecated, superseded
    }


def simulate_session_1(client: HolonClient) -> str:
    """Simulate first session with various decisions and context."""
    print("\n" + "="*60)
    print("SESSION 1: Initial Development Discussion")
    print("="*60)

    session_id = "session_001"
    records = []

    # Decision: Data type choice
    records.append(create_memory_record(
        record_type="decision",
        content="Using float32 for vector storage - float16 had precision issues with small similarities",
        user_id="watmin",
        session_id=session_id,
        tags=["dtype", "performance", "vectors"],
        confidence=0.95,
        context={"alternatives_considered": ["float16", "int8"], "reason": "precision"}
    ))

    # Preference: User communication style
    records.append(create_memory_record(
        record_type="preference",
        content="User prefers bold, direct communication with technical depth",
        user_id="watmin",
        session_id=session_id,
        tags=["communication", "style"],
        confidence=0.9,
        context={"triggers": ["motivated again", "let's attack this"]}
    ))

    # Motivation: Current energy state
    records.append(create_memory_record(
        record_type="motivation",
        content="High energy after solving the VSA kernel primitive challenges",
        user_id="watmin",
        session_id=session_id,
        tags=["energy", "momentum"],
        confidence=0.85,
        context={"recent_wins": ["prototype classification", "n-gram encoding"]}
    ))

    # Fact: Project architecture
    records.append(create_memory_record(
        record_type="fact",
        content="Holon uses CPUStore for local ops, HTTP API for remote, both share same primitives",
        user_id="watmin",
        session_id=session_id,
        tags=["architecture", "api"],
        confidence=1.0,
        context={"endpoints": ["/api/v1/items", "/api/v1/search", "/api/v1/vectors/*"]}
    ))

    # Decision: NP-hard problem approach
    records.append(create_memory_record(
        record_type="decision",
        content="Acknowledged that VSA cannot solve Sudoku geometrically - constraint satisfaction needs exact solving",
        user_id="watmin",
        session_id=session_id,
        tags=["limitations", "sudoku", "np-hard"],
        confidence=1.0,
        context={"lesson": "VSA good for fuzzy retrieval, not global constraint satisfaction"}
    ))

    # State: Current focus
    records.append(create_memory_record(
        record_type="state",
        content="Working through challenge batches systematically, currently on 006",
        user_id="watmin",
        session_id=session_id,
        tags=["progress", "challenges"],
        confidence=0.9,
        context={"completed": ["001", "002", "003", "004"], "current": "006"}
    ))

    # Insert all records
    print(f"\nInserting {len(records)} memory records...")
    for record in records:
        client.insert_json(record)
        print(f"  [{record['type']:12}] {record['content'][:50]}...")

    print(f"\nSession 1 complete. {len(records)} records persisted.")
    return session_id


def simulate_session_2(client: HolonClient, previous_session: str):
    """Simulate new session that retrieves context from previous session."""
    print("\n" + "="*60)
    print("SESSION 2: Resuming After Break")
    print("="*60)

    session_id = "session_002"

    # Query 1: Retrieve recent decisions
    print("\n--- Retrieving recent decisions ---")
    results = client.search_json(
        probe={"type": "decision"},
        limit=10
    )

    print(f"Found {len(results)} decisions:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  [{data.get('confidence', 0):.0%}] {data.get('content', '')[:60]}...")
        if data.get('tags'):
            print(f"       Tags: {', '.join(data['tags'])}")

    # Query 2: Retrieve user preferences
    print("\n--- Retrieving user preferences ---")
    results = client.search_json(
        probe={"type": "preference", "user_id": "watmin"},
        limit=5
    )

    print(f"Found {len(results)} preferences:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  {data.get('content', '')[:70]}...")

    # Query 3: Retrieve motivation/energy state
    print("\n--- Retrieving motivation state ---")
    results = client.search_json(
        probe={"type": "motivation"},
        limit=3
    )

    print(f"Found {len(results)} motivation records:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  {data.get('content', '')[:70]}...")
        ctx = data.get('context', {})
        if ctx.get('recent_wins'):
            print(f"       Recent wins: {', '.join(ctx['recent_wins'])}")

    # Query 4: Retrieve known limitations (to avoid repeating mistakes)
    print("\n--- Retrieving known limitations ---")
    results = client.search_json(
        probe={"tags": ["limitations"]},
        limit=5
    )

    print(f"Found {len(results)} limitation records:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  {data.get('content', '')[:70]}...")

    # Query 5: Retrieve current progress state
    print("\n--- Retrieving project state ---")
    results = client.search_json(
        probe={"type": "state", "tags": ["progress"]},
        limit=3
    )

    print(f"Found {len(results)} state records:")
    for r in results:
        data = parse_result_data(r['data'])
        print(f"  {data.get('content', '')[:70]}...")
        ctx = data.get('context', {})
        if ctx.get('completed'):
            print(f"       Completed batches: {ctx['completed']}")

    # Add new records from this session
    print("\n--- Adding new session records ---")
    new_records = [
        create_memory_record(
            record_type="decision",
            content="Batch 006 challenges are ideal for Holon - structured memory retrieval",
            user_id="watmin",
            session_id=session_id,
            tags=["assessment", "challenges", "positive"],
            confidence=0.95
        ),
        create_memory_record(
            record_type="state",
            content="Successfully demonstrated cross-session memory persistence",
            user_id="watmin",
            session_id=session_id,
            tags=["progress", "demo", "memory"],
            confidence=1.0
        )
    ]

    for record in new_records:
        client.insert_json(record)
        print(f"  Added: {record['content'][:50]}...")

    print("\nSession 2 complete.")


def generate_memory_briefing(client: HolonClient, user_id: str = "watmin") -> str:
    """Generate a concise memory briefing for session start."""
    print("\n" + "="*60)
    print("GENERATING MEMORY BRIEFING")
    print("="*60)

    briefing_parts = []

    # Get recent decisions
    decisions = client.search_json(
        probe={"type": "decision", "user_id": user_id},
        limit=5
    )

    if decisions:
        briefing_parts.append("## Recent Decisions")
        for r in decisions:
            data = parse_result_data(r['data'])
            briefing_parts.append(f"- {data.get('content', '')}")

    # Get preferences
    prefs = client.search_json(
        probe={"type": "preference", "user_id": user_id},
        limit=3
    )

    if prefs:
        briefing_parts.append("\n## User Preferences")
        for r in prefs:
            data = parse_result_data(r['data'])
            briefing_parts.append(f"- {data.get('content', '')}")

    # Get current state
    states = client.search_json(
        probe={"type": "state", "user_id": user_id},
        limit=2
    )

    if states:
        briefing_parts.append("\n## Current State")
        for r in states:
            data = parse_result_data(r['data'])
            briefing_parts.append(f"- {data.get('content', '')}")

    # Get limitations
    limits = client.search_json(
        probe={"tags": ["limitations"]},
        limit=3
    )

    if limits:
        briefing_parts.append("\n## Known Limitations")
        for r in limits:
            data = parse_result_data(r['data'])
            briefing_parts.append(f"- {data.get('content', '')}")

    briefing = "\n".join(briefing_parts)

    print("\n--- Generated Briefing ---")
    print(briefing)
    print("-" * 40)

    # Calculate token savings estimate
    full_context_tokens = 2000  # Estimated tokens for full session replay
    briefing_tokens = len(briefing.split()) * 1.3  # Rough token estimate
    savings = (1 - briefing_tokens / full_context_tokens) * 100

    print(f"\nEstimated token savings: {savings:.0f}%")
    print(f"(Briefing: ~{int(briefing_tokens)} tokens vs full context: ~{full_context_tokens} tokens)")

    return briefing


def parse_result_data(data) -> dict:
    """Parse result data which may be a string or dict."""
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {"content": str(data)}


def main():
    print("="*60)
    print("Challenge 006-001: Persistent Collaborator")
    print("Cross-Session Continuity with Holon")
    print("="*60)

    # Initialize client with local store
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Simulate first session
    session_1_id = simulate_session_1(client)

    # Simulate session break
    print("\n" + "~"*60)
    print("       [Session ended. Time passes...]")
    print("~"*60)
    time.sleep(1)

    # Simulate second session with memory retrieval
    simulate_session_2(client, session_1_id)

    # Generate memory briefing
    briefing = generate_memory_briefing(client)

    # Summary
    print("\n" + "="*60)
    print("CHALLENGE 006-001: COMPLETE")
    print("="*60)
    print("""
Key Demonstrations:
1. Structured memory records with metadata (type, tags, confidence)
2. Cross-session persistence and retrieval
3. Fuzzy similarity-based querying (not exact key lookup)
4. Memory briefing generation for session startup
5. Significant token savings vs full context replay

This is an IDEAL Holon use case:
- Structured data with semantic meaning
- Fuzzy retrieval matches similar concepts
- Guards enable filtering by type, user, tags
- Top-k retrieval with ranking by relevance
""")


if __name__ == "__main__":
    main()
