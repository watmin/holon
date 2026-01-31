#!/usr/bin/env python3
"""
Challenge 006-003: User State Mirror - Adaptive Personalization

Demonstrates Holon's ability to track user state (energy, focus, style
preferences) and adapt responses accordingly.

This is an IDEAL use case for Holon:
- Lightweight state records with temporal evolution
- Query most recent state for response tuning
- Track patterns over time (energy arc, focus shifts)
- Fuzzy matching of user signals

Run: ./scripts/run_with_venv.sh python scripts/challenges/006-batch/003-solution.py
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from holon import HolonClient
from holon.cpu_store import CPUStore


def create_user_state(
    user_id: str,
    energy_level: str,
    current_focus: str,
    preferred_style: str,
    excitement_triggers: Optional[List[str]] = None,
    frustration_triggers: Optional[List[str]] = None,
    recent_context: Optional[str] = None
) -> dict:
    """Create a user state snapshot."""
    return {
        "user_id": user_id,
        "energy_level": energy_level,  # high, medium, low
        "current_focus": current_focus,
        "preferred_style": preferred_style,  # bold, pragmatic, technical-deep, casual
        "excitement_triggers": excitement_triggers or [],
        "frustration_triggers": frustration_triggers or [],
        "recent_context": recent_context,
        "timestamp": datetime.now().isoformat(),
        "record_type": "user_state"
    }


def parse_result_data(data) -> dict:
    """Parse result data which may be a string or dict."""
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {}


def simulate_session_evolution(client: HolonClient):
    """Simulate a session with evolving user state."""
    print("\n" + "="*60)
    print("SIMULATING SESSION EVOLUTION")
    print("="*60)

    user_id = "watmin"
    states = []

    # State 1: Session start - exploring
    print("\n[Turn 1] User starts exploring a new problem...")
    state1 = create_user_state(
        user_id=user_id,
        energy_level="medium",
        current_focus="exploring batch 006 challenges",
        preferred_style="pragmatic",
        excitement_triggers=["new challenges", "well-suited problems"],
        frustration_triggers=["scope creep", "broken APIs"],
        recent_context="Just finished doc cleanup, feeling organized"
    )
    states.append(state1)
    client.insert_json(state1)
    print(f"  Energy: {state1['energy_level']} | Focus: {state1['current_focus']}")

    time.sleep(0.1)  # Small delay to ensure timestamp ordering

    # State 2: Getting excited about good fit
    print("\n[Turn 3] User sees these are good Holon problems...")
    state2 = create_user_state(
        user_id=user_id,
        energy_level="high",
        current_focus="batch 006 memory challenges",
        preferred_style="bold",
        excitement_triggers=["Holon sweet spot", "structured data", "fuzzy retrieval"],
        frustration_triggers=["np-hard delusions"],
        recent_context="Realized 006 is ideal for Holon - not NP-hard nonsense"
    )
    states.append(state2)
    client.insert_json(state2)
    print(f"  Energy: {state2['energy_level']} | Focus: {state2['current_focus']}")

    time.sleep(0.1)

    # State 3: Deep in implementation
    print("\n[Turn 8] Deep in implementation, focused flow...")
    state3 = create_user_state(
        user_id=user_id,
        energy_level="high",
        current_focus="implementing 006-001 persistent collaborator",
        preferred_style="technical-deep",
        excitement_triggers=["working code", "good patterns"],
        frustration_triggers=["type errors", "API mismatches"],
        recent_context="Building memory record system, seeing good results"
    )
    states.append(state3)
    client.insert_json(state3)
    print(f"  Energy: {state3['energy_level']} | Focus: {state3['current_focus']}")

    time.sleep(0.1)

    # State 4: Hit a snag
    print("\n[Turn 12] Hit a minor issue, slight frustration...")
    state4 = create_user_state(
        user_id=user_id,
        energy_level="medium",
        current_focus="debugging JSON parsing",
        preferred_style="pragmatic",
        excitement_triggers=["quick fixes"],
        frustration_triggers=["string vs dict confusion", "inconsistent returns"],
        recent_context="parse_result_data helper needed again"
    )
    states.append(state4)
    client.insert_json(state4)
    print(f"  Energy: {state4['energy_level']} | Focus: {state4['current_focus']}")

    time.sleep(0.1)

    # State 5: Back on track
    print("\n[Turn 15] Issue resolved, momentum restored...")
    state5 = create_user_state(
        user_id=user_id,
        energy_level="high",
        current_focus="completing all 006 challenges",
        preferred_style="bold",
        excitement_triggers=["finishing strong", "clean solutions"],
        frustration_triggers=[],
        recent_context="All 4 challenges looking great, ideal Holon use cases"
    )
    states.append(state5)
    client.insert_json(state5)
    print(f"  Energy: {state5['energy_level']} | Focus: {state5['current_focus']}")

    print(f"\nRecorded {len(states)} state snapshots.")
    return states


def query_current_state(client: HolonClient, user_id: str = "watmin"):
    """Query the most recent user state."""
    print("\n" + "="*60)
    print("QUERYING CURRENT USER STATE")
    print("="*60)

    results = client.search_json(
        probe={"user_id": user_id, "record_type": "user_state"},
        limit=1
    )

    if results:
        data = parse_result_data(results[0]['data'])
        print(f"\nCurrent state for {user_id}:")
        print(f"  Energy Level: {data.get('energy_level', 'unknown')}")
        print(f"  Current Focus: {data.get('current_focus', 'unknown')}")
        print(f"  Preferred Style: {data.get('preferred_style', 'unknown')}")
        print(f"  Context: {data.get('recent_context', 'none')}")

        triggers = data.get('excitement_triggers', [])
        if triggers:
            print(f"  Excitement Triggers: {', '.join(triggers)}")

        return data
    else:
        print("No state found for user.")
        return None


def adapt_response(state: dict) -> dict:
    """Generate response adaptations based on user state."""
    print("\n" + "="*60)
    print("RESPONSE ADAPTATIONS")
    print("="*60)

    adaptations = {
        "tone": "neutral",
        "detail_level": "medium",
        "emoji_density": "none",
        "suggestion_style": "balanced",
        "encouragement": False
    }

    energy = state.get('energy_level', 'medium')
    style = state.get('preferred_style', 'pragmatic')

    # Adapt based on energy
    if energy == "high":
        adaptations["tone"] = "enthusiastic"
        adaptations["emoji_density"] = "light"
        adaptations["encouragement"] = True
        adaptations["suggestion_style"] = "bold"
    elif energy == "low":
        adaptations["tone"] = "supportive"
        adaptations["detail_level"] = "concise"
        adaptations["suggestion_style"] = "gentle"

    # Adapt based on preferred style
    if style == "bold":
        adaptations["suggestion_style"] = "aggressive"
        adaptations["tone"] = "direct"
    elif style == "technical-deep":
        adaptations["detail_level"] = "comprehensive"
    elif style == "casual":
        adaptations["emoji_density"] = "medium"
        adaptations["tone"] = "friendly"

    print(f"\nBased on state (energy={energy}, style={style}):")
    for key, value in adaptations.items():
        print(f"  {key}: {value}")

    # Generate sample response variations
    print("\n--- Sample Response Variations ---")

    base_message = "The persistent collaborator solution is working well."

    if adaptations["tone"] == "enthusiastic":
        print(f"  Enthusiastic: 'The persistent collaborator is working great! This is exactly what Holon excels at.'")
    elif adaptations["tone"] == "supportive":
        print(f"  Supportive: 'Good progress on the persistent collaborator. Take your time with the details.'")
    else:
        print(f"  Neutral: '{base_message}'")

    return adaptations


def analyze_energy_arc(client: HolonClient, user_id: str = "watmin"):
    """Analyze the energy arc over the session."""
    print("\n" + "="*60)
    print("ENERGY ARC ANALYSIS")
    print("="*60)

    results = client.search_json(
        probe={"user_id": user_id, "record_type": "user_state"},
        limit=20
    )

    energy_map = {"high": 3, "medium": 2, "low": 1}
    energy_sequence = []

    for r in results:
        data = parse_result_data(r['data'])
        energy = data.get('energy_level', 'medium')
        focus = data.get('current_focus', '')[:30]
        energy_sequence.append((energy, focus))

    print("\nSession Energy Timeline:")
    for i, (energy, focus) in enumerate(reversed(energy_sequence)):
        bar = "=" * (energy_map.get(energy, 2) * 5)
        print(f"  [{i+1}] {bar} {energy:6} | {focus}...")

    # Calculate trends
    if len(energy_sequence) >= 2:
        start_energy = energy_map.get(energy_sequence[-1][0], 2)
        end_energy = energy_map.get(energy_sequence[0][0], 2)
        trend = end_energy - start_energy

        if trend > 0:
            print(f"\nTrend: RISING (started at {energy_sequence[-1][0]}, now at {energy_sequence[0][0]})")
        elif trend < 0:
            print(f"\nTrend: FALLING (started at {energy_sequence[-1][0]}, now at {energy_sequence[0][0]})")
        else:
            print(f"\nTrend: STABLE (maintaining {energy_sequence[0][0]})")


def main():
    print("="*60)
    print("Challenge 006-003: User State Mirror")
    print("Adaptive Personalization with Holon")
    print("="*60)

    # Initialize client with local store
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Simulate session with evolving state
    states = simulate_session_evolution(client)

    # Query current state
    current_state = query_current_state(client)

    # Show how we'd adapt responses
    if current_state:
        adapt_response(current_state)

    # Analyze energy arc
    analyze_energy_arc(client)

    # Summary
    print("\n" + "="*60)
    print("CHALLENGE 006-003: COMPLETE")
    print("="*60)
    print("""
Key Demonstrations:
1. Lightweight user state snapshots
2. Temporal evolution tracking
3. Most-recent state retrieval
4. Response adaptation based on state
5. Energy arc analysis over session

Benefits:
- Responses auto-tune to user mood/energy
- Continuity across sessions (remembers preferences)
- Pattern detection (energy trends, frustration points)
- Makes AI feel like it "knows" the user

This is fuzzy retrieval at its best - no exact matching needed,
just "find the most relevant state for this user."
""")


if __name__ == "__main__":
    main()
