#!/usr/bin/env python3
"""
Challenge 007-001: Holon-Powered Rete Rule Engine

Demonstrates a rule engine that combines:
1. Exact matching (traditional Rete) for precise conditions
2. Fuzzy matching (Holon) for similarity-based conditions
3. k-NN style prototype matching (HTTP-compatible)
4. Multi-condition rules with joins
5. Truth maintenance system

All operations use Holon's search API - no local vector operations.
Works identically in local and HTTP modes.

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/001-rete-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/001-rete-solution.py --http
"""

import argparse
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from holon import CPUStore, HolonClient


def parse_data(data):
    """Parse data field - may be string in HTTP mode."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


class TruthMaintenanceSystem:
    """Track fact dependencies for automatic retraction."""

    def __init__(self):
        self.derived_facts: Dict[str, Set[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}

    def record_derivation(self, derived_id: str, source_ids: List[str]):
        """Record that a derived fact came from source facts."""
        self.derived_facts[derived_id] = set(source_ids)
        for source_id in source_ids:
            if source_id not in self.dependents:
                self.dependents[source_id] = set()
            self.dependents[source_id].add(derived_id)

    def get_dependents(self, fact_id: str) -> Set[str]:
        """Get all facts that depend on this fact (recursively)."""
        to_retract = {fact_id}
        if fact_id in self.dependents:
            for dependent_id in self.dependents[fact_id]:
                to_retract.update(self.get_dependents(dependent_id))
        return to_retract

    def clear_fact(self, fact_id: str):
        """Clear a fact from TMS records."""
        if fact_id in self.derived_facts:
            source_ids = self.derived_facts[fact_id]
            del self.derived_facts[fact_id]
            for source_id in source_ids:
                if source_id in self.dependents:
                    self.dependents[source_id].discard(fact_id)
        if fact_id in self.dependents:
            del self.dependents[fact_id]


class Rule:
    """A rule with conditions and an action."""

    def __init__(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        action: Callable,
        join_spec: Optional[Dict[str, str]] = None,
        description: str = "",
    ):
        self.name = name
        self.conditions = conditions
        self.action = action
        self.join_spec = join_spec or {}
        self.description = description
        self.activation_count = 0


class ReteSession:
    """
    Holon-powered Rete session.

    HTTP-Compatible: All similarity operations use search_json().
    No local vector operations (numpy).
    """

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        self.use_http = use_http
        self.base_url = base_url

        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.rules: List[Rule] = []
        self.facts: Dict[str, Dict] = {}
        self.tms = TruthMaintenanceSystem()

    def add_rule(self, rule: Rule):
        """Add a rule to the session."""
        self.rules.append(rule)
        print(f"📋 Added rule: {rule.name}")

    def insert(self, fact: Dict[str, Any], is_derived: bool = False) -> str:
        """Insert a fact into the session."""
        if "fact_id" not in fact:
            fact["fact_id"] = str(uuid.uuid4())

        fact_id = fact["fact_id"]
        fact["_is_derived"] = is_derived

        self.client.insert_json(fact)
        self.facts[fact_id] = fact

        type_name = fact.get("_type", "unknown")
        print(f"➕ {'Derived' if is_derived else 'Input'} fact: {fact_id[:8]}... ({type_name})")
        return fact_id

    def insert_prototype_examples(self, category: str, examples: List[Dict]):
        """
        Insert prototype examples for k-NN matching.

        Instead of computing a prototype vector locally, we store
        all examples with a category marker for k-NN lookup.
        """
        print(f"🧠 Storing {len(examples)} examples for '{category}'...")
        for i, example in enumerate(examples):
            example_fact = {
                **example,
                "_prototype_category": category,
                "_prototype_example": True,
                "fact_id": f"proto_{category}_{i}",
            }
            self.client.insert_json(example_fact)
        print(f"   ✅ Examples stored for k-NN matching")

    def retract(self, fact_id: str):
        """Retract a fact and all dependent facts."""
        if fact_id not in self.facts:
            print(f"⚠️  Fact {fact_id} not found")
            return

        to_retract = self.tms.get_dependents(fact_id)
        print(f"🗑️  Retracting {len(to_retract)} fact(s)...")

        for fid in to_retract:
            if fid in self.facts:
                del self.facts[fid]
                self.tms.clear_fact(fid)

    def match_condition(
        self, condition: Dict[str, Any], k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Match a condition against facts using Holon search.

        Supports:
        - Standard probe matching (fuzzy by default)
        - Guard filters (exact constraints)
        - Prototype matching via k-NN
        """
        # Check for prototype-based matching
        if "_similar_to_category" in condition:
            return self._match_prototype(condition, k)

        # Build probe and guard from condition
        probe = {}
        guard = {}

        for key, value in condition.items():
            if key.startswith("_"):
                continue  # Skip special markers
            elif isinstance(value, dict) and any(k.startswith("$") for k in value):
                guard[key] = value
            else:
                probe[key] = value

        # Search using Holon
        threshold = condition.get("_threshold", 0.0)
        results = self.client.search_json(
            probe=probe, guard=guard, limit=100, threshold=threshold
        )

        # Filter by type if specified
        matches = []
        type_filter = condition.get("_type")
        for result in results:
            fact = parse_data(result["data"])
            if not isinstance(fact, dict):
                continue
            if type_filter and fact.get("_type") != type_filter:
                continue
            # Skip prototype examples
            if fact.get("_prototype_example"):
                continue
            matches.append((fact, result["score"]))

        return matches

    def _match_prototype(
        self, condition: Dict[str, Any], k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Match facts similar to a prototype category using k-NN.

        Uses stored prototype examples to find similar facts.
        """
        category = condition["_similar_to_category"]
        type_filter = condition.get("_type")
        threshold = condition.get("_threshold", 0.0)

        # Get prototype examples for this category
        proto_results = self.client.search_json(
            probe={"_prototype_category": category},
            limit=k,
            threshold=0.0,
        )

        if not proto_results:
            return []

        # Use the first prototype example as a representative probe
        first_proto = parse_data(proto_results[0]["data"])
        if not isinstance(first_proto, dict):
            return []

        # Build a probe from the prototype's key fields
        probe = {
            key: value
            for key, value in first_proto.items()
            if not key.startswith("_") and key not in ["fact_id", "_prototype_category", "_prototype_example"]
        }

        # Add type filter
        if type_filter:
            probe["_type"] = type_filter

        # Search for similar facts
        results = self.client.search_json(probe=probe, limit=100, threshold=threshold)

        # Filter out prototype examples
        matches = []
        for r in results:
            fact = parse_data(r["data"])
            if not isinstance(fact, dict):
                continue
            if fact.get("_prototype_example"):
                continue
            if fact.get("_is_derived"):
                continue
            if type_filter and fact.get("_type") != type_filter:
                continue
            matches.append((fact, r["score"]))

        return matches

    def check_join(self, facts: List[Dict], join_spec: Dict[str, str]) -> bool:
        """Check if facts satisfy join conditions."""
        if not join_spec:
            return True

        for left_path, right_path in join_spec.items():
            left_parts = left_path.split(".")
            right_parts = right_path.split(".")

            left_fact = None
            right_fact = None
            for fact in facts:
                if fact.get("_type") == left_parts[0]:
                    left_fact = fact
                if fact.get("_type") == right_parts[0]:
                    right_fact = fact

            if not left_fact or not right_fact:
                return False

            left_val = left_fact.get(left_parts[1]) if len(left_parts) > 1 else left_fact
            right_val = right_fact.get(right_parts[1]) if len(right_parts) > 1 else right_fact

            if left_val != right_val:
                return False

        return True

    def fire_rules(self) -> List[Dict]:
        """Fire all rules and return activations."""
        print(f"\n🔥 Firing rules...")
        activations = []

        for rule in self.rules:
            # Match each condition
            condition_matches = []
            for condition in rule.conditions:
                matches = self.match_condition(condition)
                if not matches:
                    break  # All conditions must match
                condition_matches.append(matches)

            if len(condition_matches) != len(rule.conditions):
                continue

            # Generate combinations of matched facts
            from itertools import product

            for combo in product(*condition_matches):
                facts = [fact for fact, score in combo]
                scores = [score for fact, score in combo]

                # Check joins
                if not self.check_join(facts, rule.join_spec):
                    continue

                # Calculate confidence
                confidence = min(scores) if scores else 1.0

                # Fire action
                result = rule.action(facts)
                if result:
                    activations.append({
                        "rule": rule.name,
                        "facts": [f.get("fact_id", "?")[:8] for f in facts],
                        "confidence": confidence,
                        "result": result,
                    })
                    rule.activation_count += 1

        print(f"   → {len(activations)} activation(s)")
        return activations

    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            "facts": len(self.facts),
            "derived_facts": sum(1 for f in self.facts.values() if f.get("_is_derived")),
            "rules": len(self.rules),
            "rule_activations": {r.name: r.activation_count for r in self.rules},
        }


def demo_exact_matching(session: ReteSession):
    """Demo 1: Traditional exact matching."""
    print("\n" + "=" * 70)
    print("DEMO 1: Exact Matching (Traditional Rete)")
    print("=" * 70)

    alerts_created = []

    def high_value_action(facts: List[Dict]) -> Optional[Dict]:
        order = next(f for f in facts if f["_type"] == "Order")
        customer = next(f for f in facts if f["_type"] == "Customer")

        alert = {
            "_type": "Alert",
            "priority": "high",
            "order_id": order["id"],
            "customer_id": customer["id"],
            "reason": "High-value platinum customer order",
        }
        fact_id = session.insert(alert, is_derived=True)
        alerts_created.append(fact_id)
        return alert

    rule = Rule(
        name="high-value-platinum",
        conditions=[
            {"_type": "Order", "status": "pending"},
            {"_type": "Customer", "tier": "platinum"},
        ],
        action=high_value_action,
        join_spec={"Order.customer_id": "Customer.id"},
        description="Alert on high-value platinum orders",
    )
    session.add_rule(rule)

    print("\n📥 Inserting facts...")
    session.insert({
        "_type": "Order",
        "id": "order-123",
        "customer_id": "cust-456",
        "status": "pending",
        "total": 15000,
    })
    session.insert({
        "_type": "Customer",
        "id": "cust-456",
        "name": "Acme Corp",
        "tier": "platinum",
    })

    activations = session.fire_rules()
    print(f"\n✅ Exact matching: {len(activations)} alert(s) created")
    return len(activations)


def demo_fuzzy_matching(session: ReteSession):
    """Demo 2: Fuzzy matching with k-NN prototypes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Fuzzy Matching with k-NN Prototypes (HTTP-Compatible)")
    print("=" * 70)

    # Create and store fraud examples
    fraud_examples = [
        {"_type": "Transaction", "amount": 12000, "country": "NG", "time_hour": 3, "ip_changes": 5},
        {"_type": "Transaction", "amount": 15000, "country": "RU", "time_hour": 2, "ip_changes": 4},
        {"_type": "Transaction", "amount": 10000, "country": "CN", "time_hour": 4, "ip_changes": 6},
        {"_type": "Transaction", "amount": 18000, "country": "NG", "time_hour": 1, "ip_changes": 7},
        {"_type": "Transaction", "amount": 14000, "country": "RU", "time_hour": 3, "ip_changes": 5},
    ]
    session.insert_prototype_examples("fraud_pattern", fraud_examples)

    flags_created = []

    def flag_suspicious_action(facts: List[Dict]) -> Optional[Dict]:
        transaction = facts[0]
        flag = {
            "_type": "Flag",
            "transaction_id": transaction.get("id", "unknown"),
            "reason": "Similar to known fraud patterns",
        }
        session.insert(flag, is_derived=True)
        flags_created.append(transaction.get("id"))
        return flag

    rule = Rule(
        name="similar-to-fraud",
        conditions=[{
            "_type": "Transaction",
            "_similar_to_category": "fraud_pattern",
            "_threshold": 0.05,
        }],
        action=flag_suspicious_action,
        description="Flag transactions similar to fraud prototype",
    )
    session.add_rule(rule)

    print("\n📥 Inserting test transactions...")
    test_txns = [
        {"_type": "Transaction", "id": "txn-001", "amount": 11000, "country": "NG", "time_hour": 3, "ip_changes": 5},
        {"_type": "Transaction", "id": "txn-002", "amount": 5000, "country": "US", "time_hour": 14, "ip_changes": 1},
        {"_type": "Transaction", "id": "txn-003", "amount": 13000, "country": "RU", "time_hour": 2, "ip_changes": 4},
        {"_type": "Transaction", "id": "txn-004", "amount": 200, "country": "US", "time_hour": 10, "ip_changes": 0},
    ]
    for txn in test_txns:
        session.insert(txn)

    activations = session.fire_rules()

    # Filter to just the fuzzy matching results
    fuzzy_activations = [a for a in activations if a["rule"] == "similar-to-fraud"]
    print(f"\n✅ Fuzzy matching: {len(fuzzy_activations)} transaction(s) flagged")

    for act in fuzzy_activations:
        txn_id = act['result'].get('transaction_id', '?')
        print(f"   - {txn_id}: confidence={act['confidence']:.3f}")

    return len(fuzzy_activations)


def demo_truth_maintenance(session: ReteSession):
    """Demo 3: Truth maintenance with retraction."""
    print("\n" + "=" * 70)
    print("DEMO 3: Truth Maintenance")
    print("=" * 70)

    stats_before = session.get_stats()
    print(f"\n📊 Before retraction:")
    print(f"   Total facts: {stats_before['facts']}")
    print(f"   Derived facts: {stats_before['derived_facts']}")

    order_facts = [f for f in session.facts.values() if f.get("_type") == "Order"]
    if order_facts:
        order = order_facts[0]
        order_id = order["fact_id"]

        print(f"\n🗑️  Retracting order: {order_id[:8]}...")
        session.retract(order_id)

        stats_after = session.get_stats()
        print(f"\n📊 After retraction:")
        print(f"   Total facts: {stats_after['facts']}")
        print(f"   Derived facts: {stats_after['derived_facts']}")
        retracted = stats_before['facts'] - stats_after['facts']
        print(f"   ✅ Retracted: {retracted} fact(s)")
        return retracted
    return 0


def main():
    parser = argparse.ArgumentParser(description="Holon-Powered Rete Rule Engine")
    parser.add_argument("--http", action="store_true", help="Use HTTP API")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()

    print("=" * 70)
    print("HOLON-POWERED RETE RULE ENGINE (HTTP-Compatible)")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    if args.http:
        print(f"   Server: {args.url}")
        print("   ⚠️  Ensure server is running")

    start_time = time.time()
    session = ReteSession(use_http=args.http, base_url=args.url)

    # Run demos
    exact_alerts = demo_exact_matching(session)
    fuzzy_flags = demo_fuzzy_matching(session)
    retracted = demo_truth_maintenance(session)

    elapsed = time.time() - start_time
    stats = session.get_stats()

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"""
    Mode: {mode}
    Elapsed: {elapsed:.2f}s

    Facts: {stats['facts']}
    Derived Facts: {stats['derived_facts']}
    Rules: {stats['rules']}

    Results:
      Exact matching alerts: {exact_alerts}
      Fuzzy matching flags: {fuzzy_flags}
      Facts retracted: {retracted}

    ✅ HTTP-Compatible Implementation:
       - All similarity via search_json() (no local numpy)
       - k-NN prototype matching (stored as facts)
       - Works identically in local and HTTP modes

    This combines traditional Rete (precise logic)
    with Holon's VSA capabilities (fuzzy similarity).
    """)


if __name__ == "__main__":
    main()
