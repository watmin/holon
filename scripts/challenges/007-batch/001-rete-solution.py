#!/usr/bin/env python3
"""
Challenge 007-001: Holon-Powered Rete Rule Engine

Demonstrates a rule engine that combines:
1. Exact matching (traditional Rete) for precise conditions
2. Fuzzy matching (Holon) for similarity-based conditions
3. Prototype-based rules that fire on "similar enough" facts
4. Multi-condition rules with joins
5. Truth maintenance system

Usage:
    # Local mode
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/001-rete-solution.py

    # HTTP mode (requires server running)
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/001-rete-solution.py --http
"""

import argparse
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from holon import CPUStore, HolonClient


class TruthMaintenanceSystem:
    """Track fact dependencies for automatic retraction."""

    def __init__(self):
        self.derived_facts: Dict[str, Set[str]] = {}  # derived_id -> source_ids
        self.dependents: Dict[str, Set[str]] = {}  # source_id -> derived_ids

    def record_derivation(self, derived_id: str, source_ids: List[str]):
        """Record that a derived fact came from source facts."""
        self.derived_facts[derived_id] = set(source_ids)
        for source_id in source_ids:
            if source_id not in self.dependents:
                self.dependents[source_id] = set()
            self.dependents[source_id].add(derived_id)

    def get_dependents(self, fact_id: str) -> Set[str]:
        """Get all facts that depend on this fact."""
        to_retract = {fact_id}

        # Cascade to dependents
        if fact_id in self.dependents:
            for dependent_id in self.dependents[fact_id]:
                to_retract.update(self.get_dependents(dependent_id))

        return to_retract

    def clear_fact(self, fact_id: str):
        """Clear a fact from TMS records."""
        # Remove from derived_facts
        if fact_id in self.derived_facts:
            source_ids = self.derived_facts[fact_id]
            del self.derived_facts[fact_id]

            # Remove from dependents
            for source_id in source_ids:
                if source_id in self.dependents:
                    self.dependents[source_id].discard(fact_id)

        # Remove from dependents
        if fact_id in self.dependents:
            del self.dependents[fact_id]


class Rule:
    """Represents a rule with conditions and an action."""

    def __init__(
        self,
        name: str,
        conditions: List[Dict[str, Any]],
        action: Callable,
        join_spec: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.conditions = conditions
        self.action = action
        self.join_spec = join_spec or {}
        self.activation_count = 0

    def get_probe_guard(self, condition: Dict[str, Any]) -> Tuple[Dict, Dict, Optional[Any], float]:
        """Split condition into probe, guard, similarity check, and threshold."""
        probe = {}
        guard = {}
        similarity_check = None
        threshold = 0.0

        for key, value in condition.items():
            if key == "_similar_to":
                similarity_check = value
            elif key == "_threshold":
                threshold = value
            elif key.startswith("_"):
                continue  # Skip other special markers
            elif isinstance(value, dict) and any(k.startswith("$") for k in value):
                guard[key] = value
            else:
                probe[key] = value

        return probe, guard, similarity_check, threshold


class ReteSession:
    """Holon-powered Rete session with exact and fuzzy matching."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        self.use_http = use_http
        self.base_url = base_url

        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.rules: List[Rule] = []
        self.facts: Dict[str, Dict] = {}  # fact_id -> fact
        self.tms = TruthMaintenanceSystem()
        self.prototypes: Dict[str, Any] = {}  # name -> prototype vector
        self.iteration = 0

    def add_rule(self, rule: Rule):
        """Add a rule to the session."""
        self.rules.append(rule)
        print(f"📋 Added rule: {rule.name}")

    def insert(self, fact: Dict[str, Any], is_derived: bool = False) -> str:
        """Insert a fact into the session."""
        # Ensure fact has an ID
        if "fact_id" not in fact:
            fact["fact_id"] = str(uuid.uuid4())

        fact_id = fact["fact_id"]
        fact["_is_derived"] = is_derived

        # Store in Holon
        self.client.insert_json(fact)
        self.facts[fact_id] = fact

        print(f"➕ {'Derived' if is_derived else 'Input'} fact: {fact_id[:8]}... {fact.get('_type', 'unknown')}")
        return fact_id

    def retract(self, fact_id: str):
        """Retract a fact and all dependent facts."""
        if fact_id not in self.facts:
            print(f"⚠️  Fact {fact_id} not found")
            return

        # Get all facts to retract (including dependents)
        to_retract = self.tms.get_dependents(fact_id)

        print(f"🗑️  Retracting {len(to_retract)} fact(s)...")
        for fid in to_retract:
            if fid in self.facts:
                del self.facts[fid]
                self.tms.clear_fact(fid)

    def match_condition(self, condition: Dict[str, Any]) -> List[Tuple[Dict, float]]:
        """Match a single condition against facts."""
        rule = Rule("temp", [condition], lambda x: None)
        probe, guard, similarity_check, threshold = rule.get_probe_guard(condition)

        if similarity_check is not None:
            # Fuzzy matching with prototype
            if isinstance(similarity_check, str) and similarity_check in self.prototypes:
                # Use learned prototype - we need to search by comparing manually
                prototype_vec = self.prototypes[similarity_check]
                matches = []

                # Get all facts of the right type
                type_filter = {"_type": condition.get("_type")} if "_type" in condition else {}
                all_facts = self.client.search_json(probe=type_filter, limit=1000, threshold=0.0)

                # Score each fact against the prototype
                import numpy as np
                for result in all_facts:
                    fact = result["data"]
                    fact_vec = self.client.encode_vectors(fact)
                    if isinstance(fact_vec, list):
                        fact_vec = np.array(fact_vec)

                    # Calculate similarity
                    if isinstance(prototype_vec, np.ndarray) and isinstance(fact_vec, np.ndarray):
                        score = float(np.dot(prototype_vec, fact_vec) / (np.linalg.norm(prototype_vec) * np.linalg.norm(fact_vec) + 1e-10))
                    else:
                        score = 0.0

                    if score >= threshold:
                        matches.append((fact, score))

                # Sort by score
                matches.sort(key=lambda x: x[1], reverse=True)
                return matches[:100]
            elif isinstance(similarity_check, dict):
                # Use provided data as similarity target
                results = self.client.search_json(probe=similarity_check, limit=100, threshold=threshold)
            else:
                results = []
        else:
            # Standard probe + guard search
            results = self.client.search_json(probe=probe, guard=guard, limit=100, threshold=threshold)

        # Return facts with scores
        matches = []
        for result in results:
            fact = result["data"]
            score = result["score"]
            # Filter by type if specified
            if "_type" in condition and fact.get("_type") != condition["_type"]:
                continue
            matches.append((fact, score))

        return matches

    def check_join(self, facts: List[Dict], join_spec: Dict[str, str]) -> bool:
        """Check if facts satisfy join conditions."""
        if not join_spec:
            return True

        for left_path, right_path in join_spec.items():
            # Parse paths like "Order.customer_id" and "Customer.id"
            left_parts = left_path.split(".")
            right_parts = right_path.split(".")

            # Find facts by type
            left_fact = None
            right_fact = None
            for fact in facts:
                if fact.get("_type") == left_parts[0]:
                    left_fact = fact
                if fact.get("_type") == right_parts[0]:
                    right_fact = fact

            if not left_fact or not right_fact:
                return False

            # Get values
            left_val = left_fact.get(left_parts[1]) if len(left_parts) > 1 else left_fact
            right_val = right_fact.get(right_parts[1]) if len(right_parts) > 1 else right_fact

            if left_val != right_val:
                return False

        return True

    def fire_rules(self) -> List[Dict]:
        """Fire all rules and return activations."""
        self.iteration += 1
        print(f"\n🔥 Firing rules (iteration {self.iteration})...")

        activations = []

        for rule in self.rules:
            # Match each condition
            condition_matches = []
            for condition in rule.conditions:
                matches = self.match_condition(condition)
                condition_matches.append(matches)

            if not condition_matches:
                continue

            # Generate all combinations of matched facts
            from itertools import product

            for combo in product(*condition_matches):
                facts = [fact for fact, score in combo]
                scores = [score for fact, score in combo]

                # Check joins
                if not self.check_join(facts, rule.join_spec):
                    continue

                # Calculate overall confidence (min of scores)
                confidence = min(scores) if scores else 1.0

                # Fire action
                result = rule.action(facts)

                if result:
                    activation = {
                        "rule": rule.name,
                        "facts": facts,
                        "confidence": confidence,
                        "result": result,
                    }
                    activations.append(activation)
                    rule.activation_count += 1

        print(f"   → {len(activations)} activation(s)")
        return activations

    def learn_prototype(self, name: str, examples: List[Dict]):
        """Learn a prototype from example facts."""
        print(f"🧠 Learning prototype '{name}' from {len(examples)} examples...")

        # Encode all examples
        import numpy as np
        vectors = []
        for example in examples:
            vec = self.client.encode_vectors(example)
            # Convert to numpy array if it's a list
            if isinstance(vec, list):
                vec = np.array(vec)
            vectors.append(vec)

        # Average vectors to create prototype
        if vectors:
            if self.use_http:
                # For HTTP mode, use simple average
                prototype = np.mean(vectors, axis=0)
            else:
                # For local mode, use store's prototype method
                prototype = self.store.prototype(vectors)
            self.prototypes[name] = prototype
            print(f"   → Prototype learned: {len(prototype)}D vector")
        else:
            print(f"   ⚠️  No examples to learn from")

    def get_stats(self) -> Dict:
        """Get session statistics."""
        return {
            "facts": len(self.facts),
            "derived_facts": sum(1 for f in self.facts.values() if f.get("_is_derived")),
            "rules": len(self.rules),
            "rule_activations": {r.name: r.activation_count for r in self.rules},
            "prototypes": len(self.prototypes),
        }


def demo_exact_matching(session: ReteSession):
    """Demo 1: Traditional exact matching."""
    print("\n" + "=" * 70)
    print("DEMO 1: Exact Matching (Traditional Rete)")
    print("=" * 70)

    # Define rule: High-value platinum orders
    def high_value_action(facts: List[Dict]) -> Optional[Dict]:
        order = next(f for f in facts if f["_type"] == "Order")
        customer = next(f for f in facts if f["_type"] == "Customer")

        # Create alert
        alert = {
            "_type": "Alert",
            "alert_id": str(uuid.uuid4()),
            "priority": "high",
            "order_id": order["id"],
            "customer_id": customer["id"],
            "reason": "High-value platinum customer order",
        }
        session.insert(alert, is_derived=True)
        return alert

    rule = Rule(
        name="high-value-platinum",
        conditions=[
            {"_type": "Order", "status": "pending"},
            {"_type": "Customer", "tier": "platinum"},
        ],
        action=high_value_action,
        join_spec={"Order.customer_id": "Customer.id"},
    )
    session.add_rule(rule)

    # Insert facts
    print("\n📥 Inserting facts...")
    session.insert(
        {
            "_type": "Order",
            "id": "order-123",
            "customer_id": "cust-456",
            "status": "pending",
            "total": 15000,
        }
    )
    session.insert(
        {
            "_type": "Customer",
            "id": "cust-456",
            "name": "Acme Corp",
            "tier": "platinum",
        }
    )

    # Fire rules
    activations = session.fire_rules()

    print(f"\n✅ Exact matching: {len(activations)} alert(s) created")


def demo_fuzzy_matching(session: ReteSession):
    """Demo 2: Fuzzy matching with prototypes."""
    print("\n" + "=" * 70)
    print("DEMO 2: Fuzzy Matching with Prototypes")
    print("=" * 70)

    # Create fraud examples
    fraud_examples = [
        {
            "_type": "Transaction",
            "amount": 12000,
            "country": "NG",
            "time_hour": 3,
            "ip_changes": 5,
        },
        {
            "_type": "Transaction",
            "amount": 15000,
            "country": "RU",
            "time_hour": 2,
            "ip_changes": 4,
        },
        {
            "_type": "Transaction",
            "amount": 10000,
            "country": "CN",
            "time_hour": 4,
            "ip_changes": 6,
        },
    ]

    # Learn fraud prototype
    session.learn_prototype("fraud_pattern", fraud_examples)

    # Define fuzzy rule
    def flag_suspicious_action(facts: List[Dict]) -> Optional[Dict]:
        transaction = facts[0]
        flag = {
            "_type": "Flag",
            "flag_id": str(uuid.uuid4()),
            "transaction_id": transaction.get("id", "unknown"),
            "reason": "Similar to known fraud patterns",
        }
        session.insert(flag, is_derived=True)
        return flag

    rule = Rule(
        name="similar-to-fraud",
        conditions=[
            {
                "_type": "Transaction",
                "_similar_to": "fraud_pattern",
                "_threshold": 0.15,  # Lower threshold for VSA similarity
            }
        ],
        action=flag_suspicious_action,
    )
    session.add_rule(rule)

    # Insert test transactions
    print("\n📥 Inserting test transactions...")
    test_txns = [
        {
            "_type": "Transaction",
            "id": "txn-001",
            "amount": 11000,
            "country": "NG",
            "time_hour": 3,
            "ip_changes": 5,
        },  # Very similar
        {
            "_type": "Transaction",
            "id": "txn-002",
            "amount": 5000,
            "country": "US",
            "time_hour": 14,
            "ip_changes": 1,
        },  # Not similar
        {
            "_type": "Transaction",
            "id": "txn-003",
            "amount": 13000,
            "country": "RU",
            "time_hour": 2,
            "ip_changes": 4,
        },  # Similar
    ]

    for txn in test_txns:
        session.insert(txn)

    # Fire rules
    activations = session.fire_rules()

    print(f"\n✅ Fuzzy matching: {len(activations)} transaction(s) flagged")
    for act in activations:
        txn = act["facts"][0]
        print(f"   - {txn.get('id', 'unknown')}: confidence={act['confidence']:.3f}")


def demo_hybrid_matching(session: ReteSession):
    """Demo 3: Hybrid exact + fuzzy matching."""
    print("\n" + "=" * 70)
    print("DEMO 3: Hybrid Exact + Fuzzy Matching")
    print("=" * 70)

    # Define hybrid rule
    def escalate_action(facts: List[Dict]) -> Optional[Dict]:
        customer = next(f for f in facts if f["_type"] == "Customer")
        transaction = next(f for f in facts if f["_type"] == "Transaction")

        escalation = {
            "_type": "Escalation",
            "escalation_id": str(uuid.uuid4()),
            "customer_id": customer["id"],
            "transaction_id": transaction.get("id", "unknown"),
            "reason": "Platinum customer with suspicious activity",
        }
        session.insert(escalation, is_derived=True)
        return escalation

    rule = Rule(
        name="suspicious-platinum",
        conditions=[
            {"_type": "Customer", "tier": "platinum"},  # Exact
            {
                "_type": "Transaction",
                "_similar_to": "fraud_pattern",
                "_threshold": 0.5,
            },  # Fuzzy
        ],
        action=escalate_action,
        join_spec={"Transaction.customer_id": "Customer.id"},
    )
    session.add_rule(rule)

    # Insert facts
    print("\n📥 Inserting facts...")
    session.insert(
        {
            "_type": "Customer",
            "id": "cust-789",
            "name": "VIP Corp",
            "tier": "platinum",
        }
    )
    session.insert(
        {
            "_type": "Transaction",
            "id": "txn-004",
            "customer_id": "cust-789",
            "amount": 14000,
            "country": "CN",
            "time_hour": 4,
            "ip_changes": 7,
        }
    )

    # Fire rules
    activations = session.fire_rules()

    print(f"\n✅ Hybrid matching: {len(activations)} escalation(s) created")


def demo_truth_maintenance(session: ReteSession):
    """Demo 4: Truth maintenance with retraction."""
    print("\n" + "=" * 70)
    print("DEMO 4: Truth Maintenance")
    print("=" * 70)

    # Show current facts
    stats_before = session.get_stats()
    print(f"\n📊 Before retraction:")
    print(f"   Total facts: {stats_before['facts']}")
    print(f"   Derived facts: {stats_before['derived_facts']}")

    # Find an order to retract
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
        print(f"   Retracted: {stats_before['facts'] - stats_after['facts']} fact(s)")


def demo_prototype_evolution(session: ReteSession):
    """Demo 5: Evolving prototypes with new examples."""
    print("\n" + "=" * 70)
    print("DEMO 5: Prototype Evolution")
    print("=" * 70)

    # Create initial normal behavior examples
    normal_examples = [
        {"_type": "Login", "time_hour": 9, "duration_sec": 300, "failed_attempts": 0},
        {"_type": "Login", "time_hour": 14, "duration_sec": 180, "failed_attempts": 0},
        {"_type": "Login", "time_hour": 10, "duration_sec": 240, "failed_attempts": 1},
    ]

    session.learn_prototype("normal_login", normal_examples)

    # Test before evolution
    test_login = {
        "_type": "Login",
        "time_hour": 3,
        "duration_sec": 60,
        "failed_attempts": 5,
    }

    print("\n🧪 Testing against initial prototype...")
    probe_vec = session.client.encode_vectors(test_login)
    normal_vec = session.prototypes["normal_login"]

    import numpy as np

    score_before = float(np.dot(probe_vec, normal_vec))
    print(f"   Similarity to normal: {score_before:.4f}")

    # Add new normal example and update prototype
    new_normal = {
        "_type": "Login",
        "time_hour": 11,
        "duration_sec": 200,
        "failed_attempts": 0,
    }

    print(f"\n🔄 Evolving prototype with new example...")
    new_vec = session.client.encode_vectors(new_normal)

    # Update with weighted average
    alpha = 0.1  # weight for new example
    updated_proto = [(1 - alpha) * old + alpha * new for old, new in zip(normal_vec, new_vec)]
    session.prototypes["normal_login"] = updated_proto

    score_after = float(np.dot(probe_vec, updated_proto))
    print(f"   Similarity after evolution: {score_after:.4f}")
    print(f"   Change: {score_after - score_before:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Holon-Powered Rete Rule Engine")
    parser.add_argument("--http", action="store_true", help="Use HTTP API instead of local store")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for HTTP mode")
    args = parser.parse_args()

    print("=" * 70)
    print("HOLON-POWERED RETE RULE ENGINE")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    if args.http:
        print(f"   Server: {args.url}")
        print("   ⚠️  Ensure server is running: ./scripts/run_with_venv.sh python scripts/server/holon_server.py")

    # Create session
    start_time = time.time()
    session = ReteSession(use_http=args.http, base_url=args.url)

    # Run demos
    demo_exact_matching(session)
    demo_fuzzy_matching(session)
    demo_hybrid_matching(session)
    demo_truth_maintenance(session)
    demo_prototype_evolution(session)

    # Final stats
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
    Prototypes: {stats['prototypes']}

    Rule Activations:
    {chr(10).join(f'      {name}: {count}' for name, count in stats['rule_activations'].items())}

    ✅ Rete engine demonstrates:
       - Exact pattern matching
       - Fuzzy similarity matching
       - Prototype learning
       - Multi-condition rules with joins
       - Truth maintenance
       - Prototype evolution

    This combines the best of traditional Rete (precise logic)
    with Holon's VSA capabilities (fuzzy similarity).
    """)


if __name__ == "__main__":
    main()
