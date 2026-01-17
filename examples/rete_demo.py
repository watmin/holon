#!/usr/bin/env python3
"""
Rete-like Forward Chaining Demo using Holon.

This demonstrates rule-based reasoning where facts are stored in Holon,
rules are queries that match patterns, and forward chaining adds derived facts.
"""

from holon import CPUStore
import json

class ReteDemo:
    def __init__(self):
        self.store = CPUStore()
        self.rules = []
        self.fact_counter = 0

    def add_fact(self, fact):
        """Add a fact to the knowledge base."""
        fact_id = f"fact_{self.fact_counter}"
        self.fact_counter += 1
        fact_with_id = {"id": fact_id, **fact}
        self.store.insert(json.dumps(fact_with_id))
        print(f"➕ Added fact: {fact_with_id}")
        return fact_id

    def add_rule(self, name, conditions, action):
        """Add a rule: conditions (query dict), action (function to call on matches)."""
        self.rules.append({
            "name": name,
            "conditions": conditions,
            "action": action
        })
        print(f"📋 Added rule: {name}")

    def run_forward_chaining(self, max_iterations=5):
        """Run forward chaining: check rules, fire actions, add new facts."""
        print("\n🔄 Starting Forward Chaining...\n")

        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            fired = False

            for rule in self.rules:
                # Query for matching facts
                probe = json.dumps(rule["conditions"])
                results = self.store.query(probe, top_k=10)

                if results:
                    print(f"🎯 Rule '{rule['name']}' matched {len(results)} facts")
                    for result in results:
                        # Fire action
                        new_fact = rule["action"](result[2])
                        if new_fact:
                            self.add_fact(new_fact)
                            fired = True

            if not fired:
                print("✋ No rules fired. Stopping.")
                break

        print("\n🏁 Forward Chaining Complete!")

def main():
    demo = ReteDemo()

    # Define rules (similar to Clara-style)
    def parent_rule(fact):
        # If someone is a parent, infer they have children
        if "parent_of" in fact:
            return {"person": fact["person"], "has_children": True, "derived_from": "parent_rule"}

    def grandparent_rule(fact):
        # If someone has children and is parent of someone who has children
        if fact.get("has_children") and "person" in fact:
            # Check if this person is parent of someone with children
            grandparents = demo.store.query(
                json.dumps({"parent_of": fact["person"], "has_children": True}),
                top_k=5
            )
            if grandparents:
                return {"person": fact["person"], "is_grandparent": True, "derived_from": "grandparent_rule"}

    demo.add_rule("infer_has_children", {"parent_of": {"$any": True}}, parent_rule)
    demo.add_rule("infer_grandparent", {"has_children": True}, grandparent_rule)

    # Initial facts
    demo.add_fact({"person": "alice", "parent_of": "bob"})
    demo.add_fact({"person": "bob", "parent_of": "charlie"})

    # Run forward chaining
    demo.run_forward_chaining()

    # Query final state
    print("\n📊 Final Knowledge Base:")
    all_facts = demo.store.query('{}', top_k=20)  # Simple query for all
    for fact in all_facts:
        print(f"  {fact[2]}")

if __name__ == "__main__":
    main()