#!/usr/bin/env python3
"""
Rete-like Forward Chaining Demo using Holon.

This demonstrates rule-based reasoning where facts are stored in Holon,
rules are queries that match patterns, and forward chaining adds derived facts.
"""

import json

from holon import CPUStore, HolonClient


class ReteDemo:
    def __init__(self):
        self.store = CPUStore()
        self.client = HolonClient(local_store=self.store)
        self.rules = []
        self.fact_counter = 0
        self.input_facts = []  # For replayability

    def add_fact(self, fact, is_input=True):
        """Add a fact to the knowledge base."""
        fact_id = f"fact_{self.fact_counter}"
        self.fact_counter += 1
        fact_with_meta = {"id": fact_id, "is_input": is_input, **fact}
        stored_id = self.client.insert_json(fact_with_meta)
        if is_input:
            self.input_facts.append(fact_with_meta)
        print(f"➕ Added {'input' if is_input else 'derived'} fact: {fact_with_meta}")
        return fact_id

    def add_rule(self, name, conditions, action):
        """Add a rule: conditions (query dict), action (function to call on matches)."""
        self.rules.append({"name": name, "conditions": conditions, "action": action})
        print(f"📋 Added rule: {name}")

    def run_forward_chaining(self, max_iterations=3):
        """Run forward chaining: check rules, fire actions, add new facts."""
        print("\n🔄 Starting Forward Chaining...\n")

        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            fired = False

            for rule in self.rules:
                # Query for matching facts
                results = self.client.search_json(rule["conditions"], top_k=10)

                if results:
                    print(f"🎯 Rule '{rule['name']}' fired ({len(results)} matches)")
                    # Fire actions
                    for result in results:
                        new_fact = rule["action"](result["data"])
                        if new_fact:
                            self.add_fact(new_fact, is_input=False)
                            fired = True

            if not fired:
                print("✋ No rules fired. Stopping.")
                break

        print("\n🏁 Forward Chaining Complete!")

    def replay(self):
        """Replay: Reset to input facts, rerun rules."""
        print("\n🔄 Replaying from input facts...")
        # Save original input facts before resetting
        original_input_facts = [fact.copy() for fact in self.input_facts]
        self.store = CPUStore()  # Reset store
        self.client = HolonClient(local_store=self.store)  # Reset client too
        self.fact_counter = 0
        self.input_facts = []  # Clear the list
        for fact in original_input_facts:
            self.add_fact(fact, is_input=True)
        self.run_forward_chaining()

    def query_derived(self, query):
        """Query for derived facts with specific findings."""
        # Use guard to filter derived facts
        guard = {"is_input": False}
        if "finding" in query:
            guard["finding"] = query["finding"]
        results = self.client.search_json(query, guard=guard, top_k=10)
        return [r["data"] for r in results]


def main():
    demo = ReteDemo()

    # Define rules (similar to Clara-style)
    def parent_rule(fact):
        # If someone is a parent, infer they have children
        if "parent_of" in fact:
            return {
                "person": fact["person"],
                "has_children": True,
                "finding": "parent_inferred",
                "derived_from": "parent_rule",
            }

    def grandparent_rule(fact):
        # If someone has children and is parent of someone who has children
        if fact.get("has_children") and "person" in fact:
            # Check if this person is parent of someone with children
            grandparents = demo.client.search_json(
                {"parent_of": fact["person"], "has_children": True}, top_k=5
            )
            if grandparents:
                return {
                    "person": fact["person"],
                    "is_grandparent": True,
                    "finding": "grandparent_inferred",
                    "derived_from": "grandparent_rule",
                }

    demo.add_rule("infer_has_children", {"parent_of": {"$any": True}}, parent_rule)
    demo.add_rule("infer_grandparent", {"has_children": True}, grandparent_rule)

    # Initial facts
    demo.add_fact({"person": "alice", "parent_of": "bob"})
    demo.add_fact({"person": "bob", "parent_of": "charlie"})

    # Run forward chaining
    demo.run_forward_chaining()

    # Query derived facts
    print("\n🔍 Querying Derived Facts:")
    parents = demo.query_derived({"finding": "parent_inferred"})
    grandparents = demo.query_derived({"finding": "grandparent_inferred"})
    print(f"  Parents inferred: {len(parents)}")
    print(f"  Grandparents inferred: {len(grandparents)}")

    # Demonstrate replayability
    print("\n🎬 Demonstrating Replayability:")
    print("Original run complete. Replaying...")
    demo.replay()

    # Query again to verify
    parents_replay = demo.query_derived({"finding": "parent_inferred"})
    print(f"  After replay - Parents: {len(parents_replay)}")


if __name__ == "__main__":
    main()
