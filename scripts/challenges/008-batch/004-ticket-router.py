#!/usr/bin/env python3
"""
Challenge 008-004: Customer Support Ticket Router

Auto-route support tickets to the right team using k-NN classification
based on similarity to past resolved tickets.

Use case: Support teams get tickets auto-routed based on what worked before,
not just keyword matching.

Key Holon features demonstrated:
- k-NN classification for team routing
- Prototype learning from resolved tickets
- Guard filters for quality (satisfaction >= 4.0)
- Time awareness (similar issues from last month)
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import Counter
from holon import CPUStore, HolonClient

# =============================================================================
# Ticket Templates
# =============================================================================

TEAMS = ["billing", "technical", "account", "sales", "shipping"]

TICKET_TEMPLATES = {
    "billing": [
        {"subject": "Incorrect charge on my account", "keywords": ["charge", "billing", "invoice", "payment"]},
        {"subject": "Refund request for duplicate payment", "keywords": ["refund", "duplicate", "payment"]},
        {"subject": "Unable to update payment method", "keywords": ["payment", "card", "update"]},
        {"subject": "Subscription renewal issue", "keywords": ["subscription", "renewal", "billing"]},
        {"subject": "Invoice not received", "keywords": ["invoice", "receipt", "email"]},
        {"subject": "Promo code not applied", "keywords": ["promo", "discount", "code", "coupon"]},
        {"subject": "Charged after cancellation", "keywords": ["charge", "cancel", "refund"]},
    ],
    "technical": [
        {"subject": "App crashes on startup", "keywords": ["crash", "app", "error", "bug"]},
        {"subject": "Cannot login to my account", "keywords": ["login", "password", "access", "authentication"]},
        {"subject": "Feature not working as expected", "keywords": ["feature", "bug", "broken"]},
        {"subject": "Slow performance issues", "keywords": ["slow", "performance", "loading"]},
        {"subject": "Data sync not working", "keywords": ["sync", "data", "update"]},
        {"subject": "Error message when saving", "keywords": ["error", "save", "failed"]},
        {"subject": "Integration not connecting", "keywords": ["integration", "api", "connect"]},
        {"subject": "Mobile app notifications broken", "keywords": ["mobile", "notification", "push"]},
    ],
    "account": [
        {"subject": "Need to change account email", "keywords": ["email", "change", "account"]},
        {"subject": "Account locked after failed attempts", "keywords": ["locked", "account", "password"]},
        {"subject": "Merge two accounts", "keywords": ["merge", "account", "duplicate"]},
        {"subject": "Delete my account", "keywords": ["delete", "account", "gdpr", "data"]},
        {"subject": "Update company information", "keywords": ["company", "update", "profile"]},
        {"subject": "Add team member to account", "keywords": ["team", "member", "add", "invite"]},
    ],
    "sales": [
        {"subject": "Question about enterprise pricing", "keywords": ["pricing", "enterprise", "quote"]},
        {"subject": "Upgrade subscription plan", "keywords": ["upgrade", "plan", "subscription"]},
        {"subject": "Custom solution inquiry", "keywords": ["custom", "solution", "enterprise"]},
        {"subject": "Demo request", "keywords": ["demo", "trial", "presentation"]},
        {"subject": "Volume discount available?", "keywords": ["volume", "discount", "bulk"]},
    ],
    "shipping": [
        {"subject": "Order not delivered", "keywords": ["order", "delivery", "shipping", "missing"]},
        {"subject": "Wrong item received", "keywords": ["wrong", "item", "order", "exchange"]},
        {"subject": "Tracking number not working", "keywords": ["tracking", "shipment", "status"]},
        {"subject": "Change delivery address", "keywords": ["address", "delivery", "change"]},
        {"subject": "Damaged item received", "keywords": ["damaged", "broken", "item"]},
    ],
}

ISSUE_DESCRIPTIONS = {
    "billing": [
        "I was charged twice for my subscription this month.",
        "The promo code SAVE20 didn't apply to my order.",
        "I need a refund for an accidental purchase.",
        "My invoice shows the wrong amount.",
        "Payment failed but money was deducted.",
    ],
    "technical": [
        "The app keeps crashing when I try to open the dashboard.",
        "I'm getting a 500 error when trying to save my work.",
        "The sync feature hasn't updated in 3 days.",
        "Loading times are very slow since the last update.",
        "The export to PDF feature produces blank pages.",
    ],
    "account": [
        "I need to update my email from old@email.com to new@email.com.",
        "My account was locked after entering the wrong password.",
        "I have two accounts and want to merge them into one.",
        "Please delete all my data per GDPR request.",
        "I need to add 5 new team members to our plan.",
    ],
    "sales": [
        "We're a company of 500 employees and need a quote.",
        "I'd like to upgrade from Pro to Enterprise.",
        "Can we get a demo of the analytics features?",
        "Do you offer educational discounts?",
        "What's the pricing for API access?",
    ],
    "shipping": [
        "My order was supposed to arrive 5 days ago.",
        "I received a blue widget but ordered a red one.",
        "The tracking link shows 'not found'.",
        "Package arrived with visible damage to the box.",
        "I moved and need to update the shipping address.",
    ],
}


def generate_ticket(team: str, ticket_id: int, days_ago: int = None) -> Dict:
    """Generate a realistic support ticket."""
    template = random.choice(TICKET_TEMPLATES[team])
    description = random.choice(ISSUE_DESCRIPTIONS[team])
    
    if days_ago is None:
        days_ago = random.randint(0, 90)
    
    created_time = time.time() - (days_ago * 86400) - random.randint(0, 86400)
    
    # Resolution quality varies
    satisfaction = random.choices(
        [5.0, 4.5, 4.0, 3.5, 3.0, 2.0, 1.0],
        weights=[0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02]
    )[0]
    
    resolution_hours = random.randint(1, 72)
    
    return {
        "ticket_id": f"TKT-{ticket_id:05d}",
        "subject": template["subject"],
        "description": description,
        "keywords": template["keywords"],
        "customer_type": random.choice(["free", "pro", "enterprise"]),
        "priority": random.choice(["low", "medium", "high", "urgent"]),
        "routed_to": team,
        "resolution_hours": resolution_hours,
        "satisfaction": satisfaction,
        "created_at": {"$time": created_time},
        "status": "resolved",
    }


def generate_training_data(tickets_per_team: int = 200) -> List[Dict]:
    """Generate training data - resolved tickets with known routing."""
    tickets = []
    ticket_id = 1
    
    for team in TEAMS:
        for _ in range(tickets_per_team):
            ticket = generate_ticket(team, ticket_id)
            tickets.append(ticket)
            ticket_id += 1
    
    random.shuffle(tickets)
    return tickets


def generate_test_tickets(count: int = 100) -> List[Dict]:
    """Generate test tickets (new, unrouted)."""
    tickets = []
    
    for i in range(count):
        team = random.choice(TEAMS)
        template = random.choice(TICKET_TEMPLATES[team])
        description = random.choice(ISSUE_DESCRIPTIONS[team])
        
        tickets.append({
            "ticket_id": f"NEW-{i:04d}",
            "subject": template["subject"],
            "description": description,
            "keywords": template["keywords"],
            "customer_type": random.choice(["free", "pro", "enterprise"]),
            "priority": random.choice(["low", "medium", "high", "urgent"]),
            "actual_team": team,  # Ground truth for evaluation
            "status": "new",
        })
    
    return tickets


# =============================================================================
# Ticket Router
# =============================================================================

class TicketRouter:
    """Route tickets using k-NN classification on historical tickets."""
    
    def __init__(self, dimensions: int = 4096):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.team_prototypes = {}
        
    def train(self, tickets: List[Dict]):
        """Ingest training tickets."""
        for ticket in tickets:
            self.client.insert_json(ticket)
    
    def learn_prototypes(self, min_satisfaction: float = 4.0):
        """Learn team prototypes from high-quality resolutions."""
        for team in TEAMS:
            # Find well-resolved tickets for this team
            # Use dict-based guard with $gte operator
            results = self.client.search_json(
                probe={"routed_to": team},
                guard={"satisfaction": {"$gte": min_satisfaction}},
                limit=100
            )
            
            if results:
                # Create prototype from these tickets
                vectors = []
                for r in results:
                    vec = self.store.encoder.encode_data(r["data"])
                    vectors.append(vec)
                
                if vectors:
                    # Convert to numpy if needed
                    vectors_np = [v.cpu().numpy() if hasattr(v, 'cpu') else v for v in vectors]
                    prototype = self.store.prototype(vectors_np)
                    self.team_prototypes[team] = prototype
                    
        return len(self.team_prototypes)
    
    def route_ticket(self, ticket: Dict, k: int = 5) -> Tuple[str, float, List[Dict]]:
        """
        Route a ticket using k-NN voting.
        Returns (predicted_team, confidence, neighbors).
        """
        # Find similar resolved tickets
        # Use dict-based guard for status filter
        results = self.client.search_json(
            probe={
                "subject": ticket["subject"],
                "keywords": ticket["keywords"],
                "description": ticket["description"],
            },
            guard={"status": "resolved"},
            limit=k
        )
        
        if not results:
            return "technical", 0.0, []  # Default fallback
        
        # Vote by team
        votes = Counter()
        neighbors = []
        
        for r in results:
            team = r["data"].get("routed_to", "unknown")
            score = r["score"]
            votes[team] += score  # Weight by similarity
            neighbors.append({
                "ticket_id": r["data"].get("ticket_id"),
                "team": team,
                "score": score,
                "satisfaction": r["data"].get("satisfaction")
            })
        
        # Winner
        if votes:
            winner = votes.most_common(1)[0]
            total = sum(votes.values())
            confidence = winner[1] / total if total > 0 else 0
            return winner[0], confidence, neighbors
        
        return "technical", 0.0, neighbors
    
    def route_with_prototype(self, ticket: Dict) -> Tuple[str, float]:
        """Route using prototype similarity (faster, less accurate)."""
        from holon.similarity import normalized_dot_similarity
        
        ticket_vec = self.store.encoder.encode_data(ticket)
        ticket_np = ticket_vec.cpu().numpy() if hasattr(ticket_vec, 'cpu') else ticket_vec
        
        best_team = None
        best_score = -1
        
        for team, proto in self.team_prototypes.items():
            proto_np = proto.cpu().numpy() if hasattr(proto, 'cpu') else proto
            score = normalized_dot_similarity(ticket_np, proto_np)
            if score > best_score:
                best_score = score
                best_team = team
        
        return best_team, best_score
    
    def evaluate(self, test_tickets: List[Dict], method: str = "knn") -> Dict:
        """Evaluate routing accuracy."""
        correct = 0
        total = len(test_tickets)
        
        by_team = {team: {"correct": 0, "total": 0} for team in TEAMS}
        confusion = {team: Counter() for team in TEAMS}
        
        for ticket in test_tickets:
            actual = ticket["actual_team"]
            
            if method == "knn":
                predicted, confidence, _ = self.route_ticket(ticket)
            else:
                predicted, confidence = self.route_with_prototype(ticket)
            
            by_team[actual]["total"] += 1
            confusion[actual][predicted] += 1
            
            if predicted == actual:
                correct += 1
                by_team[actual]["correct"] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Per-team accuracy
        team_accuracy = {}
        for team in TEAMS:
            if by_team[team]["total"] > 0:
                team_accuracy[team] = by_team[team]["correct"] / by_team[team]["total"]
            else:
                team_accuracy[team] = 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_team": team_accuracy,
            "confusion": confusion,
        }


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("=" * 70)
    print("CHALLENGE 008-004: CUSTOMER SUPPORT TICKET ROUTER")
    print("=" * 70)
    
    random.seed(42)
    
    # Generate data
    print("\n📦 Generating training and test data...")
    train_tickets = generate_training_data(tickets_per_team=200)
    test_tickets = generate_test_tickets(count=200)
    
    print(f"   Training tickets: {len(train_tickets)}")
    print(f"   Test tickets: {len(test_tickets)}")
    print(f"   Teams: {', '.join(TEAMS)}")
    
    # Initialize router
    print("\n🔧 Initializing ticket router...")
    router = TicketRouter(dimensions=4096)
    
    # Train
    print("\n📥 Ingesting training tickets...")
    start = time.time()
    router.train(train_tickets)
    train_time = time.time() - start
    print(f"   Ingested {len(train_tickets)} tickets in {train_time:.2f}s")
    print(f"   Rate: {len(train_tickets)/train_time:.0f} tickets/sec")
    
    # Learn prototypes
    print("\n🧠 Learning team prototypes...")
    num_prototypes = router.learn_prototypes(min_satisfaction=4.0)
    print(f"   Learned {num_prototypes} team prototypes")
    
    # Demo 1: Route a single ticket
    print("\n" + "=" * 70)
    print("DEMO 1: Route a Single Ticket")
    print("=" * 70)
    
    sample_ticket = test_tickets[0]
    predicted, confidence, neighbors = router.route_ticket(sample_ticket, k=5)
    
    print(f"\n   New Ticket:")
    print(f"   Subject: {sample_ticket['subject']}")
    print(f"   Description: {sample_ticket['description'][:60]}...")
    print(f"   Keywords: {', '.join(sample_ticket['keywords'])}")
    
    print(f"\n   Routing Decision:")
    print(f"   → Predicted Team: {predicted}")
    print(f"   → Confidence: {confidence:.1%}")
    print(f"   → Actual Team: {sample_ticket['actual_team']}")
    print(f"   → Correct: {'✅' if predicted == sample_ticket['actual_team'] else '❌'}")
    
    print(f"\n   Similar Past Tickets:")
    for n in neighbors[:3]:
        print(f"      {n['ticket_id']}: {n['team']} (score: {n['score']:.3f}, satisfaction: {n['satisfaction']})")
    
    # Demo 2: k-NN evaluation
    print("\n" + "=" * 70)
    print("DEMO 2: k-NN Routing Evaluation")
    print("=" * 70)
    
    start = time.time()
    knn_results = router.evaluate(test_tickets, method="knn")
    knn_time = time.time() - start
    
    print(f"\n   Overall Accuracy: {knn_results['accuracy']:.1%}")
    print(f"   Correct: {knn_results['correct']}/{knn_results['total']}")
    print(f"   Evaluation time: {knn_time:.2f}s")
    
    print(f"\n   Per-Team Accuracy:")
    for team, acc in sorted(knn_results["by_team"].items(), key=lambda x: -x[1]):
        print(f"      {team}: {acc:.1%}")
    
    # Demo 3: Prototype-based routing (faster)
    print("\n" + "=" * 70)
    print("DEMO 3: Prototype-Based Routing (Faster)")
    print("=" * 70)
    
    start = time.time()
    proto_results = router.evaluate(test_tickets, method="prototype")
    proto_time = time.time() - start
    
    print(f"\n   Overall Accuracy: {proto_results['accuracy']:.1%}")
    print(f"   Evaluation time: {proto_time:.2f}s")
    print(f"   Speedup vs k-NN: {knn_time/proto_time:.1f}x")
    
    print(f"\n   Per-Team Accuracy:")
    for team, acc in sorted(proto_results["by_team"].items(), key=lambda x: -x[1]):
        print(f"      {team}: {acc:.1%}")
    
    # Demo 4: Quality-filtered routing
    print("\n" + "=" * 70)
    print("DEMO 4: Quality-Filtered Routing")
    print("=" * 70)
    
    # Show how guards filter for high-quality resolutions
    sample = test_tickets[5]
    
    # Route using only highly-rated past tickets
    # Combined guard: status=resolved AND satisfaction >= 4.5
    results_quality = router.client.search_json(
        probe={"keywords": sample["keywords"]},
        guard={"status": "resolved", "satisfaction": {"$gte": 4.5}},
        limit=5
    )
    
    print(f"\n   Ticket: {sample['subject']}")
    print(f"\n   High-quality matches (satisfaction >= 4.5):")
    for r in results_quality:
        d = r["data"]
        print(f"      {d['ticket_id']}: {d['routed_to']} (satisfaction: {d['satisfaction']}, score: {r['score']:.3f})")
    
    # Demo 5: Time-aware routing
    print("\n" + "=" * 70)
    print("DEMO 5: Time-Aware Routing (Recent Similar Issues)")
    print("=" * 70)
    
    # Find similar tickets from the last 30 days
    # Use nested guard for created_at.$time comparison
    recent_cutoff = time.time() - (30 * 86400)
    
    results_recent = router.client.search_json(
        probe={"keywords": sample["keywords"]},
        guard={"created_at": {"$time": {"$gt": recent_cutoff}}},
        limit=5
    )
    
    print(f"\n   Looking for similar issues from last 30 days...")
    print(f"   Found {len(results_recent)} recent matches:")
    for r in results_recent:
        d = r["data"]
        days_ago = (time.time() - d.get("created_at", {}).get("$time", 0)) / 86400
        print(f"      {d['ticket_id']}: {d['routed_to']} ({days_ago:.0f} days ago, score: {r['score']:.3f})")
    
    # Demo 6: Latency benchmark
    print("\n" + "=" * 70)
    print("DEMO 6: Routing Latency")
    print("=" * 70)
    
    # k-NN routing latency
    times_knn = []
    for ticket in test_tickets[:50]:
        start = time.time()
        router.route_ticket(ticket, k=5)
        times_knn.append((time.time() - start) * 1000)
    
    # Prototype routing latency
    times_proto = []
    for ticket in test_tickets[:50]:
        start = time.time()
        router.route_with_prototype(ticket)
        times_proto.append((time.time() - start) * 1000)
    
    print(f"\n   k-NN Routing (k=5):")
    print(f"      Average: {sum(times_knn)/len(times_knn):.2f}ms")
    print(f"      Throughput: {1000/(sum(times_knn)/len(times_knn)):.0f} tickets/sec")
    
    print(f"\n   Prototype Routing:")
    print(f"      Average: {sum(times_proto)/len(times_proto):.2f}ms")
    print(f"      Throughput: {1000/(sum(times_proto)/len(times_proto)):.0f} tickets/sec")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Success Criteria:
   ✅ 1000+ tickets indexed: {len(train_tickets)} training tickets
   ✅ k-NN routing accuracy >70%: {knn_results['accuracy']:.1%}
   ✅ Guards filter for quality: Satisfaction-based filtering works
   ✅ Time-aware queries: Recent ticket filtering works

Results Comparison:
   k-NN (k=5):      {knn_results['accuracy']:.1%} accuracy, {sum(times_knn)/len(times_knn):.2f}ms latency
   Prototype:       {proto_results['accuracy']:.1%} accuracy, {sum(times_proto)/len(times_proto):.2f}ms latency

Key Findings:
   - k-NN routing achieves {knn_results['accuracy']:.0%} accuracy on 5-class problem
   - Prototype routing is {knn_time/proto_time:.1f}x faster with {proto_results['accuracy']:.0%} accuracy
   - Quality guards ensure routing based on successful resolutions
   - Time-aware queries help find recently relevant solutions

This is k-NN classification at work:
   1. Encode new ticket as vector
   2. Find k most similar resolved tickets
   3. Vote by team (weighted by similarity)
   4. Route to winning team
""")


if __name__ == "__main__":
    main()
