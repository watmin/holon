#!/usr/bin/env python3
"""
Personal Task Memory System using Holon VSA/HDC

This script demonstrates a fuzzy task management system that can store tasks
with various attributes and retrieve them using similarity-based queries,
guards, negations, and wildcards.
"""

import json
import uuid
from datetime import datetime, timedelta

from holon import CPUStore, HolonClient


def create_sample_tasks():
    """Generate 30 realistic sample tasks with varied attributes."""

    # Helper function to generate future dates
    def future_date(days_from_now):
        return (datetime.now() + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

    tasks = [
        # Work-related tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Prepare quarterly sales presentation",
            "project": "work",
            "priority": "high",
            "due": future_date(3),
            "tags": ["presentation", "sales", "urgent"],
            "context": ["computer", "meeting"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Code review for authentication module",
            "project": "work",
            "priority": "medium",
            "due": future_date(7),
            "tags": ["code-review", "security", "backend"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Update documentation for API endpoints",
            "project": "work",
            "priority": "low",
            "due": future_date(14),
            "tags": ["documentation", "api"],
            "context": ["computer"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Schedule team building event",
            "project": "work",
            "priority": "medium",
            "due": future_date(21),
            "tags": ["team-building", "planning"],
            "context": ["phone", "computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Fix critical bug in payment processing",
            "project": "work",
            "priority": "high",
            "due": future_date(1),
            "tags": ["bug-fix", "payment", "urgent", "critical"],
            "context": ["computer"],
            "status": "todo",
        },
        # Personal tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Grocery shopping for the week",
            "project": "personal",
            "priority": "medium",
            "due": future_date(2),
            "tags": ["shopping", "food"],
            "context": ["errand", "car"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Call dentist for checkup appointment",
            "project": "personal",
            "priority": "medium",
            "due": future_date(5),
            "tags": ["health", "appointment"],
            "context": ["phone"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Renew car insurance",
            "project": "personal",
            "priority": "high",
            "due": future_date(30),
            "tags": ["insurance", "car", "renewal"],
            "context": ["computer", "phone"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Clean out garage",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["cleaning", "organization"],
            "context": ["home"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Read 'The Phoenix Project'",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["reading", "learning"],
            "context": ["home"],
            "status": "todo",
        },
        # Side project tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Design logo for indie game project",
            "project": "side",
            "priority": "medium",
            "due": future_date(10),
            "tags": ["design", "gaming", "creative"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Learn Rust programming language",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["learning", "programming", "rust"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Build portfolio website",
            "project": "side",
            "priority": "high",
            "due": future_date(45),
            "tags": ["web-development", "portfolio"],
            "context": ["computer"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Research AI model deployment strategies",
            "project": "side",
            "priority": "medium",
            "due": future_date(20),
            "tags": ["research", "ai", "ml"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Contribute to open source project",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["open-source", "contribution"],
            "context": ["computer"],
            "status": "todo",
        },
        # More work tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Prepare performance review self-assessment",
            "project": "work",
            "priority": "medium",
            "due": future_date(25),
            "tags": ["performance-review", "self-assessment"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Organize team knowledge sharing session",
            "project": "work",
            "priority": "low",
            "due": future_date(35),
            "tags": ["team-building", "knowledge-sharing"],
            "context": ["meeting", "computer"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Migrate legacy database to new schema",
            "project": "work",
            "priority": "high",
            "due": future_date(60),
            "tags": ["database", "migration", "backend"],
            "context": ["computer"],
            "status": "todo",
        },
        # More personal tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Plan summer vacation itinerary",
            "project": "personal",
            "priority": "low",
            "due": future_date(90),
            "tags": ["vacation", "planning", "travel"],
            "context": ["computer", "phone"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Buy birthday gift for sister",
            "project": "personal",
            "priority": "medium",
            "due": future_date(12),
            "tags": ["shopping", "gift", "birthday"],
            "context": ["errand", "online"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Update resume and LinkedIn profile",
            "project": "personal",
            "priority": "medium",
            "due": future_date(40),
            "tags": ["career", "resume", "linkedin"],
            "context": ["computer"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Deep clean apartment",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["cleaning", "home"],
            "context": ["home"],
            "status": "todo",
        },
        # More side project tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Write technical blog post about VSA",
            "project": "side",
            "priority": "medium",
            "due": future_date(15),
            "tags": ["writing", "blog", "technical", "vsa"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Create YouTube tutorial series",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["video", "tutorial", "content-creation"],
            "context": ["computer", "recording"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Build mobile app prototype",
            "project": "side",
            "priority": "high",
            "due": future_date(50),
            "tags": ["mobile", "app-development", "prototype"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Attend local tech meetup",
            "project": "side",
            "priority": "low",
            "due": future_date(18),
            "tags": ["networking", "meetup", "tech"],
            "context": ["social", "outdoor"],
            "status": "todo",
        },
        # Final work tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Conduct security audit of application",
            "project": "work",
            "priority": "high",
            "due": future_date(8),
            "tags": ["security", "audit", "compliance"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Optimize database query performance",
            "project": "work",
            "priority": "medium",
            "due": future_date(22),
            "tags": ["performance", "database", "optimization"],
            "context": ["computer"],
            "status": "waiting",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Create user onboarding flow",
            "project": "work",
            "priority": "high",
            "due": future_date(28),
            "tags": ["ux", "onboarding", "user-experience"],
            "context": ["computer", "design"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Set up automated testing pipeline",
            "project": "work",
            "priority": "medium",
            "due": future_date(32),
            "tags": ["testing", "automation", "ci-cd"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Document API design decisions",
            "project": "work",
            "priority": "low",
            "due": None,
            "tags": ["documentation", "api", "architecture"],
            "context": ["computer"],
            "status": "todo",
        },
    ]

    return tasks


def ingest_tasks(client, tasks):
    """Ingest tasks into the Holon store via client using batch operations for better performance."""
    print(f"📥 Ingesting {len(tasks)} tasks into Holon memory...")

    # Use batch insert for much better performance
    ids = client.insert_batch_json(tasks)
    print(f"  ✓ Batch inserted {len(tasks)} tasks in one operation")

    print("✅ All tasks ingested successfully!")


def query_tasks(client, query, description, top_k=10, guard=None, negations=None, threshold=0.0):
    """Query tasks and display results."""
    print(f"\n🔍 {description}")
    print(f"Query: {query}")
    if guard:
        print(f"Guard: {guard}")
    if negations:
        print(f"Negations: {negations}")
    if threshold > 0.0:
        print(f"Threshold: {threshold}")

    try:
        # Parse query if it's a JSON string, otherwise use as dict
        if isinstance(query, str):
            import json
            query_dict = json.loads(query)
        else:
            query_dict = query

        results = client.search_json(
            query_dict, guard=guard, negations=negations, limit=top_k, threshold=threshold
        )

        if not results:
            print("  ❌ No matching tasks found")
            return

        print(
            f"  ✅ Found {len(results)} matching tasks (showing top {min(top_k, len(results))}):"
        )

        for i, result in enumerate(results):
            task = result["data"]
            score = result["score"]
            print(f"\n  {i+1}. [{score:.3f}] {task['title']}")
            print(
                f"     Project: {task['project']} | Priority: {task['priority']} | "
                f"Status: {task['status']}"
            )
            if task.get("due"):
                print(f"     Due: {task['due']}")
            if task.get("tags"):
                print(f"     Tags: {', '.join(task['tags'])}")
            if task.get("context"):
                print(f"     Context: {', '.join(task['context'])}")

    except Exception as e:
        print(f"  ❌ Query failed: {e}")


def main():
    """Main demonstration function."""
    print("🧠 Personal Task Memory System Demo")
    print("=" * 50)

    # Initialize Holon store and client
    print("🚀 Initializing Holon CPUStore and Client...")
    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)
    print("✅ Store and client initialized with 16,000 dimensions")

    # Create and ingest sample tasks
    tasks = create_sample_tasks()
    ingest_tasks(client, tasks)

    # Demonstrate various query types
    print("\n" + "=" * 50)
    print("🧪 QUERY DEMONSTRATIONS")
    print("=" * 50)

    # 1. Fuzzy similarity query - find tasks with similar titles
    query_tasks(
        client,
        {"title": "prepare presentation", "project": "work", "priority": "high"},
        "1. FUZZY SIMILARITY: Tasks similar to preparing presentations",
        top_k=5  # Focus on top 5 most relevant
    )

    # 2. Guard query (high priority tasks)
    query_tasks(client, {"priority": "high"}, "2. GUARDS: All high-priority tasks")

    # 3. Negation query (tasks NOT in work project)
    query_tasks(
        client,
        {"project": "work"},
        "3. NEGATIONS: Tasks NOT in work project",
        negations={"project": {"$not": "work"}},
    )

    # 4. Wildcard query (any priority level)
    query_tasks(
        client, {"priority": {"$any": True}}, "4. WILDCARDS: Tasks with any priority"
    )

    # 5. Disjunction query (work OR personal projects)
    query_tasks(
        client,
        {"$or": [{"project": "work"}, {"project": "personal"}]},
        "5. DISJUNCTIONS: Tasks in work OR personal projects",
    )

    # 6. Combined query with guard (urgent tasks NOT done)
    query_tasks(
        client,
        {"tags": ["urgent"]},
        "6. COMBINED: Urgent tasks that are NOT done",
        negations={"status": {"$not": "done"}},
    )

    # 7. Context-based query (computer tasks)
    query_tasks(
        client,
        {"context": ["computer"]},
        "7. CONTEXT FILTERING: Tasks that can be done on computer",
    )

    # 8. Status-based query (active tasks)
    query_tasks(
        client, {"status": "todo"}, "8. STATUS FILTERING: Tasks that are still todo"
    )

    # 9. Complex query: Medium/high priority side projects NOT waiting
    query_tasks(
        client,
        {"project": "side"},
        "9. COMPLEX: Side projects NOT in waiting status",
        guard={"$or": [{"priority": "medium"}, {"priority": "high"}]},
        negations={"status": {"$not": "waiting"}},
    )

    # 10. Tag-based similarity (learning-related tasks)
    query_tasks(
        client, {"tags": ["learning"], "project": "side"},
        "10. TAG SIMILARITY: Side project learning tasks",
        top_k=3
    )

    # 11. Advanced OR logic: High priority work tasks OR urgent personal tasks
    query_tasks(
        client,
        {},
        "11. ADVANCED OR LOGIC: High priority work OR urgent personal tasks",
        guard={
            "$or": [
                {"project": "work", "priority": "high"},
                {"project": "personal", "tags": ["urgent"]}
            ]
        },
        top_k=5
    )

    # 12. Complex negation: Active tasks NOT in computer/phone contexts (errand-only tasks)
    query_tasks(
        client,
        {"status": "todo"},
        "12. COMPLEX NEGATION: Active tasks NOT computer/phone (errand-only)",
        negations={
            "context": {"$not_contains": "computer"},
            "context": {"$not_contains": "phone"}
        },
        top_k=5
    )

    # 13. Wildcard with complex guards: Any priority side project that's NOT waiting
    query_tasks(
        client,
        {"project": "side", "priority": {"$any": True}},
        "13. WILDCARD + GUARDS: Any priority side projects NOT waiting",
        negations={"status": {"$not": "waiting"}},
        top_k=5
    )

    # ========================================
    # NEW: Demonstrate new VSA primitives
    # ========================================
    print("\n" + "=" * 50)
    print("🆕 NEW VSA PRIMITIVE DEMONSTRATIONS")
    print("=" * 50)

    demonstrate_new_primitives(store, client, tasks)

    print("\n" + "=" * 50)
    print("🎉 Task Memory Demo Complete!")
    print(
        "Holon successfully demonstrated fuzzy retrieval, guards, negations, wildcards, "
        "advanced OR logic, AND new VSA primitives (prototype, difference, negate, amplify)"
    )
    print("=" * 50)


def demonstrate_new_primitives(store, client, tasks):
    """Demonstrate the new VSA primitives added during Challenge 004."""
    import numpy as np
    from holon.similarity import normalized_dot_similarity as cosine_similarity

    print("\n🧬 1. PROTOTYPE: Extract common patterns from task categories")
    print("-" * 50)

    # Encode all tasks and group by project
    work_vecs = []
    personal_vecs = []
    side_vecs = []

    for task in tasks:
        vec = store.encoder.encode_data(task)
        if task["project"] == "work":
            work_vecs.append(vec)
        elif task["project"] == "personal":
            personal_vecs.append(vec)
        elif task["project"] == "side":
            side_vecs.append(vec)

    # Create prototypes for each category
    work_proto = store.prototype(work_vecs) if work_vecs else None
    personal_proto = store.prototype(personal_vecs) if personal_vecs else None
    side_proto = store.prototype(side_vecs) if side_vecs else None

    print(f"  Created prototypes from: work={len(work_vecs)}, personal={len(personal_vecs)}, side={len(side_vecs)} tasks")

    # Check how well prototypes separate categories
    if work_proto is not None and personal_proto is not None:
        sep = cosine_similarity(work_proto, personal_proto)
        print(f"  Prototype separation (work vs personal): {sep:.4f}")
        print(f"  (Lower = more distinct categories)")

    # Use prototypes to classify new queries
    test_query = {"title": "submit expense report", "tags": ["finance"]}
    test_vec = store.encoder.encode_data(test_query)

    if work_proto is not None and personal_proto is not None and side_proto is not None:
        work_sim = cosine_similarity(test_vec, work_proto)
        personal_sim = cosine_similarity(test_vec, personal_proto)
        side_sim = cosine_similarity(test_vec, side_proto)

        print(f"\n  Classifying query: {test_query}")
        print(f"    → work similarity:     {work_sim:.4f}")
        print(f"    → personal similarity: {personal_sim:.4f}")
        print(f"    → side similarity:     {side_sim:.4f}")
        best = max([("work", work_sim), ("personal", personal_sim), ("side", side_sim)], key=lambda x: x[1])
        print(f"    → Best match: {best[0]}")

    print("\n🔬 2. DIFFERENCE: Find what makes tasks unique")
    print("-" * 50)

    # Compare two similar tasks
    task1 = tasks[0]  # First work task
    task2 = tasks[5]  # First personal task

    vec1 = store.encoder.encode_data(task1)
    vec2 = store.encoder.encode_data(task2)

    # Compute difference
    diff = store.difference(vec1, vec2)

    print(f"  Task 1: {task1['title'][:40]}... ({task1['project']})")
    print(f"  Task 2: {task2['title'][:40]}... ({task2['project']})")
    print(f"  Difference vector norm: {np.linalg.norm(diff):.1f}")

    # The difference should be more similar to task1 than task2
    sim_to_1 = cosine_similarity(diff, vec1)
    sim_to_2 = cosine_similarity(diff, vec2)
    print(f"  Difference → Task1 similarity: {sim_to_1:.4f}")
    print(f"  Difference → Task2 similarity: {sim_to_2:.4f}")

    print("\n📢 3. AMPLIFY: Boost specific attributes in search")
    print("-" * 50)

    # Amplify "urgent" attribute in a search
    base_query = {"project": "work", "status": "todo"}
    base_vec = store.encoder.encode_data(base_query)

    urgent_vec = store.encoder.encode_data({"tags": ["urgent"]})
    amplified = store.amplify(base_vec, urgent_vec, strength=2.0)

    print(f"  Base query: {base_query}")
    print(f"  Amplifying: 'urgent' tag with strength=2.0")

    # Find tasks similar to amplified query
    print("\n  Top 3 matches with amplified 'urgent':")
    scored = []
    for i, task in enumerate(tasks):
        task_vec = store.encoder.encode_data(task)
        score = cosine_similarity(amplified, task_vec)
        scored.append((score, i, task))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, _, task in scored[:3]:
        urgent_marker = "🔴" if "urgent" in task.get("tags", []) else "  "
        print(f"    {urgent_marker} [{score:.3f}] {task['title'][:50]}")

    print("\n🚫 4. NEGATE: Remove unwanted components from search")
    print("-" * 50)

    # Start with a broad query
    broad_query = {"project": "work"}
    broad_vec = store.encoder.encode_data(broad_query)

    # Negate "meeting" context
    meeting_vec = store.encoder.encode_data({"context": ["meeting"]})
    negated = store.negate(broad_vec, meeting_vec)

    print(f"  Broad query: {broad_query}")
    print(f"  Negating: 'meeting' context")

    print("\n  Results after negation (should deprioritize meeting tasks):")
    scored = []
    for i, task in enumerate(tasks):
        if task["project"] == "work":
            task_vec = store.encoder.encode_data(task)
            score_before = cosine_similarity(broad_vec, task_vec)
            score_after = cosine_similarity(negated, task_vec)
            scored.append((score_after, score_before, i, task))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score_after, score_before, _, task in scored[:5]:
        has_meeting = "meeting" in task.get("context", [])
        marker = "📅" if has_meeting else "  "
        delta = score_after - score_before
        print(f"    {marker} [{score_after:.3f}] (Δ{delta:+.3f}) {task['title'][:45]}")

    print("\n🎨 5. BLEND: Create weighted combination of criteria")
    print("-" * 50)

    # Blend work priority with learning interest
    work_high = store.encoder.encode_data({"project": "work", "priority": "high"})
    learning = store.encoder.encode_data({"tags": ["learning"]})

    # 70% work focus, 30% learning interest
    blended = store.blend(work_high, learning, alpha=0.7)

    print("  Blending: 70% (work + high priority) + 30% (learning)")
    print("\n  Top 5 blended matches:")

    scored = []
    for i, task in enumerate(tasks):
        task_vec = store.encoder.encode_data(task)
        score = cosine_similarity(blended, task_vec)
        scored.append((score, i, task))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, _, task in scored[:5]:
        is_work_high = task["project"] == "work" and task["priority"] == "high"
        has_learning = "learning" in task.get("tags", [])
        markers = ""
        if is_work_high:
            markers += "💼"
        if has_learning:
            markers += "📚"
        print(f"    {markers:3} [{score:.3f}] {task['title'][:45]}")


if __name__ == "__main__":
    main()
