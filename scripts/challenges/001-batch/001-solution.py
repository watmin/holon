#!/usr/bin/env python3
"""
Personal Task Memory System using Holon VSA/HDC

This script demonstrates a fuzzy task management system that can store tasks
with various attributes and retrieve them using similarity-based queries,
guards, negations, and wildcards.
"""

import uuid
import json
from datetime import datetime, timedelta
from holon import CPUStore


def create_sample_tasks():
    """Generate 30 realistic sample tasks with varied attributes."""

    # Helper function to generate future dates
    def future_date(days_from_now):
        return (datetime.now() + timedelta(days=days_from_now)).strftime('%Y-%m-%d')

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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Code review for authentication module",
            "project": "work",
            "priority": "medium",
            "due": future_date(7),
            "tags": ["code-review", "security", "backend"],
            "context": ["computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Update documentation for API endpoints",
            "project": "work",
            "priority": "low",
            "due": future_date(14),
            "tags": ["documentation", "api"],
            "context": ["computer"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Schedule team building event",
            "project": "work",
            "priority": "medium",
            "due": future_date(21),
            "tags": ["team-building", "planning"],
            "context": ["phone", "computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Fix critical bug in payment processing",
            "project": "work",
            "priority": "high",
            "due": future_date(1),
            "tags": ["bug-fix", "payment", "urgent", "critical"],
            "context": ["computer"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Call dentist for checkup appointment",
            "project": "personal",
            "priority": "medium",
            "due": future_date(5),
            "tags": ["health", "appointment"],
            "context": ["phone"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Renew car insurance",
            "project": "personal",
            "priority": "high",
            "due": future_date(30),
            "tags": ["insurance", "car", "renewal"],
            "context": ["computer", "phone"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Clean out garage",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["cleaning", "organization"],
            "context": ["home"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Read 'The Phoenix Project'",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["reading", "learning"],
            "context": ["home"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Learn Rust programming language",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["learning", "programming", "rust"],
            "context": ["computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Build portfolio website",
            "project": "side",
            "priority": "high",
            "due": future_date(45),
            "tags": ["web-development", "portfolio"],
            "context": ["computer"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Research AI model deployment strategies",
            "project": "side",
            "priority": "medium",
            "due": future_date(20),
            "tags": ["research", "ai", "ml"],
            "context": ["computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Contribute to open source project",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["open-source", "contribution"],
            "context": ["computer"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Organize team knowledge sharing session",
            "project": "work",
            "priority": "low",
            "due": future_date(35),
            "tags": ["team-building", "knowledge-sharing"],
            "context": ["meeting", "computer"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Migrate legacy database to new schema",
            "project": "work",
            "priority": "high",
            "due": future_date(60),
            "tags": ["database", "migration", "backend"],
            "context": ["computer"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Buy birthday gift for sister",
            "project": "personal",
            "priority": "medium",
            "due": future_date(12),
            "tags": ["shopping", "gift", "birthday"],
            "context": ["errand", "online"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Update resume and LinkedIn profile",
            "project": "personal",
            "priority": "medium",
            "due": future_date(40),
            "tags": ["career", "resume", "linkedin"],
            "context": ["computer"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Deep clean apartment",
            "project": "personal",
            "priority": "low",
            "due": None,
            "tags": ["cleaning", "home"],
            "context": ["home"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Create YouTube tutorial series",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["video", "tutorial", "content-creation"],
            "context": ["computer", "recording"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Build mobile app prototype",
            "project": "side",
            "priority": "high",
            "due": future_date(50),
            "tags": ["mobile", "app-development", "prototype"],
            "context": ["computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Attend local tech meetup",
            "project": "side",
            "priority": "low",
            "due": future_date(18),
            "tags": ["networking", "meetup", "tech"],
            "context": ["social", "outdoor"],
            "status": "todo"
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
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Optimize database query performance",
            "project": "work",
            "priority": "medium",
            "due": future_date(22),
            "tags": ["performance", "database", "optimization"],
            "context": ["computer"],
            "status": "waiting"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Create user onboarding flow",
            "project": "work",
            "priority": "high",
            "due": future_date(28),
            "tags": ["ux", "onboarding", "user-experience"],
            "context": ["computer", "design"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Set up automated testing pipeline",
            "project": "work",
            "priority": "medium",
            "due": future_date(32),
            "tags": ["testing", "automation", "ci-cd"],
            "context": ["computer"],
            "status": "todo"
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Document API design decisions",
            "project": "work",
            "priority": "low",
            "due": None,
            "tags": ["documentation", "api", "architecture"],
            "context": ["computer"],
            "status": "todo"
        }
    ]

    return tasks


def ingest_tasks(store, tasks):
    """Ingest tasks into the Holon store."""
    print(f"📥 Ingesting {len(tasks)} tasks into Holon memory...")

    for i, task in enumerate(tasks):
        # Convert to JSON string for Holon ingestion
        task_json = json.dumps(task)
        task_id = store.insert(task_json)
        if (i + 1) % 10 == 0:
            print(f"  ✓ Ingested {i + 1}/{len(tasks)} tasks")

    print("✅ All tasks ingested successfully!")


def query_tasks(store, query, description, top_k=10, guard=None, negations=None):
    """Query tasks and display results."""
    print(f"\n🔍 {description}")
    print(f"Query: {query}")
    if guard:
        print(f"Guard: {guard}")
    if negations:
        print(f"Negations: {negations}")

    try:
        results = store.query(query, guard=guard, negations=negations, top_k=top_k, threshold=0.0)

        if not results:
            print("  ❌ No matching tasks found")
            return

        print(f"  ✅ Found {len(results)} matching tasks (showing top {min(top_k, len(results))}):")

        for i, (task_id, score, task_data) in enumerate(results):
            task = task_data  # Already parsed JSON
            print(f"\n  {i+1}. [{score:.3f}] {task['title']}")
            print(f"     Project: {task['project']} | Priority: {task['priority']} | Status: {task['status']}")
            if task.get('due'):
                print(f"     Due: {task['due']}")
            if task.get('tags'):
                print(f"     Tags: {', '.join(task['tags'])}")
            if task.get('context'):
                print(f"     Context: {', '.join(task['context'])}")

    except Exception as e:
        print(f"  ❌ Query failed: {e}")


def main():
    """Main demonstration function."""
    print("🧠 Personal Task Memory System Demo")
    print("=" * 50)

    # Initialize Holon store
    print("🚀 Initializing Holon CPUStore...")
    store = CPUStore(dimensions=16000)
    print("✅ Store initialized with 16,000 dimensions")

    # Create and ingest sample tasks
    tasks = create_sample_tasks()
    ingest_tasks(store, tasks)

    # Demonstrate various query types
    print("\n" + "=" * 50)
    print("🧪 QUERY DEMONSTRATIONS")
    print("=" * 50)

    # 1. Fuzzy similarity query
    query_tasks(store,
                '{"title": "prepare presentation"}',
                "1. FUZZY SIMILARITY: Tasks similar to 'prepare presentation'")

    # 2. Guard query (high priority tasks)
    query_tasks(store,
                '{"priority": "high"}',
                "2. GUARDS: All high-priority tasks")

    # 3. Negation query (tasks NOT in work project)
    query_tasks(store,
                '{"project": "work"}',
                "3. NEGATIONS: Tasks NOT in work project",
                negations={"project": {"$not": "work"}})

    # 4. Wildcard query (any priority level)
    query_tasks(store,
                '{"priority": {"$any": true}}',
                "4. WILDCARDS: Tasks with any priority")

    # 5. Disjunction query (work OR personal projects)
    query_tasks(store,
                '{"$or": [{"project": "work"}, {"project": "personal"}]}',
                "5. DISJUNCTIONS: Tasks in work OR personal projects")

    # 6. Combined query with guard (urgent tasks NOT done)
    query_tasks(store,
                '{"tags": ["urgent"]}',
                "6. COMBINED: Urgent tasks that are NOT done",
                negations={"status": {"$not": "done"}})

    # 7. Context-based query (computer tasks)
    query_tasks(store,
                '{"context": ["computer"]}',
                "7. CONTEXT FILTERING: Tasks that can be done on computer")

    # 8. Status-based query (active tasks)
    query_tasks(store,
                '{"status": "todo"}',
                "8. STATUS FILTERING: Tasks that are still todo")

    # 9. Complex query: Medium/high priority side projects NOT waiting
    query_tasks(store,
                '{"project": "side"}',
                "9. COMPLEX: Side projects NOT in waiting status",
                guard={"$or": [{"priority": "medium"}, {"priority": "high"}]},
                negations={"status": {"$not": "waiting"}})

    # 10. Tag-based similarity (learning-related tasks)
    query_tasks(store,
                '{"tags": ["learning"]}',
                "10. TAG SIMILARITY: Learning-related tasks")

    print("\n" + "=" * 50)
    print("🎉 Task Memory Demo Complete!")
    print("Holon successfully demonstrated fuzzy retrieval, guards, negations, and wildcards")
    print("=" * 50)


if __name__ == "__main__":
    main()