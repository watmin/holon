#!/usr/bin/env python3
"""
Personal Task Memory System - HTTP API Version

This demonstrates the task memory system working via Holon's HTTP API,
proving the solution works with both in-memory and remote deployments.
"""

import json
from datetime import datetime, timedelta

import requests

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"


class HTTPTaskStore:
    """HTTP client for task memory that works with Holon server."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_prefix = API_PREFIX
        self.tasks = {}  # Local cache

    def health_check(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(
                f"{self.base_url}{self.api_prefix}/health", timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def insert_task(self, task: dict) -> str:
        """Insert a task via HTTP API."""
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/items",
            json={"data": json.dumps(task), "data_type": "json"},
            timeout=10,
        )
        response.raise_for_status()
        task_id = response.json()["id"]
        self.tasks[task_id] = task
        return task_id

    def batch_insert_tasks(self, tasks: list) -> list:
        """Batch insert tasks via HTTP API."""
        items = [json.dumps(task) for task in tasks]
        response = requests.post(
            f"{self.base_url}{self.api_prefix}/items/batch",
            json={"items": items, "data_type": "json"},
            timeout=30,
        )
        response.raise_for_status()
        ids = response.json()["ids"]
        for i, task_id in enumerate(ids):
            self.tasks[task_id] = tasks[i]
        return ids

    def search_tasks(
        self,
        query: dict,
        limit: int = 10,
        guard: dict = None,
        negations: dict = None,
        threshold: float = 0.0,
    ) -> list:
        """Search tasks via HTTP API."""
        payload = {
            "probe": json.dumps(query),
            "data_type": "json",
            "top_k": limit,
            "threshold": threshold,
        }
        if guard:
            payload["guard"] = json.dumps(guard)
        # Note: negations support depends on server implementation

        response = requests.post(
            f"{self.base_url}{self.api_prefix}/search",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()["results"]

        # Enrich with cached task data
        enriched = []
        for r in results:
            task_data = self.tasks.get(r["id"], r.get("data", {}))
            enriched.append({"id": r["id"], "score": r["score"], "data": task_data})
        return enriched


def create_sample_tasks():
    """Generate sample tasks."""

    def future_date(days_from_now):
        return (datetime.now() + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

    return [
        {
            "id": "task-001",
            "title": "Prepare quarterly sales presentation",
            "project": "work",
            "priority": "high",
            "due": future_date(3),
            "tags": ["presentation", "sales", "urgent"],
            "context": ["computer", "meeting"],
            "status": "todo",
        },
        {
            "id": "task-002",
            "title": "Code review for authentication module",
            "project": "work",
            "priority": "medium",
            "due": future_date(7),
            "tags": ["code-review", "security", "backend"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": "task-003",
            "title": "Grocery shopping for the week",
            "project": "personal",
            "priority": "medium",
            "due": future_date(2),
            "tags": ["shopping", "food"],
            "context": ["errand", "car"],
            "status": "todo",
        },
        {
            "id": "task-004",
            "title": "Learn Rust programming language",
            "project": "side",
            "priority": "low",
            "due": None,
            "tags": ["learning", "programming", "rust"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": "task-005",
            "title": "Fix critical bug in payment processing",
            "project": "work",
            "priority": "high",
            "due": future_date(1),
            "tags": ["bug-fix", "payment", "urgent", "critical"],
            "context": ["computer"],
            "status": "todo",
        },
        {
            "id": "task-006",
            "title": "Build portfolio website",
            "project": "side",
            "priority": "high",
            "due": future_date(45),
            "tags": ["web-development", "portfolio"],
            "context": ["computer"],
            "status": "waiting",
        },
    ]


def main():
    """Main demonstration."""
    print("🧠 Personal Task Memory System - HTTP API Demo")
    print("=" * 60)

    store = HTTPTaskStore()

    # Health check
    print("\n🔗 Checking Holon HTTP service...")
    if not store.health_check():
        print("❌ Server not running. Start with:")
        print("   ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
        return

    health = requests.get(f"{BASE_URL}{API_PREFIX}/health").json()
    print(f"✅ Connected: {health['status']} | Backend: {health['backend']}")

    # Create and insert tasks
    tasks = create_sample_tasks()
    print(f"\n📥 Inserting {len(tasks)} tasks via HTTP...")
    ids = store.batch_insert_tasks(tasks)
    print(f"✅ Inserted {len(ids)} tasks")

    # Query demonstrations
    print("\n" + "=" * 60)
    print("🔍 QUERY DEMONSTRATIONS VIA HTTP")
    print("=" * 60)

    # 1. Fuzzy similarity
    print("\n1. FUZZY SIMILARITY: Tasks similar to 'prepare presentation'")
    results = store.search_tasks(
        {"title": "prepare presentation", "project": "work"}, limit=3
    )
    for r in results:
        print(f"   [{r['score']:.3f}] {r['data'].get('title', 'N/A')}")

    # 2. Priority filter
    print("\n2. HIGH PRIORITY: Tasks with high priority")
    results = store.search_tasks({"priority": "high"}, limit=5)
    for r in results:
        task = r["data"]
        print(f"   [{r['score']:.3f}] {task.get('title', 'N/A')} ({task.get('project')})")

    # 3. Project filter
    print("\n3. SIDE PROJECTS: Tasks in side project")
    results = store.search_tasks({"project": "side"}, limit=5)
    for r in results:
        task = r["data"]
        print(f"   [{r['score']:.3f}] {task.get('title', 'N/A')}")

    # 4. Tag search
    print("\n4. URGENT TASKS: Tasks with urgent tag")
    results = store.search_tasks({"tags": ["urgent"]}, limit=5)
    for r in results:
        task = r["data"]
        print(f"   [{r['score']:.3f}] {task.get('title', 'N/A')}")

    # 5. Context search
    print("\n5. COMPUTER TASKS: Tasks for computer context")
    results = store.search_tasks({"context": ["computer"]}, limit=5)
    for r in results:
        task = r["data"]
        print(f"   [{r['score']:.3f}] {task.get('title', 'N/A')}")

    print("\n" + "=" * 60)
    print("✅ HTTP API DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey achievements:")
    print("   - Tasks stored via HTTP POST to /api/v1/items")
    print("   - Batch insert via /api/v1/items/batch")
    print("   - Similarity search via /api/v1/search")
    print("   - Same queries work in-memory OR via HTTP")
    print("   - Ready for production deployment")


if __name__ == "__main__":
    main()
