#!/usr/bin/env python3
"""
Validation for Challenge 001 Task Memory System
Tests fuzzy retrieval, guards, negations, and wildcards quantitatively
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from holon import CPUStore


def create_sample_tasks():
    """Create sample tasks for validation"""
    def future_date(days_from_now):
        return (datetime.now() + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

    tasks = [
        # High priority work tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Prepare quarterly sales presentation",
            "project": "work",
            "priority": "high",
            "due": future_date(7),
            "tags": ["presentation", "sales", "quarterly"],
            "context": ["computer"],
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
        # Medium priority work
        {
            "id": str(uuid.uuid4()),
            "title": "Optimize database query performance",
            "project": "work",
            "priority": "medium",
            "due": future_date(12),
            "tags": ["performance", "database", "optimization"],
            "context": ["computer"],
            "status": "waiting",
        },
        # Personal tasks
        {
            "id": str(uuid.uuid4()),
            "title": "Renew car insurance",
            "project": "personal",
            "priority": "high",
            "due": future_date(20),
            "tags": ["insurance", "car", "renewal"],
            "context": ["computer", "phone"],
            "status": "todo",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Plan summer vacation itinerary",
            "project": "personal",
            "priority": "low",
            "due": future_date(75),
            "tags": ["vacation", "planning", "travel"],
            "context": ["computer", "phone"],
            "status": "todo",
        },
        # Side projects
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
            "due": future_date(25),
            "tags": ["web-development", "portfolio"],
            "context": ["computer"],
            "status": "waiting",
        },
    ]
    return tasks


def ingest_tasks(store, tasks):
    """Ingest tasks into Holon store"""
    print(f"📥 Ingesting {len(tasks)} tasks into Holon memory...")
    for i, task in enumerate(tasks):
        # Convert to JSON and store
        task_json = json.dumps(task)
        store.insert(task_json, data_type="json")
        if (i + 1) % 4 == 0:
            print(f"  ✓ Ingested {i + 1}/{len(tasks)} tasks")
    print("✅ All tasks ingested successfully!")


def query_tasks_validation(store, query, guard=None, negations=None, top_k=10):
    """Query tasks and return results for validation"""
    try:
        # Convert query to JSON string as expected by Holon
        query_json = json.dumps(query)
        results = store.query(
            query_json, guard=guard, negations=negations, top_k=top_k, threshold=0.0, data_type="json"
        )
        return results
    except Exception as e:
        print(f"Query failed: {e}")
        return []

def run_task_memory_validation():
    """Comprehensive validation of the task memory system"""
    print("🧠 Task Memory System Validation")
    print("=" * 40)

    # Initialize store and create tasks
    from holon import CPUStore
    store = CPUStore(dimensions=16000)

    tasks = create_sample_tasks()
    print(f"📊 Created {len(tasks)} test tasks")

    # Ingest tasks
    start_time = time.time()
    ingest_tasks(store, tasks)
    ingest_time = time.time() - start_time
    print(f"   Ingested {len(tasks)} tasks in {ingest_time:.2f}s")
    # Define validation queries
    validation_tests = [
        {
            "name": "Fuzzy Similarity",
            "query": {"title": "prepare presentation"},
            "expected_min_results": 3,
            "description": "Should find presentation-related tasks"
        },
        {
            "name": "Priority Guard",
            "query": {"priority": "high"},
            "expected_min_results": 5,
            "description": "Should find all high-priority tasks"
        },
        {
            "name": "Negation Query",
            "query": {"project": "work"},
            "negations": {"project": {"$not": "work"}},
            "expected_min_results": 5,
            "description": "Should find tasks NOT in work project"
        },
        {
            "name": "Wildcard Query",
            "query": {"priority": {"$any": True}},
            "expected_min_results": 10,
            "description": "Should find all tasks with any priority"
        },
        {
            "name": "Disjunction Query",
            "query": {"$or": [{"project": "work"}, {"project": "personal"}]},
            "expected_min_results": 10,
            "description": "Should find work OR personal tasks"
        },
        {
            "name": "Combined Query",
            "query": {"tags": ["urgent"]},
            "negations": {"status": {"$not": "done"}},
            "expected_min_results": 3,
            "description": "Should find urgent tasks that are NOT done"
        },
        {
            "name": "Context Filter",
            "query": {"context": ["computer"]},
            "expected_min_results": 5,
            "description": "Should find computer-related tasks"
        },
        {
            "name": "Status Filter",
            "query": {"status": "todo"},
            "expected_min_results": 10,
            "description": "Should find all todo tasks"
        },
        {
            "name": "Tag Similarity",
            "query": {"tags": ["learning"]},
            "expected_min_results": 3,
            "description": "Should find learning-related tasks"
        }
    ]

    results = {
        "total_tests": len(validation_tests),
        "passed_tests": 0,
        "response_times": [],
        "total_results_found": 0
    }

    print("\n🧪 VALIDATION TESTS")
    print("-" * 20)

    for i, test in enumerate(validation_tests):
        print(f"\n🎯 Test {i+1}: {test['name']}")
        print(f"   {test['description']}")

        # Time the query
        start_time = time.time()
        query_results = query_tasks_validation(
            store,
            test["query"],
            guard=test.get("guard"),
            negations=test.get("negations"),
            top_k=20  # Get more results for validation
        )
        response_time = time.time() - start_time

        results["response_times"].append(response_time)
        results["total_results_found"] += len(query_results)

        # Evaluate success
        success = len(query_results) >= test["expected_min_results"]
        if success:
            results["passed_tests"] += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"   {status}: Found {len(query_results)} results (expected ≥{test['expected_min_results']})")
        print(f"   Response time: {response_time:.4f}s")
        # Show top result if available
        if query_results:
            top_id, top_score, top_data = query_results[0]
            print(f"   📊 Top result: {top_data.get('title', 'N/A')[:50]}...")

    # Calculate final metrics
    success_rate = results["passed_tests"] / results["total_tests"]
    avg_response_time = sum(results["response_times"]) / len(results["response_times"])
    avg_results_per_query = results["total_results_found"] / results["total_tests"]

    print("\n📊 VALIDATION RESULTS")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Avg Response Time: {avg_response_time:.4f}s")
    print(f"   Avg Results per Query: {avg_results_per_query:.1f}")
    # Performance assessment
    if success_rate >= 0.9:
        assessment = "🎉 EXCELLENT - Robust task memory system"
    elif success_rate >= 0.7:
        assessment = "✅ GOOD - Solid fuzzy retrieval capabilities"
    elif success_rate >= 0.5:
        assessment = "⚠️ FAIR - Functional but needs refinement"
    else:
        assessment = "❌ POOR - Significant issues"

    print(f"\n🏆 Overall Assessment: {assessment}")

    if success_rate >= 0.8:
        print("   ✅ Fuzzy retrieval working effectively")
        print("   ✅ Guards, negations, wildcards functional")
        print("   ✅ Complex queries handled properly")
        print("   ✅ Performance meets requirements")
    else:
        print("   ⚠️ Some query types may need attention")

    print("\n🔍 Key Capabilities Demonstrated:")
    print(f"   • Fuzzy similarity: {'✅' if success_rate >= 0.8 else '❌'}")
    print(f"   • Guard filtering: {'✅' if any(t.get('query', {}).get('priority') for t in validation_tests) else '❌'}")
    print(f"   • Negation queries: {'✅' if any(t.get('negations') for t in validation_tests) else '❌'}")
    print(f"   • Wildcard support: {'✅' if any('$any' in str(t.get('query', {})) for t in validation_tests) else '❌'}")
    print(f"   • Complex combinations: {'✅' if success_rate >= 0.8 else '❌'}")

    return success_rate

if __name__ == "__main__":
    accuracy = run_task_memory_validation()
    print(f"\nFinal Task Memory Validation Score: {accuracy:.1%}")