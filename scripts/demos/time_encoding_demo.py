#!/usr/bin/env python3
"""
Time Encoding Demo - Comprehensive exploration of $time marker.

Demonstrates:
1. Basic time encoding with $time marker
2. Circular similarity (same hour different days, month wraparound)
3. Positional similarity (recent vs old)
4. Combined structure + time queries
5. Different resolution levels

Usage:
    ./scripts/run_with_venv.sh python scripts/demos/time_encoding_demo.py
"""

from datetime import datetime, timedelta
import time

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity


def timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def demo_1_basic_time_encoding():
    """Demo 1: Basic time encoding."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Time Encoding")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert documents with $time markers
    docs = [
        {
            "order_id": "001",
            "customer": {"tier": "platinum"},
            "created_at": {"$time": timestamp(datetime(2024, 12, 15, 14, 30))},
        },
        {
            "order_id": "002",
            "customer": {"tier": "gold"},
            "created_at": {"$time": timestamp(datetime(2024, 12, 15, 10, 0))},
        },
        {
            "order_id": "003",
            "customer": {"tier": "platinum"},
            "created_at": {"$time": timestamp(datetime(2024, 6, 15, 14, 30))},
        },
    ]

    print("\n📥 Inserting 3 orders with timestamps...")
    for doc in docs:
        client.insert_json(doc)

    # Query with similar time
    print("\n🔍 Query: Platinum orders around 2pm in December")
    probe = {
        "customer": {"tier": "platinum"},
        "created_at": {"$time": timestamp(datetime(2024, 12, 16, 14, 0))},
    }

    results = client.search_json(probe=probe, limit=5, threshold=0.0)

    print(f"   Found {len(results)} results:")
    for r in results:
        data = r["data"]
        print(f"   - Order {data['order_id']}: score={r['score']:.3f}")


def demo_2_circular_similarity():
    """Demo 2: Circular similarity - same time of day, different days."""
    print("\n" + "=" * 70)
    print("DEMO 2: Circular Similarity (Same Hour, Different Days)")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Create events at the same hour on different days
    base_date = datetime(2024, 12, 1, 14, 30)  # 2:30 PM

    docs = []
    for i in range(7):  # One week of events at 2:30 PM
        dt = base_date + timedelta(days=i)
        docs.append({
            "event_id": f"event_{i}",
            "event_type": "meeting",
            "scheduled_at": {"$time": timestamp(dt)},
            "day_name": dt.strftime("%A"),
        })

    # Add some events at different times
    docs.append({
        "event_id": "morning_event",
        "event_type": "standup",
        "scheduled_at": {"$time": timestamp(datetime(2024, 12, 5, 9, 0))},
        "day_name": "Thursday",
    })
    docs.append({
        "event_id": "evening_event",
        "event_type": "review",
        "scheduled_at": {"$time": timestamp(datetime(2024, 12, 5, 18, 0))},
        "day_name": "Thursday",
    })

    print(f"\n📥 Inserting {len(docs)} events...")
    for doc in docs:
        client.insert_json(doc)

    # Query for events around 2:30 PM
    print("\n🔍 Query: Events around 2:30 PM")
    probe = {
        "event_type": "meeting",
        "scheduled_at": {"$time": timestamp(datetime(2024, 12, 10, 14, 30))},
    }

    results = client.search_json(probe=probe, limit=5, threshold=0.0)

    print(f"   Found {len(results)} results:")
    for r in results:
        data = r["data"]
        print(f"   - {data['event_id']} ({data['day_name']}): score={r['score']:.3f}")

    print("\n   ✅ Events at 2:30 PM should score higher than morning/evening events")


def demo_3_month_wraparound():
    """Demo 3: Month wraparound - December similar to January."""
    print("\n" + "=" * 70)
    print("DEMO 3: Month Wraparound (December ~ January)")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Create monthly reports
    docs = []
    for month in range(1, 13):
        docs.append({
            "report_id": f"report_{month:02d}",
            "report_type": "monthly_sales",
            "period": {"$time": timestamp(datetime(2024, month, 15, 12, 0))},
            "month_name": datetime(2024, month, 1).strftime("%B"),
        })

    print(f"\n📥 Inserting {len(docs)} monthly reports...")
    for doc in docs:
        client.insert_json(doc)

    # Query for late December - should be similar to January
    print("\n🔍 Query: Reports similar to late December timing")
    probe = {
        "report_type": "monthly_sales",
        "period": {"$time": timestamp(datetime(2024, 12, 28, 12, 0))},
    }

    results = client.search_json(probe=probe, limit=5, threshold=0.0)

    print(f"   Top 5 results:")
    for r in results:
        data = r["data"]
        print(f"   - {data['month_name']}: score={r['score']:.3f}")

    print("\n   ✅ December and January should both score high (circular wrap)")


def demo_4_positional_recency():
    """Demo 4: Positional similarity - recent events score higher."""
    print("\n" + "=" * 70)
    print("DEMO 4: Positional Similarity (Recency)")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Create events at various times in the past
    now = datetime(2024, 12, 15, 12, 0)

    time_offsets = [
        ("1 hour ago", timedelta(hours=1)),
        ("1 day ago", timedelta(days=1)),
        ("1 week ago", timedelta(weeks=1)),
        ("1 month ago", timedelta(days=30)),
        ("6 months ago", timedelta(days=180)),
        ("1 year ago", timedelta(days=365)),
    ]

    docs = []
    for label, offset in time_offsets:
        dt = now - offset
        docs.append({
            "log_id": label.replace(" ", "_"),
            "log_type": "system",
            "logged_at": {"$time": timestamp(dt)},
            "label": label,
        })

    print(f"\n📥 Inserting {len(docs)} log entries at various times...")
    for doc in docs:
        client.insert_json(doc)

    # Query for "now"
    print("\n🔍 Query: Logs similar to 'now'")
    probe = {
        "log_type": "system",
        "logged_at": {"$time": timestamp(now)},
    }

    results = client.search_json(probe=probe, limit=10, threshold=0.0)

    print(f"   Results ordered by similarity:")
    for r in results:
        data = r["data"]
        print(f"   - {data['label']}: score={r['score']:.3f}")

    print("\n   ✅ Recent events should score higher than older ones")


def demo_5_structure_plus_time():
    """Demo 5: Combined structure + time queries."""
    print("\n" + "=" * 70)
    print("DEMO 5: Structure + Time Combined Queries")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Create a realistic dataset: orders with various attributes and times
    orders = [
        # Recent platinum orders
        {"id": "001", "customer": {"tier": "platinum", "region": "US"},
         "total": 15000, "items": ["laptop", "monitor"],
         "created_at": {"$time": timestamp(datetime(2024, 12, 14, 10, 0))}},
        {"id": "002", "customer": {"tier": "platinum", "region": "EU"},
         "total": 12000, "items": ["laptop"],
         "created_at": {"$time": timestamp(datetime(2024, 12, 13, 14, 0))}},

        # Recent gold orders
        {"id": "003", "customer": {"tier": "gold", "region": "US"},
         "total": 5000, "items": ["keyboard", "mouse"],
         "created_at": {"$time": timestamp(datetime(2024, 12, 14, 11, 0))}},

        # Old platinum orders (6 months ago)
        {"id": "004", "customer": {"tier": "platinum", "region": "US"},
         "total": 18000, "items": ["laptop", "monitor", "dock"],
         "created_at": {"$time": timestamp(datetime(2024, 6, 14, 10, 0))}},

        # Old gold orders
        {"id": "005", "customer": {"tier": "gold", "region": "US"},
         "total": 3000, "items": ["mouse"],
         "created_at": {"$time": timestamp(datetime(2024, 6, 10, 15, 0))}},
    ]

    print(f"\n📥 Inserting {len(orders)} orders...")
    for order in orders:
        client.insert_json(order)

    # Query: Recent platinum US orders
    print("\n🔍 Query: Platinum US orders from mid-December")
    probe = {
        "customer": {"tier": "platinum", "region": "US"},
        "total": 15000,
        "created_at": {"$time": timestamp(datetime(2024, 12, 15, 10, 0))},
    }

    results = client.search_json(probe=probe, limit=5, threshold=0.0)

    print(f"   Results:")
    for r in results:
        data = r["data"]
        dt_str = datetime.fromtimestamp(data["created_at"]["$time"]).strftime("%Y-%m-%d")
        print(f"   - Order {data['id']}: {data['customer']['tier']}/{data['customer']['region']} "
              f"${data['total']} on {dt_str} (score={r['score']:.3f})")

    print("\n   ✅ Recent platinum US orders should rank highest")
    print("   ✅ Old platinum US orders should rank lower (time penalty)")
    print("   ✅ Gold orders should rank lowest (structure mismatch)")


def demo_6_resolution_levels():
    """Demo 6: Different time resolution levels."""
    print("\n" + "=" * 70)
    print("DEMO 6: Time Resolution Levels")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    base_time = datetime(2024, 12, 15, 14, 30, 45)

    # Create events with different resolutions
    resolutions = ["second", "minute", "hour", "day"]

    for res in resolutions:
        print(f"\n📊 Testing {res.upper()} resolution:")

        # Insert events seconds/minutes/hours/days apart
        events = []
        if res == "second":
            offsets = [0, 1, 5, 30, 60]  # seconds
        elif res == "minute":
            offsets = [0, 1, 5, 30, 60]  # minutes
        elif res == "hour":
            offsets = [0, 1, 3, 12, 24]  # hours
        else:  # day
            offsets = [0, 1, 7, 30, 90]  # days

        for i, offset in enumerate(offsets):
            if res == "second":
                dt = base_time + timedelta(seconds=offset)
            elif res == "minute":
                dt = base_time + timedelta(minutes=offset)
            elif res == "hour":
                dt = base_time + timedelta(hours=offset)
            else:
                dt = base_time + timedelta(days=offset)

            event = {
                "event_id": f"{res}_{i}",
                "offset": offset,
                "occurred_at": {"$time": timestamp(dt), "$time_resolution": res},
            }
            events.append(event)
            client.insert_json(event)

        # Query for base time
        probe = {
            "occurred_at": {"$time": timestamp(base_time), "$time_resolution": res},
        }

        results = client.search_json(probe=probe, limit=5, threshold=0.0)

        print(f"   Similarity decay from base:")
        for r in results:
            data = r["data"]
            print(f"   - +{data['offset']} {res}s: score={r['score']:.3f}")


def demo_7_raw_vector_similarity():
    """Demo 7: Direct vector similarity comparisons."""
    print("\n" + "=" * 70)
    print("DEMO 7: Raw Time Vector Similarity")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    import numpy as np

    # Compare time vectors directly
    times = [
        ("Now", datetime(2024, 12, 15, 14, 30)),
        ("Same hour tomorrow", datetime(2024, 12, 16, 14, 30)),
        ("1 hour later today", datetime(2024, 12, 15, 15, 30)),
        ("6 hours later today", datetime(2024, 12, 15, 20, 30)),
        ("Same time last month", datetime(2024, 11, 15, 14, 30)),
        ("Same time 6 months ago", datetime(2024, 6, 15, 14, 30)),
        ("Opposite time (2:30 AM)", datetime(2024, 12, 15, 2, 30)),
    ]

    # Encode base time
    base = {"$time": timestamp(times[0][1])}
    base_vec = client.encode_vectors(base)
    if isinstance(base_vec, list):
        base_vec = np.array(base_vec)

    print(f"\n📊 Similarity to 'Now' ({times[0][1].strftime('%Y-%m-%d %H:%M')}):\n")

    for label, dt in times[1:]:
        vec = client.encode_vectors({"$time": timestamp(dt)})
        if isinstance(vec, list):
            vec = np.array(vec)
        sim = normalized_dot_similarity(base_vec, vec)
        print(f"   {label:30s}: {sim:.4f}")

    print("\n   ✅ Same hour different day = high (circular match)")
    print("   ✅ Recent times = higher than old times (positional)")
    print("   ✅ Opposite time of day = lower (circular mismatch)")


def main():
    print("=" * 70)
    print("TIME ENCODING DEMO")
    print("Exploring $time marker for temporal similarity")
    print("=" * 70)

    start = time.time()

    demo_1_basic_time_encoding()
    demo_2_circular_similarity()
    demo_3_month_wraparound()
    demo_4_positional_recency()
    demo_5_structure_plus_time()
    demo_6_resolution_levels()
    demo_7_raw_vector_similarity()

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    ✅ All demos completed in {elapsed:.2f}s

    Key Findings:
    - $time marker encodes timestamps with circular + positional components
    - Circular: Same hour/day-of-week/month are similar (wraps around)
    - Positional: Recent times are more similar than old times
    - Combined: Structure + time queries work naturally
    - Resolution: Can tune for second/minute/hour/day granularity

    Usage:
        {{"created_at": {{"$time": 1706500000}}}}  # Unix timestamp
        {{"created_at": {{"$time": "2024-01-29T10:30:00Z"}}}}  # ISO string
        {{"created_at": {{"$time": 1706500000, "$time_resolution": "minute"}}}}

    This enables queries like:
        "Find orders similar to this one from around the same time"
    Without manually constructing complex time-range filters!
    """)


if __name__ == "__main__":
    main()
