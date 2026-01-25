#!/usr/bin/env python3

"""
Demo script showing bulk insert performance comparison.
"""

import time


def demo_bulk_insert_performance():
    """Demo function showing bulk insert performance comparison."""
    # Mock timing values for demonstration
    time_ind = 2.5  # Individual insert time
    time_batch = 0.8  # Bulk insert time

    # Compare
    if time_batch > 0:
        speedup = time_ind / time_batch
        print("\n🎯 Results:")
        print(f"Individual inserts: {time_ind:.3f}s")
        print(f"Bulk insert: {time_batch:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1.5:
            print("✅ Bulk insert significantly faster!")
        else:
            print("📈 Bulk insert still more efficient for larger batches")


if __name__ == "__main__":
    demo_bulk_insert_performance()
