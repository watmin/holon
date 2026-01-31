#!/usr/bin/env python3
"""
Quick validation of all batch 007 solutions (local mode).

Runs each challenge briefly to ensure they execute without errors.
This is faster than the full HTTP test suite.

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/validate.py
"""

import subprocess
import sys
import time
from pathlib import Path


def run_solution(script_path: Path, extra_args: list = None) -> tuple:
    """Run a solution script and return success status."""
    cmd = ["./scripts/run_with_venv.sh", "python", str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30
    )
    elapsed = time.time() - start

    success = result.returncode == 0
    return success, elapsed, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("BATCH 007 - LOCAL VALIDATION")
    print("=" * 70)
    print("\nQuick validation that all solutions execute successfully...\n")

    solutions = [
        ("001-rete-solution.py", "Rete Rule Engine", []),
        ("002-code-understanding-solution.py", "Code Understanding", ["--dir", "holon"]),
        ("003-hierarchical-docs-solution.py", "Hierarchical Documents", []),
        ("004-event-sequence-solution.py", "Event Sequences", []),
        ("005-knowledge-graph-solution.py", "Knowledge Graph", []),
        ("006-medical-records-solution.py", "Medical Records", ["--count", "20"]),
        ("007-scale-experiments-solution.py", "Scale Experiments", ["--experiments", "1", "2", "3"]),
    ]

    results = []
    total_start = time.time()

    for i, (script_name, description, extra_args) in enumerate(solutions, 1):
        script_path = Path(__file__).parent / script_name

        print(f"[{i}/7] Testing {description}...", end=" ", flush=True)

        try:
            success, elapsed, stdout, stderr = run_solution(script_path, extra_args)

            if success:
                print(f"✅ ({elapsed:.1f}s)")
            else:
                print(f"❌ ({elapsed:.1f}s)")
                print(f"      Error: {stderr[-200:] if stderr else 'Unknown error'}")

            results.append((description, success, elapsed))

        except subprocess.TimeoutExpired:
            print("❌ (timeout)")
            results.append((description, False, 30.0))
        except Exception as e:
            print(f"❌ (exception: {e})")
            results.append((description, False, 0.0))

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\n✅ {successful}/{total} solutions passed")
    print(f"⏱️  Total time: {total_elapsed:.1f}s\n")

    for description, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"   {status} {description:30s} ({elapsed:.1f}s)")

    if successful == total:
        print("""
    🎉 ALL SOLUTIONS VALIDATED SUCCESSFULLY!

    All batch 007 challenge solutions execute without errors.
    Ready for HTTP testing and production deployment.
        """)
        sys.exit(0)
    else:
        print(f"""
    ⚠️  {total - successful} solution(s) failed validation.
    Review the error messages above for details.
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()
