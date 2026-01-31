#!/usr/bin/env python3
"""
Challenge 007 Batch - All Solutions via HTTP

Runs all batch 007 challenge solutions using the Holon HTTP API.
This demonstrates that all solutions work with holon as a remote service.

Usage:
    # First, start the server:
    ./scripts/run_with_venv.sh python scripts/server/holon_server.py

    # Then run all challenges via HTTP:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/all-solutions-http.py

    # Or run specific challenges:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/all-solutions-http.py --challenges 1 2 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def check_server(url: str = "http://localhost:8000") -> bool:
    """Check if server is running."""
    try:
        import requests
        response = requests.get(f"{url}/api/v1/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def run_challenge(script_path: Path, url: str) -> tuple:
    """Run a single challenge script."""
    start = time.time()

    cmd = [
        "./scripts/run_with_venv.sh",
        "python",
        str(script_path),
        "--http",
        "--url",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    return result.returncode == 0, elapsed, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Run all batch 007 challenges via HTTP")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument(
        "--challenges",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Which challenges to run (default: all)",
    )
    args = parser.parse_args()

    challenges_to_run = args.challenges or [1, 2, 3, 4, 5, 6, 7]

    print("=" * 70)
    print("BATCH 007 CHALLENGES - HTTP MODE")
    print("=" * 70)

    # Check server
    print(f"\n🔍 Checking server at {args.url}...")
    if not check_server(args.url):
        print(f"❌ Server not running at {args.url}")
        print("\n💡 Start the server with:")
        print("   ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
        sys.exit(1)

    print(f"✅ Server is running\n")

    challenges = {
        1: {
            "name": "Rete Rule Engine",
            "script": "001-rete-solution.py",
            "description": "Holon-powered rule engine with exact and fuzzy matching",
        },
        2: {
            "name": "Multi-Modal Code Understanding",
            "script": "002-code-understanding-solution.py",
            "description": "AST parsing and code metadata search",
        },
        3: {
            "name": "Hierarchical Document Retrieval",
            "script": "003-hierarchical-docs-solution.py",
            "description": "Legal/technical document navigation",
        },
        4: {
            "name": "Event Sequence Matching",
            "script": "004-event-sequence-solution.py",
            "description": "Anomaly detection with temporal patterns",
        },
        5: {
            "name": "Knowledge Graph Fragment Matching",
            "script": "005-knowledge-graph-solution.py",
            "description": "Querying graph structures with relations",
        },
        6: {
            "name": "Medical Record Matching",
            "script": "006-medical-records-solution.py",
            "description": "Fuzzy matching on clinical records",
        },
        7: {
            "name": "Scale & Limit Experiments",
            "script": "007-scale-experiments-solution.py",
            "description": "Testing Holon's limits and constraints",
        },
    }

    results = []
    total_start = time.time()

    for challenge_num in challenges_to_run:
        challenge = challenges[challenge_num]
        script_path = Path(__file__).parent / challenge["script"]

        print("=" * 70)
        print(f"CHALLENGE {challenge_num}: {challenge['name']}")
        print("=" * 70)
        print(f"{challenge['description']}\n")

        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            results.append(
                {
                    "challenge": challenge_num,
                    "name": challenge["name"],
                    "success": False,
                    "elapsed": 0,
                }
            )
            continue

        success, elapsed, stdout, stderr = run_challenge(script_path, args.url)

        # Print output
        if success:
            # Show last 50 lines of output
            lines = stdout.split("\n")
            print("\n".join(lines[-50:]))
        else:
            print(f"❌ Challenge failed!\n")
            print("STDOUT:", stdout[-500:] if len(stdout) > 500 else stdout)
            print("STDERR:", stderr[-500:] if len(stderr) > 500 else stderr)

        results.append(
            {
                "challenge": challenge_num,
                "name": challenge["name"],
                "success": success,
                "elapsed": elapsed,
            }
        )

        print(f"\n{'✅' if success else '❌'} Challenge {challenge_num} {'succeeded' if success else 'failed'} in {elapsed:.2f}s\n")

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\n📊 Results: {successful}/{total} challenges successful")
    print(f"⏱️  Total time: {total_elapsed:.2f}s\n")

    for result in results:
        status = "✅" if result["success"] else "❌"
        print(
            f"   {status} Challenge {result['challenge']}: {result['name']} ({result['elapsed']:.2f}s)"
        )

    if successful == total:
        print(
            """
    🎉 ALL CHALLENGES COMPLETED SUCCESSFULLY!

    All batch 007 challenges work with Holon as a remote service via HTTP.
    This demonstrates the unified client API working seamlessly in both
    local and remote modes.
        """
        )
    else:
        print(
            f"""
    ⚠️  {total - successful} challenge(s) failed. Check output above for details.
        """
        )

    sys.exit(0 if successful == total else 1)


if __name__ == "__main__":
    main()
