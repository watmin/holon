#!/usr/bin/env python3
"""
Comprehensive test runner for Holon implementation.
Runs all test suites to verify functionality.
"""

import subprocess
import sys
import os

def run_test_script(script_name, description):
    """Run a test script and report results."""
    print(f"\n🧪 Running {description}...")
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            print(result.stdout.strip())
        else:
            print(f"❌ {description} FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {description} ERROR: {e}")
        return False
    return True

def run_pytest():
    """Run pytest on the test suite."""
    print("\n🧪 Running pytest unit tests...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("✅ Pytest PASSED")
            # Don't print all output, too verbose
        else:
            print("❌ Pytest FAILED")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Pytest TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 Pytest ERROR: {e}")
        return False
    return True

def main():
    print("🚀 Holon Comprehensive Test Suite")
    print("=" * 50)

    tests = [
        ("scripts/test_accuracy.py", "ANN Accuracy Test"),
        ("scripts/test_guards_filtering.py", "Guards Filtering Test"),
        ("scripts/test_negation.py", "Negation Test"),
        ("scripts/test_vector_tricks.py", "Vector Tricks Test"),
    ]

    all_passed = True

    # Run pytest first
    if not run_pytest():
        all_passed = False

    # Run manual test scripts
    for script, desc in tests:
        if os.path.exists(script):
            if not run_test_script(script, desc):
                all_passed = False
        else:
            print(f"⚠️  {desc} script not found: {script}")

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Holon implementation is solid.")
    else:
        print("💥 SOME TESTS FAILED. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()