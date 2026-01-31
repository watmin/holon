#!/usr/bin/env python3
"""
Run remaining stress tests (skip scale since we have results).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from comprehensive_stress import (
    StressTestSuite,
    test_prototype_saturation,
    test_ngram_dilution,
    test_dimensionality,
    test_query_complexity,
    test_noise_tolerance,
    test_similarity_collapse,
)

def main():
    print("="*80)
    print("HOLON STRESS TESTS - REMAINING")
    print("="*80)

    suite = StressTestSuite()

    try:
        test_similarity_collapse(suite)
        test_prototype_saturation(suite)
        test_ngram_dilution(suite)
        test_dimensionality(suite)
        test_query_complexity(suite)
        test_noise_tolerance(suite)
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    suite.print_summary()

if __name__ == "__main__":
    main()
