"""
Simple test for MOVNS implementation
Quick verification of core functionality
"""

import sys
import os
import time

os.chdir('E:/pycommend/pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from movns_vns import MOVNS_VNS


def quick_test():
    """Quick functional test of MOVNS"""
    print("\nMOVNS Quick Test")
    print("="*60)

    try:
        print("\n1. Testing initialization...")
        movns = MOVNS_VNS('numpy', archive_size=10, max_iterations=2, track_metrics=False)
        print("   [OK] Initialized")

        print("\n2. Testing smart initialization...")
        sol = movns.smart_initialization('small')
        print(f"   [OK] Created solution with {sum(sol)} packages")

        print("\n3. Testing objective evaluation...")
        obj = movns.evaluate_objectives(sol)
        print(f"   [OK] Objectives: LU={-obj[0]:.1f}, SS={-obj[1]:.3f}, RSS={obj[2]:.0f}")

        print("\n4. Testing neighborhoods...")
        neighborhoods = movns.define_neighborhoods()
        print(f"   [OK] Created {len(neighborhoods)} neighborhoods")

        print("\n5. Testing archive initialization...")
        movns.initialize_archive()
        print(f"   [OK] Archive has {len(movns.archive)} solutions")

        print("\n6. Testing MOBI/P search...")
        improved = movns.mobi_p_local_search(sol, neighborhoods[0])
        print(f"   [OK] Found {len(improved)} non-dominated solutions")

        print("\n7. Running mini MOVNS (2 iterations)...")
        start = time.time()
        solutions = movns.run()
        elapsed = time.time() - start
        print(f"   [OK] Completed in {elapsed:.2f}s, found {len(solutions)} solutions")

        if solutions:
            best = solutions[0]
            print(f"\nBest solution for numpy:")
            print(f"  Packages: {', '.join(best['packages'])}")
            print(f"  LU: {best['objectives']['linked_usage']:.1f}")
            print(f"  SS: {best['objectives']['semantic_similarity']:.3f}")
            print(f"  Size: {best['objectives']['set_size']:.0f}")

        print("\n[OK] ALL TESTS PASSED!")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = quick_test()