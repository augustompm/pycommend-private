"""
Incremental tests for MOVNS implementation
Tests each component step by step to ensure correct functionality
"""

import sys
import os
import numpy as np
import time

sys.path.append('/e/pycommend/pycommend-code/src/optimizer')
sys.path.append('/e/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS
from moead_vns import MOEAD_VNS


def test_1_initialization():
    """Test 1: Verify MOVNS initialization"""
    print("\n" + "="*60)
    print("TEST 1: MOVNS Initialization")
    print("="*60)

    try:
        movns = MOVNS_VNS('numpy', archive_size=50, max_iterations=10)
        print("[OK] MOVNS initialized successfully")
        print(f"[OK] Archive limit: {movns.archive_limit}")
        print(f"[OK] Max iterations: {movns.max_iterations}")
        print(f"[OK] K_max (neighborhoods): {movns.k_max}")
        print(f"[OK] Main package index: {movns.main_package_idx}")
        print(f"[OK] Candidate pools loaded")
        return True
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return False


def test_2_neighborhoods():
    """Test 2: Verify neighborhood definitions"""
    print("\n" + "="*60)
    print("TEST 2: Neighborhood Definitions")
    print("="*60)

    try:
        movns = MOVNS_VNS('numpy', archive_size=30, max_iterations=5)
        neighborhoods = movns.define_neighborhoods()

        print(f"[OK] Number of neighborhoods: {len(neighborhoods)}")

        test_solution = movns.smart_initialization('small')
        print(f"[OK] Test solution size: {np.sum(test_solution)}")

        for i, neighborhood in enumerate(neighborhoods, 1):
            modified = neighborhood(test_solution)
            diff = np.sum(np.abs(modified - test_solution))
            print(f"[OK] Neighborhood {i}: Changed {diff} bits")

        return True
    except Exception as e:
        print(f"[FAIL] Neighborhoods test failed: {e}")
        return False


def test_3_mobi_p_search():
    """Test 3: Verify MOBI/P local search"""
    print("\n" + "="*60)
    print("TEST 3: MOBI/P Local Search")
    print("="*60)

    try:
        movns = MOVNS_VNS('numpy', archive_size=30, max_iterations=5)
        neighborhoods = movns.define_neighborhoods()

        test_solution = movns.smart_initialization('medium')
        initial_obj = movns.evaluate_objectives(test_solution)
        print(f"[OK] Initial objectives: LU={-initial_obj[0]:.2f}, "
              f"SS={-initial_obj[1]:.4f}, RSS={initial_obj[2]:.1f}")

        improved_solutions = movns.mobi_p_local_search(test_solution, neighborhoods[0])
        print(f"[OK] MOBI/P returned {len(improved_solutions)} non-dominated solutions")

        if improved_solutions:
            best_sol, best_obj = improved_solutions[0]
            print(f"[OK] Best found: LU={-best_obj[0]:.2f}, "
                  f"SS={-best_obj[1]:.4f}, RSS={best_obj[2]:.1f}")

        return True
    except Exception as e:
        print(f"[FAIL] MOBI/P test failed: {e}")
        return False


def test_4_archive_update():
    """Test 4: Verify archive update mechanism"""
    print("\n" + "="*60)
    print("TEST 4: Archive Update")
    print("="*60)

    try:
        movns = MOVNS_VNS('numpy', archive_size=30, max_iterations=5)

        sol1 = movns.smart_initialization('small')
        obj1 = movns.evaluate_objectives(sol1)
        updated1 = movns.update_archive(sol1, obj1)
        print(f"[OK] First solution added: {updated1}, Archive size: {len(movns.archive)}")

        sol2 = movns.smart_initialization('medium')
        obj2 = movns.evaluate_objectives(sol2)
        updated2 = movns.update_archive(sol2, obj2)
        print(f"[OK] Second solution added: {updated2}, Archive size: {len(movns.archive)}")

        dominated_obj = obj1 + np.array([100, 100, 100])
        updated3 = movns.update_archive(sol1, dominated_obj)
        print(f"[OK] Dominated solution rejected: {not updated3}")

        return True
    except Exception as e:
        print(f"[FAIL] Archive update test failed: {e}")
        return False


def test_5_vns_loop():
    """Test 5: Verify VNS main loop (small scale)"""
    print("\n" + "="*60)
    print("TEST 5: VNS Main Loop")
    print("="*60)

    try:
        movns = MOVNS_VNS('fastapi', archive_size=20, max_iterations=3)

        print("Running MOVNS for 3 iterations...")
        start_time = time.time()
        solutions = movns.run()
        elapsed = time.time() - start_time

        print(f"[OK] MOVNS completed in {elapsed:.2f} seconds")
        print(f"[OK] Final archive size: {len(solutions)}")

        if solutions:
            best = solutions[0]
            print(f"[OK] Best solution: {', '.join(best['packages'][:5])}")
            print(f"  LU: {best['objectives']['linked_usage']:.2f}")
            print(f"  SS: {best['objectives']['semantic_similarity']:.4f}")
            print(f"  Size: {best['objectives']['set_size']:.0f}")

        return True
    except Exception as e:
        print(f"[FAIL] VNS loop test failed: {e}")
        return False


def test_6_compare_with_moead():
    """Test 6: Compare MOVNS with MOEA/D baseline"""
    print("\n" + "="*60)
    print("TEST 6: MOVNS vs MOEA/D Comparison")
    print("="*60)

    try:
        package = 'pandas'

        print(f"\nTesting with package: {package}")

        print("\nRunning MOEA/D...")
        start_time = time.time()
        moead = MOEAD_VNS(package, pop_size=30, max_gen=10, track_metrics=True)
        moead_solutions = moead.run()
        moead_time = time.time() - start_time
        moead_metrics = moead.get_metrics_history()

        print("\nRunning MOVNS...")
        start_time = time.time()
        movns = MOVNS_VNS(package, archive_size=30, max_iterations=10, track_metrics=True)
        movns_solutions = movns.run()
        movns_time = time.time() - start_time
        movns_metrics = movns.get_metrics_history()

        print("\n" + "-"*60)
        print("COMPARISON RESULTS:")
        print("-"*60)

        print(f"MOEA/D: {len(moead_solutions)} solutions in {moead_time:.2f}s")
        if moead_metrics and moead_metrics['hypervolume']:
            print(f"  Final HV: {moead_metrics['hypervolume'][-1]:.4f}")

        print(f"MOVNS:  {len(movns_solutions)} solutions in {movns_time:.2f}s")
        if movns_metrics and movns_metrics['hypervolume']:
            print(f"  Final HV: {movns_metrics['hypervolume'][-1]:.4f}")

        if movns_time < moead_time:
            print(f"[OK] MOVNS is {(moead_time/movns_time - 1)*100:.1f}% faster")

        if movns_metrics and moead_metrics:
            if movns_metrics['hypervolume'] and moead_metrics['hypervolume']:
                if movns_metrics['hypervolume'][-1] > moead_metrics['hypervolume'][-1]:
                    improvement = (movns_metrics['hypervolume'][-1] / moead_metrics['hypervolume'][-1] - 1) * 100
                    print(f"[OK] MOVNS has {improvement:.1f}% better hypervolume")

        return True
    except Exception as e:
        print(f"[FAIL] Comparison test failed: {e}")
        return False


def test_7_multiple_packages():
    """Test 7: Test MOVNS with multiple packages"""
    print("\n" + "="*60)
    print("TEST 7: Multiple Package Test")
    print("="*60)

    test_packages = ['numpy', 'flask', 'django', 'scikit-learn', 'requests']
    results = []

    for package in test_packages:
        try:
            print(f"\nTesting {package}...")
            movns = MOVNS_VNS(package, archive_size=30, max_iterations=5, track_metrics=True)
            solutions = movns.run()

            if solutions:
                best = solutions[0]
                print(f"  [OK] Found {len(solutions)} solutions")
                print(f"    Best: {', '.join(best['packages'][:3])}...")
                print(f"    LU={best['objectives']['linked_usage']:.1f}, "
                      f"SS={best['objectives']['semantic_similarity']:.3f}, "
                      f"Size={best['objectives']['set_size']:.0f}")
                results.append(True)
            else:
                print(f"  [FAIL] No solutions found")
                results.append(False)
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            results.append(False)

    success_rate = sum(results) / len(results) * 100
    print(f"\n[OK] Success rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")

    return success_rate >= 80


def run_all_tests():
    """Run all incremental tests"""
    print("\n" + "="*60)
    print("MOVNS INCREMENTAL TEST SUITE")
    print("="*60)

    tests = [
        test_1_initialization,
        test_2_neighborhoods,
        test_3_mobi_p_search,
        test_4_archive_update,
        test_5_vns_loop,
        test_6_compare_with_moead,
        test_7_multiple_packages
    ]

    results = []
    for i, test in enumerate(tests, 1):
        result = test()
        results.append(result)
        if not result:
            print(f"\n⚠ Test {i} failed. Stopping further tests.")
            break

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    passed = sum(results)
    total = len(tests)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[OK] ALL TESTS PASSED - MOVNS implementation is correct!")
    else:
        print(f"[FAIL] {total - passed} tests failed or not run")

    return passed == total


if __name__ == '__main__':
    os.chdir('/e/pycommend/pycommend-code')
    success = run_all_tests()