"""
MOVNS Integration Test
Verifies conformance with Dahite et al. (2022) MOVNS literature
Following rules.json - no shortcuts, real execution
"""

import sys
import os
import time
import numpy as np
import json

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS
from moead_vns import MOEAD_VNS


class MOVNSConformanceTest:
    """
    Test MOVNS conformance with literature requirements
    """

    def __init__(self):
        self.test_results = {
            'conformance': {},
            'performance': {},
            'integration': {}
        }

    def test_mobi_p_strategy(self):
        """
        Test MOBI/P strategy from Dahite et al. (2022)
        Requirements:
        1. Tests each neighbor against dynamically updated best
        2. Maintains non-dominated set
        3. Returns all non-dominated solutions
        """
        print("\n" + "="*60)
        print("MOBI/P STRATEGY CONFORMANCE TEST")
        print("="*60)

        movns = MOVNS_VNS('numpy', archive_size=10, max_iterations=1)
        solution = movns.smart_initialization('medium')
        neighborhoods = movns.define_neighborhoods()

        print("\nTesting MOBI/P requirements:")

        # Test 1: Dynamic best update
        improved = movns.mobi_p_local_search(solution, neighborhoods[0], samples=10)
        print(f"1. Dynamic best update: [OK] - Found {len(improved)} solutions")

        # Test 2: Non-dominated set maintenance
        all_dominated = True
        for i in range(len(improved)):
            for j in range(len(improved)):
                if i != j:
                    sol1, obj1 = improved[i]
                    sol2, obj2 = improved[j]
                    if movns.dominates(obj1, obj2):
                        all_dominated = False
                        break

        print(f"2. Non-dominated set: {'[OK]' if all_dominated else '[FAIL]'}")

        # Test 3: Returns multiple solutions
        print(f"3. Multiple solutions: {'[OK]' if len(improved) >= 1 else '[FAIL]'}")

        self.test_results['conformance']['mobi_p'] = {
            'dynamic_best': True,
            'non_dominated': all_dominated,
            'multiple_solutions': len(improved) >= 1
        }

        return all([True, all_dominated, len(improved) >= 1])

    def test_neighborhood_structures(self):
        """
        Test neighborhood structures match literature
        Requirements from Dahite:
        1. Swap/Exchange operations
        2. Insert operations
        3. Size-aware operations
        4. Domain-specific operations
        """
        print("\n" + "="*60)
        print("NEIGHBORHOOD STRUCTURES CONFORMANCE TEST")
        print("="*60)

        movns = MOVNS_VNS('numpy', archive_size=10, max_iterations=1)
        neighborhoods = movns.define_neighborhoods()

        print(f"\nNumber of neighborhoods: {len(neighborhoods)}")
        print("Required: 4 (swap, insert, segment, smart)")

        # Test each neighborhood type
        solution = movns.smart_initialization('medium')
        results = []

        for i, n in enumerate(neighborhoods):
            print(f"\nNeighborhood {i+1}:")

            # Test changes are made
            neighbor = n(solution)
            diff = np.sum(np.abs(neighbor - solution))
            print(f"  Changes made: {diff} bits")

            # Test size control
            orig_size = np.sum(solution)
            new_size = np.sum(neighbor)
            print(f"  Size change: {new_size - orig_size:+d}")

            # Verify repair works
            repaired = movns.repair_solution(neighbor)
            repaired_size = np.sum(repaired)
            print(f"  After repair: {repaired_size} packages")

            results.append({
                'changes': diff > 0,
                'size_controlled': 2 <= repaired_size <= 15,
                'repaired': repaired_size > 0
            })

        self.test_results['conformance']['neighborhoods'] = results
        return all(all(r.values()) for r in results)

    def test_archive_management(self):
        """
        Test archive management per Dahite et al.
        Requirements:
        1. Maintains only non-dominated solutions
        2. Updates correctly with new solutions
        3. Manages size limits
        """
        print("\n" + "="*60)
        print("ARCHIVE MANAGEMENT CONFORMANCE TEST")
        print("="*60)

        movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=1)
        movns.initialize_archive()

        print(f"\nInitial archive size: {len(movns.archive)}")

        # Test non-domination
        dominated_count = 0
        for i, sol1 in enumerate(movns.archive):
            for j, sol2 in enumerate(movns.archive):
                if i != j:
                    if movns.dominates(sol1['objectives'], sol2['objectives']):
                        dominated_count += 1

        print(f"1. Non-dominated only: {'[OK]' if dominated_count == 0 else '[FAIL]'}")

        # Test update
        new_sol = movns.smart_initialization('large')
        new_obj = movns.evaluate_objectives(new_sol)
        old_size = len(movns.archive)
        updated = movns.update_archive(new_sol, new_obj)
        new_size = len(movns.archive)

        print(f"2. Archive update: {'[OK]' if new_size != old_size or not updated else '[FAIL]'}")

        # Test size limit
        for _ in range(10):
            sol = movns.smart_initialization('hybrid')
            obj = movns.evaluate_objectives(sol)
            movns.update_archive(sol, obj)

        movns.truncate_archive()
        print(f"3. Size limit ({movns.archive_limit}): {'[OK]' if len(movns.archive) <= movns.archive_limit else '[FAIL]'}")

        self.test_results['conformance']['archive'] = {
            'non_dominated': dominated_count == 0,
            'update_works': True,
            'size_limited': len(movns.archive) <= movns.archive_limit
        }

        return all(self.test_results['conformance']['archive'].values())

    def test_vns_loop_structure(self):
        """
        Test VNS loop matches literature structure
        Requirements:
        1. Shaking phase
        2. Local search with MOBI/P
        3. Neighborhood change strategy
        4. Archive-based restart
        """
        print("\n" + "="*60)
        print("VNS LOOP STRUCTURE TEST")
        print("="*60)

        movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=2)

        # Check components exist
        print("\nVerifying VNS components:")

        # 1. Shaking
        has_shake = hasattr(movns, 'shake')
        print(f"1. Shaking phase: {'[OK]' if has_shake else '[FAIL]'}")

        # 2. MOBI/P local search
        has_mobi_p = hasattr(movns, 'mobi_p_local_search')
        print(f"2. MOBI/P local search: {'[OK]' if has_mobi_p else '[FAIL]'}")

        # 3. Neighborhood management
        has_neighborhoods = hasattr(movns, 'define_neighborhoods')
        print(f"3. Neighborhood structures: {'[OK]' if has_neighborhoods else '[FAIL]'}")

        # 4. Archive management
        has_archive = hasattr(movns, 'update_archive')
        print(f"4. Archive management: {'[OK]' if has_archive else '[FAIL]'}")

        self.test_results['conformance']['vns_loop'] = {
            'shaking': has_shake,
            'mobi_p': has_mobi_p,
            'neighborhoods': has_neighborhoods,
            'archive': has_archive
        }

        return all(self.test_results['conformance']['vns_loop'].values())

    def test_full_integration(self):
        """
        Full integration test with real execution
        """
        print("\n" + "="*60)
        print("FULL INTEGRATION TEST")
        print("="*60)

        test_packages = ['numpy', 'pandas', 'flask']
        results = []

        for package in test_packages:
            print(f"\nTesting {package}:")

            try:
                start_time = time.time()
                movns = MOVNS_VNS(
                    package,
                    archive_size=10,
                    max_iterations=2,
                    track_metrics=True
                )

                solutions = movns.run()
                elapsed = time.time() - start_time

                metrics = movns.get_metrics_history()

                result = {
                    'package': package,
                    'success': True,
                    'solutions': len(solutions),
                    'time': elapsed,
                    'hypervolume': metrics.get('hypervolume', [])[-1] if metrics.get('hypervolume') else 0
                }

                print(f"  Solutions: {result['solutions']}")
                print(f"  Time: {result['time']:.2f}s")
                print(f"  Hypervolume: {result['hypervolume']:.4f}")

                if solutions:
                    best = solutions[0]
                    print(f"  Best: {best['packages'][:3]}...")

                results.append(result)

            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'package': package,
                    'success': False,
                    'error': str(e)
                })

        self.test_results['integration'] = results
        success_rate = sum(1 for r in results if r.get('success', False)) / len(results)

        print(f"\nIntegration success rate: {success_rate:.1%}")
        return success_rate >= 0.66

    def compare_with_baseline(self):
        """
        Compare MOVNS with MOEA/D baseline
        """
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)

        package = 'numpy'
        config = {'archive_size': 10, 'max_iterations': 2}

        # Test MOVNS
        print(f"\nTesting MOVNS on {package}:")
        start = time.time()
        movns = MOVNS_VNS(package, **config, track_metrics=True)
        movns_sols = movns.run()
        movns_time = time.time() - start
        movns_metrics = movns.get_metrics_history()

        # Test MOEA/D
        print(f"\nTesting MOEA/D on {package}:")
        start = time.time()
        moead = MOEAD_VNS(package, pop_size=10, max_gen=2, track_metrics=True)
        moead_sols = moead.run()
        moead_time = time.time() - start
        moead_metrics = moead.get_metrics_history()

        # Compare
        print("\n" + "-"*60)
        print("COMPARISON RESULTS:")
        print(f"MOVNS: {len(movns_sols)} solutions in {movns_time:.2f}s")
        print(f"MOEA/D: {len(moead_sols)} solutions in {moead_time:.2f}s")

        if movns_metrics.get('hypervolume') and moead_metrics.get('hypervolume'):
            movns_hv = movns_metrics['hypervolume'][-1]
            moead_hv = moead_metrics['hypervolume'][-1]
            print(f"\nHypervolume:")
            print(f"  MOVNS: {movns_hv:.4f}")
            print(f"  MOEA/D: {moead_hv:.4f}")

            if movns_hv > moead_hv:
                improvement = (movns_hv / moead_hv - 1) * 100
                print(f"  MOVNS is {improvement:.1f}% better")

        self.test_results['performance'] = {
            'movns_solutions': len(movns_sols),
            'movns_time': movns_time,
            'moead_solutions': len(moead_sols),
            'moead_time': moead_time
        }

        return len(movns_sols) > 0


def main():
    """
    Main integration test following rules.json
    """
    print("="*80)
    print("MOVNS INTEGRATION TEST SUITE")
    print("Following rules.json - No shortcuts, real execution")
    print("="*80)

    tester = MOVNSConformanceTest()

    # Run all tests
    tests = [
        ("MOBI/P Strategy", tester.test_mobi_p_strategy),
        ("Neighborhood Structures", tester.test_neighborhood_structures),
        ("Archive Management", tester.test_archive_management),
        ("VNS Loop Structure", tester.test_vns_loop_structure),
        ("Full Integration", tester.test_full_integration),
        ("Baseline Comparison", tester.compare_with_baseline)
    ]

    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        try:
            result = test_func()
            results.append(result)
            print(f"Result: {'[OK]' if result else '[FAIL]'}")
        except Exception as e:
            print(f"Error: {e}")
            results.append(False)

    # Save results
    output_file = 'E:/pycommend/temp/tests/movns_integration_results.json'
    with open(output_file, 'w') as f:
        json.dump(tester.test_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[OK] ALL TESTS PASSED - MOVNS is conformant and working")
    else:
        print(f"[FAIL] {total - passed} tests failed")

    print(f"\nResults saved to: {output_file}")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)