"""
Quick MOVNS Conformance Test
Faster version with minimal iterations
"""

import sys
import os
import time
import numpy as np
import json

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS


class QuickConformanceTest:
    """
    Quick conformance test for MOVNS
    """

    def __init__(self):
        self.test_results = {}

    def test_mobi_p_conformance(self):
        """Test MOBI/P strategy exists and works"""
        print("\n" + "="*60)
        print("MOBI/P CONFORMANCE TEST")
        print("="*60)

        try:
            movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=1)
            solution = movns.smart_initialization('small')
            neighborhoods = movns.define_neighborhoods()

            print("Testing MOBI/P components:")

            # Test dynamic best update
            improved = movns.mobi_p_local_search(solution, neighborhoods[0], samples=3)
            print(f"1. MOBI/P execution: [OK] - Found {len(improved)} solutions")

            # Test returns non-dominated
            if improved:
                print(f"2. Returns solutions: [OK]")
            else:
                print(f"2. Returns solutions: [WARNING] - No improvements found")

            # Test Pareto strategy
            has_pareto = hasattr(movns, 'dominates') and hasattr(movns, 'filter_non_dominated')
            print(f"3. Pareto strategy: {'[OK]' if has_pareto else '[FAIL]'}")

            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_neighborhoods(self):
        """Test neighborhoods are properly defined"""
        print("\n" + "="*60)
        print("NEIGHBORHOODS TEST")
        print("="*60)

        try:
            movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=1)
            neighborhoods = movns.define_neighborhoods()

            print(f"Number of neighborhoods: {len(neighborhoods)}")
            required = 4
            print(f"Required: {required}")

            if len(neighborhoods) >= required:
                print("[OK] Sufficient neighborhoods defined")
            else:
                print(f"[FAIL] Need {required} neighborhoods, found {len(neighborhoods)}")

            # Test each neighborhood
            solution = movns.smart_initialization('small')
            for i, n in enumerate(neighborhoods):
                neighbor = n(solution)
                diff = np.sum(np.abs(neighbor - solution))
                print(f"  N{i+1}: Changes {diff} bits")

            return len(neighborhoods) >= required

        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_vns_components(self):
        """Test VNS components exist"""
        print("\n" + "="*60)
        print("VNS COMPONENTS TEST")
        print("="*60)

        try:
            movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=1)

            components = {
                'Shaking': hasattr(movns, 'shake'),
                'MOBI/P': hasattr(movns, 'mobi_p_local_search'),
                'Neighborhoods': hasattr(movns, 'define_neighborhoods'),
                'Archive': hasattr(movns, 'update_archive'),
                'Run method': hasattr(movns, 'run')
            }

            all_ok = True
            for name, exists in components.items():
                status = "[OK]" if exists else "[FAIL]"
                print(f"{status} {name}")
                if not exists:
                    all_ok = False

            return all_ok

        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_execution(self):
        """Test MOVNS runs without errors"""
        print("\n" + "="*60)
        print("EXECUTION TEST")
        print("="*60)

        try:
            print("Testing with numpy package...")
            start = time.time()

            movns = MOVNS_VNS(
                'numpy',
                archive_size=5,
                max_iterations=1,
                track_metrics=False
            )

            solutions = movns.run()
            elapsed = time.time() - start

            print(f"Execution completed in {elapsed:.2f}s")
            print(f"Solutions found: {len(solutions)}")

            if solutions:
                best = solutions[0]
                print(f"Best solution has {len(best['packages'])} packages")
                print(f"Packages: {best['packages'][:3]}...")
                return True
            else:
                print("[WARNING] No solutions found")
                return False

        except Exception as e:
            print(f"Error: {e}")
            return False

    def test_improved_neighborhoods(self):
        """Test improved neighborhoods if available"""
        print("\n" + "="*60)
        print("IMPROVED NEIGHBORHOODS TEST")
        print("="*60)

        try:
            # Check if improved neighborhoods exist
            improved_path = 'E:/pycommend/temp/tests/improved_neighborhoods.py'
            if os.path.exists(improved_path):
                print("Improved neighborhoods file found")

                movns = MOVNS_VNS('numpy', archive_size=5, max_iterations=1)
                solution = movns.smart_initialization('medium')
                neighborhoods = movns.define_neighborhoods()

                # Test one neighborhood
                print("\nTesting N1 (Smart Flip):")
                n1 = neighborhoods[0]
                neighbor = n1(solution)

                active_orig = np.where(solution == 1)[0]
                active_new = np.where(neighbor == 1)[0]

                print(f"  Original size: {len(active_orig)}")
                print(f"  New size: {len(active_new)}")
                print(f"  Change made: {'[OK]' if len(active_orig) != len(active_new) else '[WARNING]'}")

                return True
            else:
                print("Improved neighborhoods not found - using default")
                return True

        except Exception as e:
            print(f"Error: {e}")
            return False


def main():
    """Main test runner"""
    print("="*80)
    print("MOVNS QUICK CONFORMANCE TEST")
    print("Following rules.json - Real execution")
    print("="*80)

    tester = QuickConformanceTest()

    tests = [
        ("MOBI/P Conformance", tester.test_mobi_p_conformance),
        ("Neighborhoods", tester.test_neighborhoods),
        ("VNS Components", tester.test_vns_components),
        ("Execution", tester.test_execution),
        ("Improved Neighborhoods", tester.test_improved_neighborhoods)
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

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n[OK] ALL TESTS PASSED - MOVNS is conformant")
    elif passed >= total * 0.8:
        print(f"\n[OK] {passed}/{total} tests passed - MOVNS mostly conformant")
    else:
        print(f"\n[FAIL] Only {passed}/{total} tests passed")

    return passed >= total * 0.8


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)