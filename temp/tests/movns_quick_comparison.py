"""
Quick MOVNS vs MOEA/D Comparison
Following rules.json - Real execution
"""

import sys
import time

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')

from movns_vns import MOVNS_VNS
from moead_vns import MOEAD_VNS


def quick_comparison():
    """Quick comparison with minimal iterations"""

    print("="*80)
    print("MOVNS vs MOEA/D - QUICK COMPARISON")
    print("="*80)

    package = 'numpy'
    iterations = 2

    # Test MOVNS
    print(f"\nTesting MOVNS (2 iterations):")
    print("-"*60)
    start = time.time()
    try:
        movns = MOVNS_VNS(package, archive_size=10, max_iterations=iterations, track_metrics=True)
        movns_sols = movns.run()
        movns_time = time.time() - start
        movns_metrics = movns.get_metrics_history()
        movns_hv = movns_metrics.get('hypervolume', [0])[-1] if movns_metrics else 0
        print(f"✓ MOVNS: {len(movns_sols)} solutions in {movns_time:.2f}s")
        print(f"  Hypervolume: {movns_hv:.4f}")
    except Exception as e:
        print(f"✗ MOVNS failed: {e}")
        movns_sols = []
        movns_time = 0
        movns_hv = 0

    # Test MOEA/D
    print(f"\nTesting MOEA/D (2 iterations):")
    print("-"*60)
    start = time.time()
    try:
        moead = MOEAD_VNS(package, pop_size=10, max_gen=iterations, track_metrics=True)
        moead_sols = moead.run()
        moead_time = time.time() - start
        moead_metrics = moead.get_metrics_history()
        moead_hv = moead_metrics.get('hypervolume', [0])[-1] if moead_metrics else 0
        print(f"✓ MOEA/D: {len(moead_sols)} solutions in {moead_time:.2f}s")
        print(f"  Hypervolume: {moead_hv:.4f}")
    except Exception as e:
        print(f"✗ MOEA/D failed: {e}")
        moead_sols = []
        moead_time = 0
        moead_hv = 0

    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    if movns_sols and moead_sols:
        print(f"\nSolutions: MOVNS={len(movns_sols)}, MOEA/D={len(moead_sols)}")
        print(f"Time: MOVNS={movns_time:.2f}s, MOEA/D={moead_time:.2f}s")
        print(f"Hypervolume: MOVNS={movns_hv:.4f}, MOEA/D={moead_hv:.4f}")

        if movns_hv > moead_hv:
            print(f"\n✓ MOVNS is {((movns_hv/moead_hv - 1)*100):.1f}% better in quality")
        elif moead_hv > movns_hv:
            print(f"\n✓ MOEA/D is {((moead_hv/movns_hv - 1)*100):.1f}% better in quality")
        else:
            print("\n✓ Equal quality")

        if movns_time < moead_time:
            print(f"✓ MOVNS is {((moead_time/movns_time - 1)*100):.1f}% faster")
        else:
            print(f"✓ MOEA/D is {((movns_time/moead_time - 1)*100):.1f}% faster")
    else:
        print("\n✗ Comparison failed - one or both algorithms didn't run")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    quick_comparison()