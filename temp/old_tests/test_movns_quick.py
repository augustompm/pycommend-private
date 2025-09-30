"""
Quick test of MOVNS Advanced vs MOEA/D
Reduced iterations for faster feedback
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized


def quick_test():
    """Quick comparison test"""
    print("="*60)
    print("QUICK TEST: MOVNS ADVANCED VS MOEA/D")
    print("="*60)

    package = 'fastapi'

    print(f"\n1. Testing MOVNS Advanced (10 iterations)")
    print("-"*60)

    algo = MOVNS_Advanced(
        package,
        archive_size=50,
        max_iterations=10,
        track_metrics=True
    )

    start = time.time()
    solutions = algo.run()
    movns_time = time.time() - start

    metrics = algo.get_metrics_history()
    movns_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

    print(f"\nResults:")
    print(f"  Hypervolume: {movns_hv:.4f}")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {movns_time:.1f}s")

    best_lu = 0
    for sol in solutions:
        obj = algo.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_lu:
            best_lu = -obj[0]
    print(f"  Best LU: {best_lu:.0f}")

    print(f"\n2. Testing MOEA/D (10 generations)")
    print("-"*60)

    algo = MOEAD_Normalized(
        package,
        pop_size=50,
        max_gen=10
    )

    start = time.time()
    solutions = algo.run()
    moead_time = time.time() - start

    metrics = algo.get_metrics_history()
    moead_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

    print(f"\nResults:")
    print(f"  Hypervolume: {moead_hv:.4f}")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {moead_time:.1f}s")

    best_lu = 0
    for sol in solutions:
        obj = algo.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_lu:
            best_lu = -obj[0]
    print(f"  Best LU: {best_lu:.0f}")

    print(f"\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    print(f"\nHypervolume:")
    print(f"  MOVNS Advanced: {movns_hv:.4f}")
    print(f"  MOEA/D:         {moead_hv:.4f}")

    if moead_hv > 0:
        ratio = movns_hv / moead_hv
        print(f"\nRatio: {ratio*100:.1f}%")

        if ratio > 1.0:
            print(f"MOVNS Advanced beats MOEA/D by {(ratio-1)*100:.1f}%")
        else:
            print(f"MOEA/D still superior by {(1-ratio)*100:.1f}%")

    print(f"\nExecution Time:")
    print(f"  MOVNS Advanced: {movns_time:.1f}s")
    print(f"  MOEA/D:         {moead_time:.1f}s")


if __name__ == "__main__":
    quick_test()