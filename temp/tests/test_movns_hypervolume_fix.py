"""
Test MOVNS Hypervolume After Fix
Following rules.json - Real execution in background
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2
from evaluation.metrics import calculate_hypervolume


def test_hypervolume_improvement():
    """Test if MOVNS achieves competitive hypervolume after fix"""

    print("="*80)
    print("MOVNS HYPERVOLUME TEST AFTER FIX")
    print("="*80)

    package = 'numpy'
    iterations = 30

    # Test MOVNS with fixed initialization
    print("\n1. Testing MOVNS with improved initialization:")
    print("-"*60)
    start = time.time()
    movns = MOVNS_VNS(package, archive_size=50, max_iterations=iterations, track_metrics=True)
    movns_sols = movns.run()
    movns_time = time.time() - start

    movns_metrics = movns.get_metrics_history()
    movns_hv = movns_metrics['hypervolume'][-1] if movns_metrics and 'hypervolume' in movns_metrics else 0

    print(f"✓ MOVNS completed")
    print(f"  Solutions: {len(movns_sols)}")
    print(f"  Time: {movns_time:.2f}s")
    print(f"  Hypervolume: {movns_hv:.4f}")

    # Show evolution of hypervolume
    if movns_metrics and 'hypervolume' in movns_metrics:
        hv_history = movns_metrics['hypervolume']
        print(f"  HV evolution: {hv_history[0]:.4f} -> {hv_history[-1]:.4f}")

    # Test NSGA-II baseline
    print("\n2. Testing NSGA-II baseline:")
    print("-"*60)
    start = time.time()
    nsga2 = NSGA2(package, pop_size=50, max_gen=iterations)
    nsga2_sols = nsga2.run()
    nsga2_time = time.time() - start

    # Calculate NSGA-II hypervolume
    nsga2_objs = [
        [sol['objectives']['linked_usage'],
         sol['objectives']['semantic_similarity'],
         -sol['objectives']['set_size']]
        for sol in nsga2_sols
    ]
    nsga2_hv = calculate_hypervolume(nsga2_objs, [0, 0, -20])

    print(f"✓ NSGA-II completed")
    print(f"  Solutions: {len(nsga2_sols)}")
    print(f"  Time: {nsga2_time:.2f}s")
    print(f"  Hypervolume: {nsga2_hv:.4f}")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\nHypervolume (Quality Metric):")
    print(f"  MOVNS:   {movns_hv:.4f}")
    print(f"  NSGA-II: {nsga2_hv:.4f}")

    ratio = (movns_hv / nsga2_hv) * 100 if nsga2_hv > 0 else 0
    print(f"  Ratio:   {ratio:.1f}%")

    print(f"\nExecution Time:")
    print(f"  MOVNS:   {movns_time:.2f}s")
    print(f"  NSGA-II: {nsga2_time:.2f}s")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if movns_hv >= nsga2_hv:
        print(f"✓ SUCCESS: MOVNS achieves {ratio:.1f}% of NSGA-II hypervolume")
        print("✓ MOVNS is ready for VNS paper")
    elif movns_hv >= nsga2_hv * 0.9:
        print(f"✓ ACCEPTABLE: MOVNS achieves {ratio:.1f}% of NSGA-II hypervolume")
        print("✓ MOVNS is competitive for VNS paper")
    else:
        print(f"✗ NEEDS WORK: MOVNS only achieves {ratio:.1f}% of NSGA-II hypervolume")
        print("✗ Further optimization required")

    # Show best solutions
    if movns_sols:
        print("\nMOVNS Best Solution:")
        best = movns_sols[0]
        print(f"  Packages: {', '.join(best['packages'][:5])}...")
        print(f"  LU={best['objectives']['linked_usage']:.2f}")
        print(f"  SS={best['objectives']['semantic_similarity']:.4f}")
        print(f"  Size={best['objectives']['set_size']:.0f}")

    if nsga2_sols:
        print("\nNSGA-II Best Solution:")
        best = nsga2_sols[0]
        print(f"  Packages: {', '.join([k for k, v in enumerate(best['chromosome']) if v == 1][:5])}...")
        print(f"  LU={best['objectives']['linked_usage']:.2f}")
        print(f"  SS={best['objectives']['semantic_similarity']:.4f}")
        print(f"  Size={best['objectives']['set_size']:.0f}")

    return {
        'movns_hv': movns_hv,
        'nsga2_hv': nsga2_hv,
        'ratio': ratio,
        'success': movns_hv >= nsga2_hv * 0.9
    }


if __name__ == '__main__':
    result = test_hypervolume_improvement()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    if result['success']:
        print("✓ MOVNS hypervolume issue RESOLVED")
    else:
        print("✗ MOVNS hypervolume still needs improvement")