"""
Profile MOVNS performance to find bottleneck
"""

import sys
import time
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2


def profile_movns_single_iteration():
    """Profile MOVNS with just 1 iteration"""
    print("="*80)
    print("PROFILING MOVNS - 1 ITERATION")
    print("="*80)

    # Create profiler
    pr = cProfile.Profile()

    # Profile MOVNS
    pr.enable()
    start = time.time()
    movns = MOVNS_VNS('numpy', archive_size=20, max_iterations=1, track_metrics=False)
    solutions = movns.run()
    elapsed = time.time() - start
    pr.disable()

    print(f"\nMOVNS Results:")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {elapsed:.2f}s")

    # Print top time consumers
    print("\nTop 10 time-consuming functions:")
    print("-"*60)
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)

    # Parse and show relevant lines
    for line in s.getvalue().split('\n'):
        if 'movns_vns' in line or 'evaluate' in line or 'mobi_p' in line or 'vns_local' in line:
            print(line)

    return elapsed


def compare_with_nsga2():
    """Compare with NSGA-II for reference"""
    print("\n" + "="*80)
    print("COMPARING WITH NSGA-II - 1 GENERATION")
    print("="*80)

    start = time.time()
    nsga2 = NSGA2('numpy', pop_size=20, max_gen=1)
    solutions = nsga2.run()
    elapsed = time.time() - start

    print(f"\nNSGA-II Results:")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {elapsed:.2f}s")

    return elapsed


def analyze_mobi_p_cost():
    """Analyze MOBI/P cost specifically"""
    print("\n" + "="*80)
    print("ANALYZING MOBI/P COST")
    print("="*80)

    # Initialize MOVNS
    movns = MOVNS_VNS('numpy', archive_size=20, max_iterations=1, track_metrics=False)

    # Test MOBI/P with different sample sizes
    test_solution = list(movns.archive)[0] if movns.archive else None

    if test_solution:
        for samples in [1, 5, 10, 20]:
            start = time.time()
            movns.mobi_p_local_search(test_solution, samples=samples, neighbor_func=movns.neighborhoods[0])
            elapsed = time.time() - start
            print(f"  MOBI/P with {samples:2d} samples: {elapsed:.3f}s")


if __name__ == '__main__':
    # Profile MOVNS
    movns_time = profile_movns_single_iteration()

    # Compare with NSGA-II
    nsga2_time = compare_with_nsga2()

    # Analyze MOBI/P
    analyze_mobi_p_cost()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"MOVNS (1 iter): {movns_time:.2f}s")
    print(f"NSGA-II (1 gen): {nsga2_time:.2f}s")
    print(f"Ratio: MOVNS is {movns_time/nsga2_time:.1f}x slower")

    if movns_time > 5:
        print("\n⚠ WARNING: MOVNS is too slow for practical use!")
        print("  For a recommendation system, response should be <1s")