"""
Test MOVNS v5 - Decomposition-Guided VNS
Compare against MOEA/D to validate hybrid approach
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v5 import MOVNS_V5
from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized

def test_algorithm(algo_name, package='fastapi', iterations=40):
    """Test an algorithm and return results"""
    print(f"\nTesting {algo_name}...")
    print("-"*40)

    start = time.time()

    if algo_name == "MOVNS v5":
        algo = MOVNS_V5(package, archive_size=100, max_iterations=iterations,
                       n_weight_vectors=30, track_metrics=True, min_no_improvement=10)
    elif algo_name == "MOVNS v2":
        algo = MOVNS_V2(package, archive_size=100, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=10)
    elif algo_name == "MOEA/D":
        algo = MOEAD_Normalized(package, pop_size=50, max_gen=iterations,
                                track_metrics=True)

    solutions = algo.run()
    exec_time = time.time() - start

    result = {
        'algorithm': algo_name,
        'solutions': len(solutions),
        'time': exec_time
    }

    # Get metrics
    metrics = algo.get_metrics_history() if hasattr(algo, 'get_metrics_history') else None

    if metrics and 'hypervolume' in metrics:
        hv = metrics['hypervolume']
        if len(hv) > 0:
            result['initial_hv'] = hv[0]
            result['final_hv'] = hv[-1]
            result['max_hv'] = max(hv)
            result['improvement'] = ((hv[-1] - hv[0]) / hv[0] * 100) if hv[0] > 0 else 0

            print(f"  Initial HV: {hv[0]:.4f}")
            print(f"  Final HV: {hv[-1]:.4f}")
            print(f"  Max HV: {max(hv):.4f}")
            print(f"  Improvement: {result['improvement']:.1f}%")

    # Get best objectives
    objectives_list = []
    for sol in solutions:
        if isinstance(sol, dict) and 'objectives' in sol:
            obj = sol['objectives']
            if isinstance(obj, dict):
                objectives_list.append([
                    obj['linked_usage'],
                    obj['semantic_similarity'],
                    obj['set_size']
                ])

    if objectives_list:
        objectives = np.array(objectives_list)
        result['best_lu'] = np.max(objectives[:, 0])
        result['best_ss'] = np.max(objectives[:, 1])
        result['best_rss'] = np.min(objectives[:, 2])

        print(f"\n  Best objectives:")
        print(f"    LU: {result['best_lu']:.1f}")
        print(f"    SS: {result['best_ss']:.4f}")
        print(f"    RSS: {result['best_rss']:.1f}")

    print(f"\n  Solutions: {result['solutions']}, Time: {result['time']:.1f}s")

    return result

def main():
    """Run comprehensive comparison"""
    print("="*60)
    print("MOVNS v5 - DECOMPOSITION-GUIDED VNS TEST")
    print("="*60)
    print("Testing hybrid approach: VNS + Decomposition")

    results = []
    iterations = 40  # Fair comparison

    # Test all algorithms
    results.append(test_algorithm("MOEA/D", iterations=iterations))
    results.append(test_algorithm("MOVNS v2", iterations=iterations))
    results.append(test_algorithm("MOVNS v5", iterations=iterations))

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\n{'Algorithm':<15} {'Final HV':<12} {'Improvement':<15} {'Time (s)':<10} {'Solutions':<10}")
    print("-"*75)

    for r in results:
        algo = r['algorithm']
        hv = r.get('final_hv', 0)
        imp = r.get('improvement', 0)
        time = r['time']
        sols = r['solutions']

        print(f"{algo:<15} {hv:<12.4f} {imp:<15.1f}% {time:<10.1f} {sols:<10}")

    # Determine winner
    best = max(results, key=lambda x: x.get('final_hv', 0))
    print("\n" + "="*60)
    print(f"BEST HYPERVOLUME: {best['algorithm']} ({best.get('final_hv', 0):.4f})")

    # Calculate improvements
    moead_hv = results[0].get('final_hv', 0.001)
    v2_hv = results[1].get('final_hv', 0.001)
    v5_hv = results[2].get('final_hv', 0.001)

    print(f"\nPerformance Analysis:")
    print(f"  MOVNS v5 vs v2: {(v5_hv/v2_hv - 1)*100:+.1f}%")
    print(f"  MOVNS v5 vs MOEA/D: {(v5_hv/moead_hv - 1)*100:+.1f}%")

    # Success criteria
    print("\n" + "="*60)
    if v5_hv > moead_hv:
        improvement = (v5_hv / moead_hv - 1) * 100
        print(f"✓ SUCCESS: MOVNS v5 beats MOEA/D by {improvement:.1f}%!")
        print(f"  Decomposition-guided VNS works!")
    elif v5_hv > v2_hv * 1.1:
        improvement = (v5_hv / v2_hv - 1) * 100
        print(f"✓ IMPROVEMENT: MOVNS v5 is {improvement:.1f}% better than v2")
        print(f"  Decomposition helps VNS performance")
    else:
        deficit = (moead_hv / v5_hv - 1) * 100
        print(f"✗ MOVNS v5 is {deficit:.1f}% behind MOEA/D")
        print(f"  Further tuning needed")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print("\nMOVNS v5 Features:")
    print("  - 6 decomposition-based neighborhoods")
    print("  - Weight vectors guide local search")
    print("  - Tchebycheff and weighted sum decomposition")
    print("  - Adaptive weight selection")

    if v5_hv > v2_hv:
        print("\nImprovement over v2:")
        print("  ✓ Decomposition provides clear search directions")
        print("  ✓ Better coverage of Pareto front")
        print("  ✓ More efficient than random neighborhoods")
    else:
        print("\nPotential issues:")
        print("  - May need more weight vectors")
        print("  - Neighborhood implementation could be refined")
        print("  - Consider longer runs")

    print("="*60)

if __name__ == "__main__":
    main()