"""
Test MOVNS v6 - VNS with Decomposition Neighborhood
Compares v2 (pure VNS), v6 (VNS+decomposition), and MOEA/D
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v2 import MOVNS_V2
from optimizer.movns_v6 import MOVNS_V6
from optimizer.moead_normalized import MOEAD_Normalized

def test_algorithm(algo_name, package='fastapi', iterations=30):
    """Test an algorithm and return results"""
    print(f"\nTesting {algo_name}...")
    print("-"*40)

    start = time.time()

    if algo_name == "MOVNS v2":
        algo = MOVNS_V2(package, archive_size=50, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=5)
    elif algo_name == "MOVNS v6":
        algo = MOVNS_V6(package, archive_size=50, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=5, n_weight_vectors=10)
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
    """Run comparison between v2, v6, and MOEA/D"""
    print("="*60)
    print("MOVNS v6 - DECOMPOSITION NEIGHBORHOOD TEST")
    print("="*60)
    print("Testing VNS with decomposition-guided neighborhood")

    results = []
    iterations = 30

    results.append(test_algorithm("MOVNS v2", iterations=iterations))
    results.append(test_algorithm("MOVNS v6", iterations=iterations))
    results.append(test_algorithm("MOEA/D", iterations=iterations))

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\n{'Algorithm':<15} {'Final HV':<12} {'Improvement':<15} {'Time (s)':<10} {'Solutions':<10}")
    print("-"*75)

    for r in results:
        algo = r['algorithm']
        hv = r.get('final_hv', 0)
        imp = r.get('improvement', 0)
        time_taken = r['time']
        sols = r['solutions']

        print(f"{algo:<15} {hv:<12.4f} {imp:<15.1f}% {time_taken:<10.1f} {sols:<10}")

    best = max(results, key=lambda x: x.get('final_hv', 0))
    print("\n" + "="*60)
    print(f"BEST HYPERVOLUME: {best['algorithm']} ({best.get('final_hv', 0):.4f})")

    v2_hv = results[0].get('final_hv', 0.001)
    v6_hv = results[1].get('final_hv', 0.001)
    moead_hv = results[2].get('final_hv', 0.001)

    print(f"\nPerformance Analysis:")
    print(f"  MOVNS v6 vs v2: {(v6_hv/v2_hv - 1)*100:+.1f}%")
    print(f"  MOVNS v6 vs MOEA/D: {(v6_hv/moead_hv - 1)*100:+.1f}%")
    print(f"  MOVNS v6 as % of MOEA/D: {(v6_hv/moead_hv)*100:.1f}%")

    print("\n" + "="*60)
    if v6_hv > v2_hv * 1.1:
        improvement = (v6_hv / v2_hv - 1) * 100
        print(f"SUCCESS: v6 improves over v2 by {improvement:.1f}%")
        print(f"Decomposition neighborhood adds value to VNS")

        if v6_hv > moead_hv * 0.85:
            print(f"\nEXCELLENT: v6 achieves {(v6_hv/moead_hv)*100:.1f}% of MOEA/D performance")
            print("Competitive with state-of-the-art while maintaining VNS identity")
    else:
        print(f"v6 performance similar to v2 ({(v6_hv/v2_hv)*100:.1f}%)")
        print(f"Decomposition overhead may not be justified")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    print("\nMOVNS v6 Design:")
    print("  - 3 traditional VNS neighborhoods")
    print("  - 1 decomposition-guided neighborhood")
    print("  - 10 pre-computed weight vectors")
    print("  - Tchebycheff decomposition for gap identification")
    print("  - Specialized local search for decomposition moves")

    print("\nExpected Benefits:")
    print("  - Better exploration of sparse regions")
    print("  - Directed search when stuck")
    print("  - Maintains VNS structure and identity")
    print("  - Lower overhead than full hybridization")

    print("="*60)

if __name__ == "__main__":
    main()