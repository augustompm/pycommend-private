"""
Test MOVNS v3 vs MOEA/D Normalized
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v3 import MOVNS_V3
from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized

def test_algorithm(algo_name, package='fastapi', iterations=30):
    """Test an algorithm and return metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {algo_name}")
    print('='*60)

    start = time.time()

    if algo_name == "MOVNS v3":
        algo = MOVNS_V3(package, archive_size=100, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=10)
    elif algo_name == "MOVNS v2":
        algo = MOVNS_V2(package, archive_size=50, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=10)
    elif algo_name == "MOEA/D Normalized":
        algo = MOEAD_Normalized(package, pop_size=50, max_gen=iterations,
                                track_metrics=True)

    solutions = algo.run()
    exec_time = time.time() - start

    metrics = algo.get_metrics_history() if hasattr(algo, 'get_metrics_history') else None

    result = {
        'algorithm': algo_name,
        'solutions': len(solutions),
        'time': exec_time
    }

    if metrics and 'hypervolume' in metrics:
        hv = metrics['hypervolume']
        if len(hv) > 0:
            result['initial_hv'] = hv[0] if hv[0] > 0 else 0.001
            result['final_hv'] = hv[-1]
            result['max_hv'] = max(hv)
            result['improvement'] = ((hv[-1] - hv[0]) / hv[0] * 100) if hv[0] > 0 else 0

            print(f"\nMetrics Summary:")
            print(f"  Initial HV: {result['initial_hv']:.4f}")
            print(f"  Final HV: {result['final_hv']:.4f}")
            print(f"  Max HV: {result['max_hv']:.4f}")
            print(f"  Improvement: {result['improvement']:.1f}%")

    # Get best objectives
    objectives_list = []
    for sol in solutions:
        if isinstance(sol, dict) and 'objectives' in sol:
            obj = sol['objectives']
            if isinstance(obj, dict):
                objectives_list.append([obj['linked_usage'], obj['semantic_similarity'], obj['set_size']])
            else:
                # Handle array format from MOEA/D
                objectives_list.append([-obj[0], -obj[1], obj[2]])
        else:
            # Handle tuple format (solution, objectives)
            if isinstance(sol, tuple) and len(sol) == 2:
                _, obj = sol
                objectives_list.append([-obj[0], -obj[1], obj[2]])

    if objectives_list:
        objectives = np.array(objectives_list)
        result['best_lu'] = np.max(objectives[:, 0])
        result['best_ss'] = np.max(objectives[:, 1])
        result['best_rss'] = np.min(objectives[:, 2])

        print(f"\nBest Objectives:")
        print(f"  LU: {result['best_lu']:.1f}")
        print(f"  SS: {result['best_ss']:.4f}")
        print(f"  RSS: {result['best_rss']:.1f}")

    print(f"\nSolutions: {result['solutions']}, Time: {result['time']:.1f}s")

    return result

def main():
    """Run comparison test"""
    print("="*60)
    print("MOVNS v3 vs MOEA/D COMPARISON TEST")
    print("="*60)
    print("Testing enhanced neighborhoods against decomposition")

    results = []

    # Test all algorithms
    results.append(test_algorithm("MOEA/D Normalized", iterations=25))
    results.append(test_algorithm("MOVNS v2", iterations=25))
    results.append(test_algorithm("MOVNS v3", iterations=25))

    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\n{'Algorithm':<20} {'Final HV':<12} {'Improvement':<15} {'Solutions':<12} {'Time (s)':<10}")
    print("-"*80)

    for r in results:
        algo = r['algorithm']
        hv = r.get('final_hv', 0)
        imp = r.get('improvement', 0)
        sols = r['solutions']
        time = r['time']

        print(f"{algo:<20} {hv:<12.4f} {imp:<15.1f}% {sols:<12} {time:<10.1f}")

    # Determine winner
    best_hv = max(results, key=lambda x: x.get('final_hv', 0))
    print("\n" + "="*60)
    print(f"BEST HYPERVOLUME: {best_hv['algorithm']} ({best_hv['final_hv']:.4f})")

    # Calculate improvement ratios
    moead_hv = results[0].get('final_hv', 0.001)
    v2_hv = results[1].get('final_hv', 0.001)
    v3_hv = results[2].get('final_hv', 0.001)

    print(f"\nPerformance Ratios:")
    print(f"  MOVNS v3 vs v2: {(v3_hv/v2_hv - 1)*100:+.1f}%")
    print(f"  MOVNS v3 vs MOEA/D: {(v3_hv/moead_hv - 1)*100:+.1f}%")

    if v3_hv > moead_hv:
        print(f"\nSUCCESS: MOVNS v3 surpasses MOEA/D by {(v3_hv/moead_hv - 1)*100:.1f}%!")
    elif v3_hv > v2_hv:
        print(f"\nIMPROVEMENT: MOVNS v3 is {(v3_hv/v2_hv - 1)*100:.1f}% better than v2")
    else:
        print(f"\nFurther optimization needed")

    print("="*60)

if __name__ == "__main__":
    main()