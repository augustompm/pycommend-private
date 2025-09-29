"""
Compare algorithms with proper normalization for v12
All algorithms use normalized objectives for fair comparison
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_vns import MOVNS_VNS
from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized

def normalize_objectives(objectives, obj_min, obj_max):
    """Normalize objectives to [0,1] range"""
    objectives = np.array(objectives)  # Ensure it's numpy array
    norm_obj = np.zeros_like(objectives)
    for i in range(len(objectives)):
        if obj_max[i] - obj_min[i] != 0:
            norm_obj[i] = (objectives[i] - obj_min[i]) / (obj_max[i] - obj_min[i])
    return np.clip(norm_obj, 0, 1)

def calculate_normalized_hypervolume(solutions):
    """Calculate hypervolume with normalized objectives"""
    if not solutions:
        return 0.0

    # Extract objectives - handle both dict and array formats
    objectives = []
    for sol in solutions:
        if isinstance(sol, dict):
            obj = sol['objectives']
            # Handle objectives as dict (MOVNS original format)
            if isinstance(obj, dict):
                objectives.append([obj['linked_usage'], obj['semantic_similarity'], obj['set_size']])
            else:
                objectives.append(obj)
        else:
            objectives.append(sol)
    objectives = np.array(objectives)

    # Define bounds based on problem knowledge
    obj_min = np.array([-10000.0, -1.0, 2.0])  # LU, SS, RSS min
    obj_max = np.array([0.0, 0.0, 15.0])        # LU, SS, RSS max

    # Normalize all objectives
    normalized_objectives = np.array([normalize_objectives(obj, obj_min, obj_max) for obj in objectives])

    # Simple hypervolume approximation with normalized values
    reference_point = np.array([1.1, 1.1, 1.1])  # Slightly beyond [1,1,1]

    volume = 0.0
    for norm_obj in normalized_objectives:
        # Calculate volume contribution
        contribution = 1.0
        for i in range(3):
            diff = reference_point[i] - norm_obj[i]
            if diff > 0:
                contribution *= diff
            else:
                contribution = 0
                break
        volume += contribution

    return volume / len(normalized_objectives)  # Average contribution

def test_algorithm(algo_name, package='fastapi', iterations=30):
    """Test an algorithm and return results with normalized HV"""
    print(f"\nTesting {algo_name}...")

    start = time.time()

    if algo_name == "MOVNS Original":
        algo = MOVNS_VNS(package, archive_size=50, max_iterations=iterations)
        solutions = algo.run()
    elif algo_name == "MOVNS v2":
        algo = MOVNS_V2(package, archive_size=50, max_iterations=iterations,
                       track_metrics=True, min_no_improvement=10)
        solutions = algo.run()
    elif algo_name == "MOEA/D Normalized":
        algo = MOEAD_Normalized(package, pop_size=50, max_gen=iterations,
                                track_metrics=True)
        solutions = algo.run()

    exec_time = time.time() - start

    # Calculate normalized hypervolume
    hv_normalized = calculate_normalized_hypervolume(solutions)

    # Get best values for each objective
    objectives = []
    for sol in solutions:
        obj = sol['objectives']
        if isinstance(obj, dict):
            objectives.append([obj['linked_usage'], obj['semantic_similarity'], obj['set_size']])
        else:
            objectives.append(obj)
    objectives = np.array(objectives)

    best_lu = -np.min(objectives[:, 0]) if len(objectives) > 0 else 0
    best_ss = -np.min(objectives[:, 1]) if len(objectives) > 0 else 0
    best_rss = np.min(objectives[:, 2]) if len(objectives) > 0 else 0

    print(f"  Solutions: {len(solutions)}")
    print(f"  Normalized HV: {hv_normalized:.4f}")
    print(f"  Best LU: {best_lu:.1f}")
    print(f"  Best SS: {best_ss:.4f}")
    print(f"  Best RSS: {best_rss:.1f}")
    print(f"  Time: {exec_time:.1f}s")

    return {
        'algo': algo_name,
        'solutions': len(solutions),
        'hv_normalized': hv_normalized,
        'best_lu': best_lu,
        'best_ss': best_ss,
        'best_rss': best_rss,
        'time': exec_time
    }

def main():
    """Run comparison with normalized metrics"""
    print("=" * 60)
    print("V12 ALGORITHM COMPARISON - NORMALIZED METRICS")
    print("=" * 60)
    print("All hypervolume values calculated with normalized objectives [0,1]")
    print("This ensures fair comparison across algorithms")
    print("=" * 60)

    results = []

    # Test all three algorithms
    results.append(test_algorithm("MOVNS Original", iterations=25))
    results.append(test_algorithm("MOVNS v2", iterations=25))
    results.append(test_algorithm("MOEA/D Normalized", iterations=25))

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'HV (Norm)':<12} {'Solutions':<12} {'Time (s)':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['algo']:<20} {r['hv_normalized']:<12.4f} {r['solutions']:<12} {r['time']:<10.1f}")

    # Find best performer
    best_hv = max(results, key=lambda x: x['hv_normalized'])
    print("\n" + "=" * 60)
    print(f"BEST NORMALIZED HYPERVOLUME: {best_hv['algo']} ({best_hv['hv_normalized']:.4f})")
    print("=" * 60)

    # Calculate ratios
    if len(results) == 3:
        movns_orig = results[0]['hv_normalized']
        movns_v2 = results[1]['hv_normalized']
        moead = results[2]['hv_normalized']

        print("\nPerformance Ratios (normalized HV):")
        print(f"  MOVNS v2 vs Original: {movns_v2/movns_orig:.1%}")
        print(f"  MOEA/D vs MOVNS Original: {moead/movns_orig:.1%}")
        print(f"  MOEA/D vs MOVNS v2: {moead/movns_v2:.1%}")

    print("\nNOTE: The HV values from v7 (0.5616) were calculated with")
    print("non-normalized objectives, making them incomparable.")
    print("These normalized values provide fair algorithm comparison.")

if __name__ == "__main__":
    main()