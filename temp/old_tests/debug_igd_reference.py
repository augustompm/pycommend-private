"""
Debug IGD+ reference set generation
"""

import sys
import os
import numpy as np

os.chdir('pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from nsga2_vns import NSGA2_VNS
from quality_metrics import QualityMetrics


def test_reference_generation():
    """Test reference set generation and IGD+ calculation"""

    print("="*70)
    print("DEBUG: IGD+ REFERENCE SET")
    print("="*70)

    # Create NSGA-II instance
    nsga2 = NSGA2_VNS('fastapi', pop_size=10, max_gen=1, track_metrics=True)

    # Generate reference set
    nsga2.generate_reference_set()

    print(f"\nReference set generated:")
    print(f"  Size: {len(nsga2.reference_set)}")
    print(f"  Shape: {nsga2.reference_set.shape}")

    # Show some reference points
    print("\nSample reference points:")
    for i in range(min(5, len(nsga2.reference_set))):
        ref = nsga2.reference_set[i]
        print(f"  Ref {i+1}: LU={ref[0]:.1f}, SS={ref[1]:.3f}, RSS={ref[2]:.1f}")

    # Get an initial population
    print("\nInitializing population...")
    population = nsga2.initialize_population()

    # Get Pareto front
    fronts = nsga2.fast_non_dominated_sort(population)
    if fronts and fronts[0]:
        pareto_indices = fronts[0]
        pareto_objectives = np.array([population[i]['objectives'] for i in pareto_indices])

        print(f"\nPareto front:")
        print(f"  Size: {len(pareto_objectives)}")

        # Show some Pareto points
        print("\nSample Pareto points:")
        for i in range(min(5, len(pareto_objectives))):
            obj = pareto_objectives[i]
            print(f"  Sol {i+1}: LU={obj[0]:.1f}, SS={obj[1]:.3f}, RSS={obj[2]:.1f}")

        # Calculate IGD+ manually
        metrics_calc = QualityMetrics()

        print("\n" + "-"*70)
        print("IGD+ CALCULATION TEST")
        print("-"*70)

        # Test 1: IGD+ with the generated reference set
        igd_plus = metrics_calc.igd_plus(pareto_objectives, nsga2.reference_set)
        print(f"\nIGD+ (Pareto vs Generated Reference): {igd_plus:.6f}")

        # Test 2: Create an ideal reference set (much better than current)
        ideal_reference = np.array([
            [-10000, -1.0, 2],   # Very high LU, perfect similarity, min size
            [-8000, -0.9, 3],
            [-6000, -0.8, 4],
            [-4000, -0.7, 5],
            [-3000, -0.6, 6],
        ])

        igd_plus_ideal = metrics_calc.igd_plus(pareto_objectives, ideal_reference)
        print(f"IGD+ (Pareto vs Ideal Reference): {igd_plus_ideal:.6f}")

        # Test 3: Create a worse reference set
        worse_reference = np.array([
            [-100, -0.1, 15],   # Low LU, low similarity, high size
            [-200, -0.2, 14],
            [-300, -0.3, 13],
            [-400, -0.4, 12],
            [-500, -0.5, 11],
        ])

        igd_plus_worse = metrics_calc.igd_plus(pareto_objectives, worse_reference)
        print(f"IGD+ (Pareto vs Worse Reference): {igd_plus_worse:.6f}")

        # Test 4: Check if reference dominates or is dominated
        print("\n" + "-"*70)
        print("DOMINANCE ANALYSIS")
        print("-"*70)

        # Check how many reference points dominate Pareto points
        ref_dominates = 0
        pareto_dominates = 0
        incomparable = 0

        for ref_point in nsga2.reference_set[:10]:  # Check first 10 ref points
            for pareto_point in pareto_objectives:
                if nsga2.dominates(ref_point, pareto_point):
                    ref_dominates += 1
                elif nsga2.dominates(pareto_point, ref_point):
                    pareto_dominates += 1
                else:
                    incomparable += 1

        total_comparisons = min(10, len(nsga2.reference_set)) * len(pareto_objectives)
        print(f"Total comparisons: {total_comparisons}")
        print(f"Reference dominates Pareto: {ref_dominates} ({ref_dominates/total_comparisons*100:.1f}%)")
        print(f"Pareto dominates Reference: {pareto_dominates} ({pareto_dominates/total_comparisons*100:.1f}%)")
        print(f"Incomparable: {incomparable} ({incomparable/total_comparisons*100:.1f}%)")

        # Analyze why IGD+ might be zero
        print("\n" + "-"*70)
        print("WHY IS IGD+ ZERO?")
        print("-"*70)

        # Check the actual IGD+ calculation step by step
        from sklearn.preprocessing import MinMaxScaler

        # Normalize both sets (as done in IGD+)
        all_points = np.vstack([pareto_objectives, nsga2.reference_set])
        scaler = MinMaxScaler()
        scaler.fit(all_points)

        norm_pareto = scaler.transform(pareto_objectives)
        norm_ref = scaler.transform(nsga2.reference_set)

        print("\nNormalized Pareto range:")
        print(f"  LU: [{norm_pareto[:, 0].min():.3f}, {norm_pareto[:, 0].max():.3f}]")
        print(f"  SS: [{norm_pareto[:, 1].min():.3f}, {norm_pareto[:, 1].max():.3f}]")
        print(f"  RSS: [{norm_pareto[:, 2].min():.3f}, {norm_pareto[:, 2].max():.3f}]")

        print("\nNormalized Reference range:")
        print(f"  LU: [{norm_ref[:, 0].min():.3f}, {norm_ref[:, 0].max():.3f}]")
        print(f"  SS: [{norm_ref[:, 1].min():.3f}, {norm_ref[:, 1].max():.3f}]")
        print(f"  RSS: [{norm_ref[:, 2].min():.3f}, {norm_ref[:, 2].max():.3f}]")

        # Calculate IGD+ distances manually for first reference point
        ref_point = norm_ref[0]
        print(f"\nFirst reference point (normalized): {ref_point}")

        min_distance = float('inf')
        for i, pareto_point in enumerate(norm_pareto):
            # IGD+ distance (only counts where ref is worse)
            diff = np.maximum(ref_point - pareto_point, 0)
            distance = np.linalg.norm(diff)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        print(f"Closest Pareto point (normalized): {norm_pareto[closest_idx]}")
        print(f"Difference vector: {np.maximum(ref_point - norm_pareto[closest_idx], 0)}")
        print(f"Distance: {min_distance:.6f}")

        if min_distance == 0:
            print("\nDistance is 0 because reference point is dominated by or equal to Pareto point!")


if __name__ == '__main__':
    test_reference_generation()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nIGD+ is zero when:")
    print("1. All reference points are dominated by the Pareto front")
    print("2. The reference set is identical to or subset of the Pareto front")
    print("3. The normalization makes distances negligible")
    print("\nFor proper IGD+, the reference set should represent")
    print("an ideal or aspirational Pareto front that is better")
    print("than what the algorithm can currently achieve.")