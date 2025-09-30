"""
Test MOVNS HV evolution - show HV at each iteration
No shortcuts, real execution following rules.json
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOVNS v22 - HV Evolution Test")
print("="*70)

# Create MOVNS instance
package_name = 'fastapi'
movns = MOVNS_V22(package_name, archive_size=50, max_iterations=10, track_metrics=False)

print(f"\nTesting with {package_name}")
print(f"Max iterations: 10")
print(f"Archive size: 50")

# Initialize archive
if len(movns.archive) == 0:
    initial = movns.smart_initialization(exploration_rate=0.3)[0]
    initial_obj = movns.evaluate_objectives(initial)
    movns.update_archive(initial, initial_obj)

print(f"\nInitial archive: {len(movns.archive)} solutions")

# Track HV at each iteration
for iteration in range(10):
    # Get current solution from archive
    if len(movns.archive) > 0:
        current_idx = np.random.randint(len(movns.archive))
        current = movns.archive[current_idx]['chromosome'].copy()
        current_obj = movns.archive[current_idx]['objectives'].copy()

    # VNS iteration
    k = 0
    while k < movns.k_max:
        neighbor = movns.get_neighborhood(current, k)
        neighbor_obj = movns.evaluate_objectives(neighbor)

        # PLS with probability
        if np.random.random() < movns.pls_probability:
            improved_neighbor, improved = movns.simple_local_search(neighbor, movns.pls_max_neighbors)
            if improved:
                neighbor = improved_neighbor
                neighbor_obj = movns.evaluate_objectives(neighbor)

        movns.update_archive(neighbor, neighbor_obj)

        if movns.dominates(neighbor_obj, current_obj):
            current = neighbor
            current_obj = neighbor_obj
            k = 0
        else:
            k += 1

    # Calculate HV after this iteration
    if len(movns.archive) > 0:
        objectives = np.array([sol['objectives'] for sol in movns.archive])

        # Print raw objectives to understand scale
        print(f"\nIteration {iteration+1}:")
        print(f"  Archive size: {len(movns.archive)}")
        print(f"  Objectives sample (first 3):")
        for i, obj in enumerate(objectives[:3]):
            print(f"    Sol {i+1}: LU={obj[0]:.1f}, SS={obj[1]:.3f}, RSS={obj[2]}")

        # Calculate HV with proper normalization
        qm = QualityMetrics()

        # Method 1: Direct HV with reference point
        ref_point = [0, 0, 15]  # For minimization objectives
        hv1 = qm.hypervolume(objectives, ref_point=ref_point)
        print(f"  HV (ref=[0,0,15]): {hv1:.4f}")

        # Method 2: Transform negatives and calculate
        transformed = objectives.copy()
        transformed[:, 0] = -transformed[:, 0]  # LU: make positive
        transformed[:, 1] = -transformed[:, 1]  # SS: make positive

        qm2 = QualityMetrics()
        ref_point2 = [20000, 1, 15]
        hv2 = qm2.hypervolume(transformed, ref_point=ref_point2)
        print(f"  HV (transformed): {hv2:.4f}")

        # Method 3: Normalized HV (0-1 scale)
        # Normalize each objective to [0, 1]
        norm_obj = objectives.copy()
        # LU: typically -20000 to 0
        norm_obj[:, 0] = (norm_obj[:, 0] + 20000) / 20000
        # SS: typically -1 to 0
        norm_obj[:, 1] = (norm_obj[:, 1] + 1) / 1
        # RSS: typically 0 to 15
        norm_obj[:, 2] = norm_obj[:, 2] / 15

        qm3 = QualityMetrics()
        hv3 = qm3.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])
        print(f"  HV (normalized 0-1): {hv3:.4f}")

print("\n" + "="*70)
print("Analysis complete")
print("="*70)