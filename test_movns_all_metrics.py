"""
Test MOVNS with ALL metrics evolution - HV, Spacing, Epsilon
Show metrics at each iteration to verify proper evolution
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOVNS v22 - All Metrics Evolution (HV, Spacing, Epsilon)")
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

# Track all metrics at each iteration
print("\n" + "-"*70)
print("Iter | Archive | HV (norm) | Spacing | Epsilon | LU Best")
print("-"*70)

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

    # Calculate metrics after this iteration
    if len(movns.archive) > 0:
        objectives = np.array([sol['objectives'] for sol in movns.archive])

        # Calculate metrics
        qm = QualityMetrics()

        # 1. Normalized HV (0-1 scale)
        norm_obj = objectives.copy()
        # Normalize each objective
        norm_obj[:, 0] = (norm_obj[:, 0] + 20000) / 20000  # LU
        norm_obj[:, 1] = (norm_obj[:, 1] + 1) / 1          # SS
        norm_obj[:, 2] = norm_obj[:, 2] / 15                # RSS

        hv = qm.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])

        # 2. Spacing (distribution uniformity)
        spacing = qm.spacing(objectives)

        # 3. Epsilon indicator (using archive as its own reference)
        # This shows convergence quality
        epsilon = qm.epsilon_indicator(objectives, reference_set=objectives)

        # 4. Best LU found
        best_lu = np.max(-objectives[:, 0])  # Remember LU is negative

        print(f"{iteration+1:4} | {len(movns.archive):7} | {hv:9.4f} | {spacing:7.4f} | {epsilon:7.4f} | {best_lu:7.0f}")

print("-"*70)

# Final analysis
objectives = np.array([sol['objectives'] for sol in movns.archive])
qm = QualityMetrics()

# Get non-dominated set
pareto_front = qm.get_non_dominated_set(objectives)

print(f"\nFinal Analysis:")
print(f"  Total archive: {len(movns.archive)} solutions")
print(f"  Non-dominated: {len(pareto_front)} solutions")
print(f"  Archive/Pareto ratio: {len(pareto_front)/len(movns.archive)*100:.1f}%")

# Calculate diversity
diversity = qm.diversity(objectives)
print(f"  Diversity: {diversity:.4f}")

# Show objective ranges
print(f"\nObjective ranges in final archive:")
print(f"  LU: [{np.min(objectives[:, 0]):.0f}, {np.max(objectives[:, 0]):.0f}]")
print(f"  SS: [{np.min(objectives[:, 1]):.3f}, {np.max(objectives[:, 1]):.3f}]")
print(f"  RSS: [{np.min(objectives[:, 2]):.0f}, {np.max(objectives[:, 2]):.0f}]")

print("\n" + "="*70)
print("Expected behavior:")
print("- HV should increase (more dominated space)")
print("- Spacing should decrease (better distribution)")
print("- Epsilon should stay near 0 (self-reference)")
print("- LU should increase (finding better solutions)")
print("="*70)