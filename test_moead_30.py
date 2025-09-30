"""
Test MOEA/D with 30 individuals - check metrics evolution
Verify that HV actually changes (not static)
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D v18 with 30 individuals - Metrics Evolution")
print("="*70)

# Create MOEA/D with 30 individuals
package_name = 'fastapi'
moead = MOEAD_V18(
    package_name,
    pop_size=30,       # Reduced to 30
    n_neighbors=5,     # Smaller neighborhood
    max_gen=10,        # Just 10 generations for testing
    theta=2.0,         # Even more reduced
    track_metrics=False
)

print(f"\nConfiguration:")
print(f"  Package: {package_name}")
print(f"  Population: 30")
print(f"  Neighbors: 5")
print(f"  Max generations: 10")
print(f"  Theta: 2.0")

# Get initial population metrics
print("\n" + "-"*70)
print("Gen | Pop | HV (norm) | Spacing | Best LU | Avg LU")
print("-"*70)

for gen in range(10):
    # Run one generation
    for i in range(moead.pop_size):
        # Get neighbors
        neighbors = moead.B[i]

        # Select parents from neighborhood
        if np.random.rand() < 0.9:  # Delta parameter
            parents_idx = np.random.choice(neighbors, 2, replace=False)
        else:
            parents_idx = np.random.choice(moead.pop_size, 2, replace=False)

        parent1 = moead.population[parents_idx[0]]
        parent2 = moead.population[parents_idx[1]]

        # Generate offspring
        offspring = moead.crossover(parent1['chromosome'], parent2['chromosome'])
        offspring = moead.mutation(offspring)

        # Evaluate
        off_obj = moead.evaluate_objectives(offspring)

        # Update reference point
        moead.z = np.minimum(moead.z, off_obj)
        moead.nadir = np.maximum(moead.nadir, off_obj)

        # Update neighbors
        updated = 0
        for j in neighbors:
            if updated >= moead.theta:
                break

            # Calculate fitness using decompose method
            current_fitness = moead.decompose(moead.population[j]['objectives'], moead.weights[j])
            offspring_fitness = moead.decompose(off_obj, moead.weights[j])

            if offspring_fitness < current_fitness:
                moead.population[j] = {
                    'chromosome': offspring.copy(),
                    'objectives': off_obj.copy()
                }
                updated += 1

    # Calculate metrics after this generation
    objectives = np.array([ind['objectives'] for ind in moead.population])

    # Metrics
    qm = QualityMetrics()

    # Normalized HV
    norm_obj = objectives.copy()
    norm_obj[:, 0] = (norm_obj[:, 0] + 20000) / 20000  # LU
    norm_obj[:, 1] = (norm_obj[:, 1] + 1) / 1          # SS
    norm_obj[:, 2] = norm_obj[:, 2] / 15                # RSS

    hv = qm.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])

    # Spacing
    spacing = qm.spacing(objectives)

    # Best and average LU
    best_lu = np.max(-objectives[:, 0])
    avg_lu = np.mean(-objectives[:, 0])

    print(f"{gen+1:3} | {len(moead.population):3} | {hv:9.4f} | {spacing:7.4f} | {best_lu:7.0f} | {avg_lu:7.0f}")

print("-"*70)

# Final analysis
objectives = np.array([ind['objectives'] for ind in moead.population])
qm = QualityMetrics()

# Get non-dominated solutions
pareto_front = qm.get_non_dominated_set(objectives)

print(f"\nFinal Analysis:")
print(f"  Population: {len(moead.population)}")
print(f"  Non-dominated: {len(pareto_front)}")
print(f"  Pareto ratio: {len(pareto_front)/len(moead.population)*100:.1f}%")

# Show objective ranges
print(f"\nObjective ranges:")
print(f"  LU: [{np.min(objectives[:, 0]):.0f}, {np.max(objectives[:, 0]):.0f}]")
print(f"  SS: [{np.min(objectives[:, 1]):.3f}, {np.max(objectives[:, 1]):.3f}]")
print(f"  RSS: [{np.min(objectives[:, 2]):.0f}, {np.max(objectives[:, 2]):.0f}]")

print("\n" + "="*70)
print("Expected: HV should increase, spacing should improve")
print("If HV is static, there's a bug in evolution or calculation")
print("="*70)