"""
Test MOEA/D with stable parameters for 30 individuals
Reduce theta to 1 for less aggressive replacement
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D Stable - 30 individuals with theta=1")
print("="*70)

# Test with different theta values
for theta in [1.0, 2.0]:
    print(f"\n{'='*70}")
    print(f"Testing with theta={theta}")
    print("="*70)

    # Create MOEA/D
    package_name = 'fastapi'
    moead = MOEAD_V18(
        package_name,
        pop_size=30,
        n_neighbors=5,
        max_gen=10,
        theta=theta,  # Less aggressive replacement
        track_metrics=False
    )

    print(f"Configuration: 30 individuals, theta={theta}")
    print("-"*70)
    print("Gen | HV | Non-dom | Best LU | Stability")
    print("-"*70)

    hv_values = []

    for gen in range(10):
        # Run generation
        if gen > 0:
            for i in range(moead.pop_size):
                neighbors = moead.B[i]

                if np.random.rand() < 0.9:
                    parents_idx = np.random.choice(neighbors, 2, replace=False)
                else:
                    parents_idx = np.random.choice(moead.pop_size, 2, replace=False)

                parent1 = moead.population[parents_idx[0]]
                parent2 = moead.population[parents_idx[1]]

                offspring = moead.crossover(parent1['chromosome'], parent2['chromosome'])
                offspring = moead.mutation(offspring)
                off_obj = moead.evaluate_objectives(offspring)

                moead.z = np.minimum(moead.z, off_obj)
                moead.nadir = np.maximum(moead.nadir, off_obj)

                updated = 0
                for j in neighbors:
                    if updated >= moead.theta:
                        break

                    current_fitness = moead.decompose(moead.population[j]['objectives'], moead.weights[j])
                    offspring_fitness = moead.decompose(off_obj, moead.weights[j])

                    if offspring_fitness < current_fitness:
                        moead.population[j] = {
                            'chromosome': offspring.copy(),
                            'objectives': off_obj.copy()
                        }
                        updated += 1

        # Calculate metrics
        objectives = np.array([ind['objectives'] for ind in moead.population])

        # Correct normalization
        norm_obj = objectives.copy()
        norm_obj[:, 0] = -norm_obj[:, 0] / 20000
        norm_obj[:, 1] = -norm_obj[:, 1]
        norm_obj[:, 2] = (20 - norm_obj[:, 2]) / 20

        qm = QualityMetrics()
        hv = qm.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])
        hv_values.append(hv)

        pareto = qm.get_non_dominated_set(objectives)
        best_lu = np.max(-objectives[:, 0])

        # Check stability
        stability = "Stable"
        if gen > 0:
            change = abs(hv - hv_values[-2]) / hv_values[-2] * 100
            if change > 20:
                stability = f"Volatile ({change:.0f}%)"
            elif change > 10:
                stability = f"Unstable ({change:.0f}%)"

        print(f"{gen:3} | {hv:.4f} | {len(pareto):7} | {best_lu:7.0f} | {stability}")

    print("-"*70)

    # Calculate overall stability
    hv_std = np.std(hv_values)
    hv_mean = np.mean(hv_values)
    cv = hv_std / hv_mean * 100

    print(f"\nStability Analysis:")
    print(f"  HV mean: {hv_mean:.4f}")
    print(f"  HV std: {hv_std:.4f}")
    print(f"  Coefficient of variation: {cv:.1f}%")
    print(f"  Verdict: {'Stable' if cv < 20 else 'Unstable' if cv < 40 else 'Very unstable'}")

print("\n" + "="*70)
print("Conclusion: theta=1 should be more stable than theta=2")
print("With 30 individuals, less aggressive replacement helps stability")
print("="*70)