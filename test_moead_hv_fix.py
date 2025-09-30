"""
Test MOEA/D HV with CORRECT normalization
Show that HV should INCREASE as solutions improve
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D HV Fix - Correct Normalization")
print("="*70)

# Create MOEA/D with 30 individuals
package_name = 'fastapi'
moead = MOEAD_V18(
    package_name,
    pop_size=30,
    n_neighbors=5,
    max_gen=10,
    theta=2.0,
    track_metrics=False
)

print(f"\nConfiguration: 30 individuals, 10 generations")
print("\n" + "-"*70)
print("Gen | HV (correct) | Best LU | Avg LU | Non-dom | Note")
print("-"*70)

for gen in range(10):
    # Run one generation
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

    # CORRECT normalization for maximization objectives
    norm_obj = objectives.copy()

    # LU: negative values, more negative is better
    # Convert to positive where higher is better
    norm_obj[:, 0] = -norm_obj[:, 0] / 20000  # Now 0 to 1, higher is better

    # SS: negative values, more negative is better
    # Convert to positive where higher is better
    norm_obj[:, 1] = -norm_obj[:, 1]  # Now 0 to 1, higher is better

    # RSS: positive values, lower is better
    # Invert so lower RSS gives higher value
    norm_obj[:, 2] = (20 - norm_obj[:, 2]) / 20  # Now higher is better

    # Now all objectives are "higher is better" in [0,1] range
    qm = QualityMetrics()
    hv = qm.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])

    # Calculate other stats
    best_lu = np.max(-objectives[:, 0])
    avg_lu = np.mean(-objectives[:, 0])

    pareto = qm.get_non_dominated_set(objectives)

    note = "Initial" if gen == 0 else ""
    if gen > 0 and gen % 3 == 0:
        note = "Checkpoint"

    print(f"{gen:3} | {hv:12.4f} | {best_lu:7.0f} | {avg_lu:7.1f} | {len(pareto):7} | {note}")

print("-"*70)

print("\n" + "="*70)
print("Expected: HV should generally INCREASE over generations")
print("If HV decreases, population is getting worse")
print("="*70)