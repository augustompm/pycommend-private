"""
Check if MOEA/D is truly elitist - it should NEVER lose HV
Track the actual Pareto front, not just the population
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D Elitism Check - HV Should Never Decrease")
print("="*70)

package_name = 'fastapi'
moead = MOEAD_V18(
    package_name,
    pop_size=30,
    n_neighbors=5,
    max_gen=10,
    theta=1.0,
    track_metrics=False
)

print("\nTracking both population AND true Pareto front")
print("-"*70)
print("Gen | Pop HV | Pareto HV | Pop Size | Pareto Size | Status")
print("-"*70)

# Track the global Pareto front (elitist archive)
global_pareto = []

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

    # Get current population objectives
    pop_objectives = np.array([ind['objectives'] for ind in moead.population])

    # Update global Pareto front (true elitism)
    for obj in pop_objectives:
        # Check if this solution should be in Pareto front
        dominated = False
        to_remove = []

        for i, pareto_sol in enumerate(global_pareto):
            # For minimization: solution A dominates B if A <= B in all objectives and A < B in at least one
            if all(pareto_sol <= obj) and any(pareto_sol < obj):  # pareto_sol dominates obj
                dominated = True
                break
            elif all(obj <= pareto_sol) and any(obj < pareto_sol):  # obj dominates pareto_sol
                to_remove.append(i)

        if not dominated:
            # Remove dominated solutions
            for idx in reversed(to_remove):
                global_pareto.pop(idx)
            # Add new solution if not already there
            exists = any(np.array_equal(obj, ps) for ps in global_pareto)
            if not exists:
                global_pareto.append(obj)

    # Calculate HV for population
    qm = QualityMetrics()
    norm_pop = pop_objectives.copy()
    norm_pop[:, 0] = -norm_pop[:, 0] / 20000
    norm_pop[:, 1] = -norm_pop[:, 1]
    norm_pop[:, 2] = (20 - norm_pop[:, 2]) / 20
    pop_hv = qm.hypervolume(norm_pop, ref_point=[1.1, 1.1, 1.1])

    # Calculate HV for global Pareto front
    if len(global_pareto) > 0:
        pareto_array = np.array(global_pareto)
        norm_pareto = pareto_array.copy()
        norm_pareto[:, 0] = -norm_pareto[:, 0] / 20000
        norm_pareto[:, 1] = -norm_pareto[:, 1]
        norm_pareto[:, 2] = (20 - norm_pareto[:, 2]) / 20
        pareto_hv = qm.hypervolume(norm_pareto, ref_point=[1.1, 1.1, 1.1])
    else:
        pareto_hv = 0.0

    # Check status
    status = "OK"
    if gen > 0:
        if pareto_hv < prev_pareto_hv - 0.0001:  # Allow tiny numerical errors
            status = "ERROR: PARETO HV DECREASED!"
        if pop_hv < prev_pop_hv * 0.8:  # Population HV can vary
            status = "Pop HV dropped"

    print(f"{gen:3} | {pop_hv:7.4f} | {pareto_hv:9.4f} | {len(moead.population):8} | {len(global_pareto):11} | {status}")

    prev_pop_hv = pop_hv
    prev_pareto_hv = pareto_hv

print("-"*70)

print("\n" + "="*70)
print("Analysis:")
print("- Population HV can oscillate (decomposition replaces solutions)")
print("- Pareto HV should NEVER decrease (true elitism)")
print("- If Pareto HV decreases, there's a bug in dominance checking")
print("="*70)