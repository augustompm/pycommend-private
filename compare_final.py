"""
Final comparison: MOVNS vs MOEA/D with correct HV calculation
Both with practical settings for fast package recommendation
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("FINAL COMPARISON: MOVNS vs MOEA/D (30 individuals, 10 iterations)")
print("="*70)

def normalize_objectives(objectives):
    """Correct normalization for HV calculation"""
    norm = objectives.copy()
    norm[:, 0] = -norm[:, 0] / 20000  # LU: higher is better
    norm[:, 1] = -norm[:, 1]  # SS: higher is better
    norm[:, 2] = (20 - norm[:, 2]) / 20  # RSS: lower is better (inverted)
    return norm

# Test MOVNS
print("\n" + "="*70)
print("MOVNS - Variable Neighborhood Search")
print("="*70)

start = time.time()
movns = MOVNS_V22('fastapi', archive_size=30, max_iterations=10, track_metrics=False)

# Initialize
if len(movns.archive) == 0:
    initial = movns.smart_initialization(exploration_rate=0.3)[0]
    initial_obj = movns.evaluate_objectives(initial)
    movns.update_archive(initial, initial_obj)

print("Iter | Archive | HV | Spacing | Best LU")
print("-"*60)

for iteration in range(10):
    # VNS iteration
    if len(movns.archive) > 0:
        current_idx = np.random.randint(len(movns.archive))
        current = movns.archive[current_idx]['chromosome'].copy()
        current_obj = movns.archive[current_idx]['objectives'].copy()

    k = 0
    while k < movns.k_max:
        neighbor = movns.get_neighborhood(current, k)
        neighbor_obj = movns.evaluate_objectives(neighbor)

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

    # Metrics
    if len(movns.archive) > 0:
        objectives = np.array([sol['objectives'] for sol in movns.archive])
        qm = QualityMetrics()
        norm = normalize_objectives(objectives)
        hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
        spacing = qm.spacing(objectives)
        best_lu = np.max(-objectives[:, 0])
        print(f"{iteration+1:4} | {len(movns.archive):7} | {hv:.4f} | {spacing:.4f} | {best_lu:7.0f}")

movns_time = time.time() - start
movns_final_objectives = np.array([sol['objectives'] for sol in movns.archive])
movns_final_hv = hv
movns_final_spacing = spacing
movns_final_lu = best_lu

# Test MOEA/D
print("\n" + "="*70)
print("MOEA/D - Decomposition with 30 individuals")
print("="*70)

start = time.time()
moead = MOEAD_V18('fastapi', pop_size=30, n_neighbors=5, max_gen=10, theta=1.0, track_metrics=False)

print("Gen | Pop | HV | Spacing | Best LU")
print("-"*60)

for gen in range(10):
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

    # Metrics
    objectives = np.array([ind['objectives'] for ind in moead.population])
    qm = QualityMetrics()
    norm = normalize_objectives(objectives)
    hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
    spacing = qm.spacing(objectives)
    best_lu = np.max(-objectives[:, 0])
    print(f"{gen+1:4} | {len(moead.population):3} | {hv:.4f} | {spacing:.4f} | {best_lu:7.0f}")

moead_time = time.time() - start
moead_final_objectives = objectives
moead_final_hv = hv
moead_final_spacing = spacing
moead_final_lu = best_lu

# Final comparison
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print("\nPerformance Metrics:")
print("-"*40)
print(f"{'Metric':<20} | {'MOVNS':>10} | {'MOEA/D':>10}")
print("-"*40)
print(f"{'Hypervolume':<20} | {movns_final_hv:10.4f} | {moead_final_hv:10.4f}")
print(f"{'Spacing':<20} | {movns_final_spacing:10.4f} | {moead_final_spacing:10.4f}")
print(f"{'Best LU':<20} | {movns_final_lu:10.0f} | {moead_final_lu:10.0f}")
print(f"{'Archive/Pop size':<20} | {len(movns.archive):10} | {len(moead.population):10}")
print(f"{'Runtime (s)':<20} | {movns_time:10.1f} | {moead_time:10.1f}")
print("-"*40)

# Winner analysis
print("\nWinner Analysis:")
winners = 0
if movns_final_hv > moead_final_hv:
    print(f"[WIN] MOVNS wins HV: {movns_final_hv:.4f} > {moead_final_hv:.4f}")
    winners += 1
else:
    print(f"[WIN] MOEA/D wins HV: {moead_final_hv:.4f} > {movns_final_hv:.4f}")

if movns_final_spacing < moead_final_spacing:
    print(f"[WIN] MOVNS wins Spacing: {movns_final_spacing:.4f} < {moead_final_spacing:.4f}")
    winners += 1
else:
    print(f"[WIN] MOEA/D wins Spacing: {moead_final_spacing:.4f} < {movns_final_spacing:.4f}")

if movns_final_lu > moead_final_lu:
    print(f"[WIN] MOVNS wins Best LU: {movns_final_lu:.0f} > {moead_final_lu:.0f}")
    winners += 1
else:
    print(f"[WIN] MOEA/D wins Best LU: {moead_final_lu:.0f} > {movns_final_lu:.0f}")

print("\n" + "="*70)
if winners >= 2:
    print(f"WINNER: MOVNS wins {winners}/3 metrics")
else:
    print(f"WINNER: MOEA/D wins {3-winners}/3 metrics")
print("="*70)