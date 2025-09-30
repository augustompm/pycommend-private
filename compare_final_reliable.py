"""
Final reliable comparison: MOVNS vs MOEA/D
Multiple runs to get average metrics
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
print("FINAL COMPARISON: MOVNS vs MOEA/D (5 runs each)")
print("="*70)

def normalize_objectives(objectives):
    """Correct normalization for HV calculation"""
    norm = objectives.copy()
    norm[:, 0] = -norm[:, 0] / 20000  # LU: higher is better
    norm[:, 1] = -norm[:, 1]  # SS: higher is better
    norm[:, 2] = (20 - norm[:, 2]) / 20  # RSS: lower is better (inverted)
    return norm

def run_movns(package_name='fastapi', iterations=10):
    """Run MOVNS and return final metrics"""
    movns = MOVNS_V22(package_name, archive_size=30, max_iterations=iterations, track_metrics=False)

    # Initialize
    if len(movns.archive) == 0:
        initial = movns.smart_initialization(exploration_rate=0.3)[0]
        initial_obj = movns.evaluate_objectives(initial)
        movns.update_archive(initial, initial_obj)

    # Run iterations
    for iteration in range(iterations):
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

    # Calculate final metrics
    objectives = np.array([sol['objectives'] for sol in movns.archive])
    qm = QualityMetrics()
    norm = normalize_objectives(objectives)
    hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
    spacing = qm.spacing(objectives)
    best_lu = np.max(-objectives[:, 0])

    return hv, spacing, best_lu, len(movns.archive)

def run_moead(package_name='fastapi', generations=10):
    """Run MOEA/D and return final metrics"""
    moead = MOEAD_V18(package_name, pop_size=30, n_neighbors=5, max_gen=generations, theta=1.0, track_metrics=False)

    # Run generations
    for gen in range(generations):
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

    # Calculate final metrics
    objectives = np.array([ind['objectives'] for ind in moead.population])
    qm = QualityMetrics()
    norm = normalize_objectives(objectives)
    hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
    spacing = qm.spacing(objectives)
    best_lu = np.max(-objectives[:, 0])

    return hv, spacing, best_lu, len(moead.population)

# Run multiple times
n_runs = 3

print(f"\nRunning {n_runs} runs for each algorithm...")
print("-"*70)

# MOVNS runs
print("\nMOVNS runs:")
movns_results = []
for run in range(n_runs):
    print(f"  Run {run+1}...", end="")
    hv, spacing, best_lu, archive_size = run_movns()
    movns_results.append([hv, spacing, best_lu, archive_size])
    print(f" HV={hv:.4f}, Spacing={spacing:.4f}, LU={best_lu:.0f}")

# MOEA/D runs
print("\nMOEA/D runs:")
moead_results = []
for run in range(n_runs):
    print(f"  Run {run+1}...", end="")
    hv, spacing, best_lu, pop_size = run_moead()
    moead_results.append([hv, spacing, best_lu, pop_size])
    print(f" HV={hv:.4f}, Spacing={spacing:.4f}, LU={best_lu:.0f}")

# Calculate statistics
movns_results = np.array(movns_results)
moead_results = np.array(moead_results)

movns_mean = np.mean(movns_results, axis=0)
movns_std = np.std(movns_results, axis=0)
moead_mean = np.mean(moead_results, axis=0)
moead_std = np.std(moead_results, axis=0)

# Final comparison
print("\n" + "="*70)
print("FINAL RESULTS (mean ± std)")
print("="*70)

print("\nPerformance Metrics:")
print("-"*50)
print(f"{'Metric':<20} | {'MOVNS':>20} | {'MOEA/D':>20}")
print("-"*50)
print(f"{'Hypervolume':<20} | {movns_mean[0]:8.4f} ± {movns_std[0]:.4f} | {moead_mean[0]:8.4f} ± {moead_std[0]:.4f}")
print(f"{'Spacing':<20} | {movns_mean[1]:8.4f} ± {movns_std[1]:.4f} | {moead_mean[1]:8.4f} ± {moead_std[1]:.4f}")
print(f"{'Best LU':<20} | {movns_mean[2]:8.0f} ± {movns_std[2]:.0f} | {moead_mean[2]:8.0f} ± {moead_std[2]:.0f}")
print(f"{'Archive/Pop size':<20} | {movns_mean[3]:8.0f} ± {movns_std[3]:.0f} | {moead_mean[3]:8.0f} ± {moead_std[3]:.0f}")
print("-"*50)

# Statistical comparison
print("\nStatistical Comparison:")
print("-"*50)

# HV comparison
if movns_mean[0] > moead_mean[0]:
    hv_ratio = (movns_mean[0] / moead_mean[0] - 1) * 100
    print(f"HV: MOVNS is {hv_ratio:.1f}% better")
else:
    hv_ratio = (moead_mean[0] / movns_mean[0] - 1) * 100
    print(f"HV: MOEA/D is {hv_ratio:.1f}% better")

# Spacing comparison (lower is better)
if movns_mean[1] < moead_mean[1]:
    spacing_ratio = (moead_mean[1] / movns_mean[1] - 1) * 100
    print(f"Spacing: MOVNS is {spacing_ratio:.1f}% better")
else:
    spacing_ratio = (movns_mean[1] / moead_mean[1] - 1) * 100
    print(f"Spacing: MOEA/D is {spacing_ratio:.1f}% better")

# LU comparison
if movns_mean[2] > moead_mean[2]:
    lu_ratio = (movns_mean[2] / moead_mean[2] - 1) * 100
    print(f"Best LU: MOVNS is {lu_ratio:.1f}% better")
else:
    lu_ratio = (moead_mean[2] / movns_mean[2] - 1) * 100
    print(f"Best LU: MOEA/D is {lu_ratio:.1f}% better")

print("-"*50)

# Winner determination
winners = 0
if movns_mean[0] > moead_mean[0]:
    winners += 1
if movns_mean[1] < moead_mean[1]:
    winners += 1
if movns_mean[2] > moead_mean[2]:
    winners += 1

print("\n" + "="*70)
if winners >= 2:
    print(f"OVERALL WINNER: MOVNS wins {winners}/3 metrics")
else:
    print(f"OVERALL WINNER: MOEA/D wins {3-winners}/3 metrics")
print("="*70)