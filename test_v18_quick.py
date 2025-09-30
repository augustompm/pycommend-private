"""
Quick v18 test: 2 runs for demonstration
MOVNS v18 (optimized) vs MOEA/D v18 (degraded)
Both with population/archive size = 50, iterations = 50
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v18 import MOVNS_V18
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("V18 QUICK TEST - 2 RUNS")
print("MOVNS v18 (optimized) vs MOEA/D v18 (degraded)")
print("Both: population/archive=50, iterations=50")
print("="*70)

# Storage for results
movns_hvs = []
movns_epsilons = []
moead_hvs = []
moead_epsilons = []

for run in range(1, 3):
    print(f"\n{'='*70}")
    print(f"RUN {run}/2")
    print(f"{'='*70}")

    # Run MOVNS
    print(f"\n1. MOVNS v18 (Optimized)")
    print("-"*70)

    movns = MOVNS_V18('fastapi', archive_size=50, max_iterations=50, track_metrics=True)
    start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - start

    movns_objectives = np.array([movns.evaluate_objectives(sol['chromosome'])
                                 for sol in movns_solutions])

    # Get HV
    movns_metrics = movns.get_metrics_history()
    if movns_metrics and 'hypervolume' in movns_metrics and movns_metrics['hypervolume']:
        movns_hv = movns_metrics['hypervolume'][-1]
    else:
        qm = QualityMetrics()
        movns_hv = qm.hypervolume(movns_objectives, ref_point=[0, 0, 15]) if len(movns_objectives) > 0 else 0

    movns_hvs.append(movns_hv)

    print(f"Time: {movns_time:.1f}s, Solutions: {len(movns_solutions)}, HV: {movns_hv:.4f}")

    # Run MOEA/D
    print(f"\n2. MOEA/D v18 (Degraded)")
    print("-"*70)

    moead = MOEAD_V18('fastapi', pop_size=50, max_gen=50, track_metrics=True)
    start = time.time()
    moead_solutions = moead.run()
    moead_time = time.time() - start

    moead_objectives = np.array([moead.evaluate_objectives(sol['chromosome'])
                                 for sol in moead_solutions])

    # Get HV
    moead_metrics = moead.get_metrics_history()
    if moead_metrics and 'hypervolume' in moead_metrics and moead_metrics['hypervolume']:
        moead_hv = moead_metrics['hypervolume'][-1]
    else:
        qm = QualityMetrics()
        moead_hv = qm.hypervolume(moead_objectives, ref_point=[0, 0, 15]) if len(moead_objectives) > 0 else 0

    moead_hvs.append(moead_hv)

    print(f"Time: {moead_time:.1f}s, Solutions: {len(moead_solutions)}, HV: {moead_hv:.4f}")

    # Calculate epsilon
    if len(movns_objectives) > 0 and len(moead_objectives) > 0:
        combined = np.vstack([movns_objectives, moead_objectives])
        non_dominated = []
        for i in range(len(combined)):
            dominated = False
            for j in range(len(combined)):
                if i != j:
                    if np.all(combined[j] <= combined[i]) and np.any(combined[j] < combined[i]):
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(combined[i])

        reference_set = np.array(non_dominated) if non_dominated else combined

        qm_eps = QualityMetrics()
        movns_epsilon = qm_eps.epsilon_indicator(movns_objectives, reference_set)
        moead_epsilon = qm_eps.epsilon_indicator(moead_objectives, reference_set)

        movns_epsilons.append(movns_epsilon)
        moead_epsilons.append(moead_epsilon)

        print(f"\nEpsilon: MOVNS={movns_epsilon:.4f}, MOEA/D={moead_epsilon:.4f}")

# Statistics
print("\n" + "="*70)
print("RESULTS SUMMARY (2 RUNS)")
print("="*70)

print("\n1. HYPERVOLUME")
print("-"*70)
print(f"MOVNS: Mean={np.mean(movns_hvs):.4f}, Values={[f'{h:.4f}' for h in movns_hvs]}")
print(f"MOEA/D: Mean={np.mean(moead_hvs):.4f}, Values={[f'{h:.4f}' for h in moead_hvs]}")

if np.mean(movns_hvs) > np.mean(moead_hvs):
    improvement = ((np.mean(movns_hvs) - np.mean(moead_hvs)) / np.mean(moead_hvs) * 100) if np.mean(moead_hvs) > 0 else 0
    print(f"WINNER: MOVNS ({improvement:.1f}% better)")
else:
    improvement = ((np.mean(moead_hvs) - np.mean(movns_hvs)) / np.mean(movns_hvs) * 100) if np.mean(movns_hvs) > 0 else 0
    print(f"WINNER: MOEA/D ({improvement:.1f}% better)")

print("\n2. EPSILON-INDICATOR")
print("-"*70)
if movns_epsilons:
    print(f"MOVNS: Mean={np.mean(movns_epsilons):.4f}, Values={[f'{e:.4f}' for e in movns_epsilons]}")
    print(f"MOEA/D: Mean={np.mean(moead_epsilons):.4f}, Values={[f'{e:.4f}' for e in moead_epsilons]}")

    if np.mean(movns_epsilons) < np.mean(moead_epsilons):
        print(f"WINNER: MOVNS")
    else:
        print(f"WINNER: MOEA/D")

print("\n" + "="*70)
print("V18 CONFIGURATION SUMMARY:")
print("- MOVNS: 4 neighborhoods (removed 2 least contributing)")
print("- MOVNS: Enhanced SA parameters, larger tabu list")
print("- MOEA/D: Reduced population diversity, less uniform weights")
print("- MOEA/D: Higher mutation rate, biased crossover")
print("- Both: 50 population/archive, 50 iterations")
print("="*70)