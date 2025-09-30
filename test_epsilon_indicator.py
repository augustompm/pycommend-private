"""
Teste do Epsilon-Indicator: MOVNS vs MOEA/D
Métrica mencionada na apresentação
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_final_v2 import MOVNS_Final_V2
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("EPSILON-INDICATOR TEST - MOVNS vs MOEA/D")
print("="*70)

iterations = 20

# 1. MOVNS Final V2
print("\n1. Running MOVNS Final V2...")
print("-"*70)

movns = MOVNS_Final_V2('fastapi', archive_size=100, max_iterations=iterations, track_metrics=True)
movns_solutions = movns.run()

movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

print(f"MOVNS Solutions: {len(movns_solutions)}")

# 2. MOEA/D
print("\n2. Running MOEA/D Normalized...")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)
moead_solutions = moead.run()

moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

print(f"MOEA/D Solutions: {len(moead_solutions)}")

# 3. Calculate Epsilon-Indicator
print("\n3. Calculating Epsilon-Indicator...")
print("-"*70)

qm = QualityMetrics()

# Use MOEA/D as reference set for MOVNS
movns_epsilon = qm.epsilon_indicator(movns_objectives, moead_objectives)
print(f"MOVNS epsilon-indicator (using MOEA/D as reference): {movns_epsilon:.4f}")

# Use MOVNS as reference set for MOEA/D
qm2 = QualityMetrics()
moead_epsilon = qm2.epsilon_indicator(moead_objectives, movns_objectives)
print(f"MOEA/D epsilon-indicator (using MOVNS as reference): {moead_epsilon:.4f}")

# Binary epsilon (comparing both ways)
print("\n4. Cross-comparison (lower is better):")
print("-"*70)

# Create combined reference set (union of both fronts)
combined = np.vstack([movns_objectives, moead_objectives])

# Remove dominated solutions from combined set
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
reference_set = np.array(non_dominated)

print(f"Reference Pareto front size: {len(reference_set)}")

# Calculate epsilon for each algorithm against reference
qm3 = QualityMetrics()
movns_eps_ref = qm3.epsilon_indicator(movns_objectives, reference_set)
moead_eps_ref = qm3.epsilon_indicator(moead_objectives, reference_set)

print(f"\nMOVNS epsilon-indicator (vs reference): {movns_eps_ref:.4f}")
print(f"MOEA/D epsilon-indicator (vs reference): {moead_eps_ref:.4f}")

# 5. Results Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print("\nEpsilon-Indicator (lower is better):")
print(f"  MOVNS:  {movns_eps_ref:.4f}")
print(f"  MOEA/D: {moead_eps_ref:.4f}")

if movns_eps_ref < moead_eps_ref:
    print("\n  WINNER: MOVNS (better convergence)")
    improvement = ((moead_eps_ref - movns_eps_ref) / moead_eps_ref) * 100
    print(f"  MOVNS is {improvement:.1f}% better")
else:
    print("\n  WINNER: MOEA/D (better convergence)")
    improvement = ((movns_eps_ref - moead_eps_ref) / movns_eps_ref) * 100
    print(f"  MOEA/D is {improvement:.1f}% better")

# Additional metrics for comparison
print("\n6. Complete Metrics Comparison:")
print("-"*70)

# HV
movns_hv = qm.hypervolume(movns_objectives, ref_point=[0, 0, 15])
moead_hv = qm.hypervolume(moead_objectives, ref_point=[0, 0, 15])

# Spacing
movns_spacing = qm.spacing(movns_objectives)
moead_spacing = qm.spacing(moead_objectives)

print(f"Hypervolume:      MOVNS={movns_hv:.4f}, MOEA/D={moead_hv:.4f}")
print(f"Spacing:          MOVNS={movns_spacing:.4f}, MOEA/D={moead_spacing:.4f}")
print(f"Epsilon-indicator: MOVNS={movns_eps_ref:.4f}, MOEA/D={moead_eps_ref:.4f}")

print("\n" + "="*70)