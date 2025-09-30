"""
Debug MOEA/D HV calculation - generation by generation
Check why HV reaches 0.37 and oscillates
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D Debug - HV Calculation Step by Step")
print("="*70)

# Create MOEA/D with 30 individuals
package_name = 'fastapi'
moead = MOEAD_V18(
    package_name,
    pop_size=30,
    n_neighbors=5,
    max_gen=3,  # Just 3 generations for detailed debug
    theta=2.0,
    track_metrics=False
)

print(f"\nInitial population created with 30 individuals")

# Analyze generation 0
objectives = np.array([ind['objectives'] for ind in moead.population])
print(f"\nGeneration 0 - Initial Population:")
print(f"  Raw objectives shape: {objectives.shape}")
print(f"  First 5 solutions:")
for i in range(min(5, len(objectives))):
    print(f"    Sol {i+1}: LU={objectives[i][0]:.0f}, SS={objectives[i][1]:.3f}, RSS={objectives[i][2]:.0f}")

# Show raw objective ranges
print(f"\n  Raw objective ranges:")
print(f"    LU: [{np.min(objectives[:,0]):.0f}, {np.max(objectives[:,0]):.0f}]")
print(f"    SS: [{np.min(objectives[:,1]):.3f}, {np.max(objectives[:,1]):.3f}]")
print(f"    RSS: [{np.min(objectives[:,2]):.0f}, {np.max(objectives[:,2]):.0f}]")

# Calculate HV with different methods
qm = QualityMetrics()

# Method 1: Direct with raw objectives
print(f"\n  Method 1 - Direct HV (raw objectives):")
try:
    hv1 = qm.hypervolume(objectives, ref_point=[0, 0, 20])
    print(f"    HV with ref=[0,0,20]: {hv1:.4f}")
except Exception as e:
    print(f"    Error: {e}")

# Method 2: Normalized to [0,1]
print(f"\n  Method 2 - Normalized HV:")
norm_obj = objectives.copy()
# Normalize each objective to [0,1]
# LU: typically -20000 to 0, make positive first
norm_obj[:, 0] = (-norm_obj[:, 0]) / 20000  # Now 0 to 1, higher is better
# SS: typically -1 to 0, make positive first
norm_obj[:, 1] = (-norm_obj[:, 1]) / 1  # Now 0 to 1, higher is better
# RSS: typically 2 to 15, minimize
norm_obj[:, 2] = (norm_obj[:, 2] - 2) / 13  # Now 0 to 1, lower is better

print(f"    Normalized objectives (first 5):")
for i in range(min(5, len(norm_obj))):
    print(f"      Sol {i+1}: LU={norm_obj[i][0]:.4f}, SS={norm_obj[i][1]:.4f}, RSS={norm_obj[i][2]:.4f}")

qm2 = QualityMetrics()
hv2 = qm2.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])
print(f"    HV with ref=[1.1,1.1,1.1]: {hv2:.4f}")

# Method 3: As in test file (different normalization)
print(f"\n  Method 3 - Test file normalization:")
norm_obj3 = objectives.copy()
norm_obj3[:, 0] = (norm_obj3[:, 0] + 20000) / 20000  # LU
norm_obj3[:, 1] = (norm_obj3[:, 1] + 1) / 1          # SS
norm_obj3[:, 2] = norm_obj3[:, 2] / 15                # RSS

print(f"    Normalized objectives (first 5):")
for i in range(min(5, len(norm_obj3))):
    print(f"      Sol {i+1}: LU={norm_obj3[i][0]:.4f}, SS={norm_obj3[i][1]:.4f}, RSS={norm_obj3[i][2]:.4f}")

qm3 = QualityMetrics()
hv3 = qm3.hypervolume(norm_obj3, ref_point=[1.1, 1.1, 1.1])
print(f"    HV with ref=[1.1,1.1,1.1]: {hv3:.4f}")

# Now run 2 generations and check evolution
print("\n" + "="*70)
print("Running 2 generations to check HV evolution...")
print("="*70)

for gen in range(2):
    print(f"\n--- Generation {gen+1} ---")

    # Run one generation
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

    # Analyze after generation
    objectives = np.array([ind['objectives'] for ind in moead.population])

    print(f"  Objective ranges after gen {gen+1}:")
    print(f"    LU: [{np.min(objectives[:,0]):.0f}, {np.max(objectives[:,0]):.0f}]")
    print(f"    SS: [{np.min(objectives[:,1]):.3f}, {np.max(objectives[:,1]):.3f}]")
    print(f"    RSS: [{np.min(objectives[:,2]):.0f}, {np.max(objectives[:,2]):.0f}]")

    # Calculate HV
    norm_obj = objectives.copy()
    norm_obj[:, 0] = (norm_obj[:, 0] + 20000) / 20000
    norm_obj[:, 1] = (norm_obj[:, 1] + 1) / 1
    norm_obj[:, 2] = norm_obj[:, 2] / 15

    qm_gen = QualityMetrics()
    hv = qm_gen.hypervolume(norm_obj, ref_point=[1.1, 1.1, 1.1])

    # Check non-dominated solutions
    pareto = qm_gen.get_non_dominated_set(objectives)

    print(f"  HV: {hv:.4f}")
    print(f"  Non-dominated: {len(pareto)} / {len(objectives)}")
    print(f"  Best LU: {np.max(-objectives[:,0]):.0f}")

print("\n" + "="*70)
print("Analysis: Check if HV calculation makes sense")
print("If HV = 0.37, what volume is being measured?")
print("="*70)