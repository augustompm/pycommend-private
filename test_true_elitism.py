"""
The real answer: MOEA/D is NOT elitist by design!
It uses decomposition, not Pareto dominance for selection.
Let's prove this and show the difference.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOEA/D is NOT Elitist - It's Decomposition-Based")
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

print("\nKey insight: MOEA/D replaces based on scalar fitness, not dominance")
print("This means it can lose good solutions if they don't fit weight vectors")
print("-"*70)

# Track what happens to a good solution
print("\nTracking solution replacement:")
print("Gen | Action | Old Fitness | New Fitness | Old LU | New LU")
print("-"*70)

replacements = []

for gen in range(5):
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

                old_obj = moead.population[j]['objectives']
                current_fitness = moead.decompose(old_obj, moead.weights[j])
                offspring_fitness = moead.decompose(off_obj, moead.weights[j])

                if offspring_fitness < current_fitness:
                    # Track replacements where we might lose good solutions
                    old_lu = -old_obj[0]
                    new_lu = -off_obj[0]

                    # Check if we're replacing a better solution (in at least one objective)
                    if old_lu > new_lu and len(replacements) < 5:
                        replacements.append({
                            'gen': gen,
                            'old_fitness': current_fitness,
                            'new_fitness': offspring_fitness,
                            'old_lu': old_lu,
                            'new_lu': new_lu,
                            'old_obj': old_obj,
                            'new_obj': off_obj
                        })
                        print(f"{gen:3} | Replace | {current_fitness:11.2f} | {offspring_fitness:11.2f} | {old_lu:7.0f} | {new_lu:7.0f}")

                    moead.population[j] = {
                        'chromosome': offspring.copy(),
                        'objectives': off_obj.copy()
                    }
                    updated += 1

print("-"*70)

if len(replacements) > 0:
    print("\nAnalysis of replacements:")
    for r in replacements[:3]:
        print(f"\nGen {r['gen']}: Replaced solution with LU={r['old_lu']:.0f} by LU={r['new_lu']:.0f}")
        print(f"  Old objectives: LU={-r['old_obj'][0]:.0f}, SS={-r['old_obj'][1]:.3f}, RSS={r['old_obj'][2]:.0f}")
        print(f"  New objectives: LU={-r['new_obj'][0]:.0f}, SS={-r['new_obj'][1]:.3f}, RSS={r['new_obj'][2]:.0f}")
        print(f"  Scalar fitness improved: {r['old_fitness']:.2f} -> {r['new_fitness']:.2f}")
        print(f"  But we lost a better LU value!")

print("\n" + "="*70)
print("Conclusion:")
print("1. MOEA/D is NOT elitist - it uses decomposition")
print("2. It can replace better solutions if they don't fit weight vectors")
print("3. This is why population HV can decrease")
print("4. The algorithm trades Pareto optimality for better distribution")
print("5. This is by design - Zhang & Li (2007) IEEE TEVC")
print("="*70)