"""
Compare MOVNS Advanced vs MOEA/D with proper HV calculation
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

def calculate_hv_for_algorithm(algo, solutions):
    """Calculate hypervolume for a set of solutions"""
    if not solutions:
        return 0.0

    objectives = []
    for sol in solutions:
        if isinstance(sol, dict) and 'chromosome' in sol:
            obj = algo.evaluate_objectives(sol['chromosome'])
        else:
            obj = algo.evaluate_objectives(sol)
        objectives.append(obj)

    objectives = np.array(objectives)

    # Normalize objectives
    obj_normalized = []
    for obj in objectives:
        norm_obj = algo.normalize_objectives(obj)
        obj_normalized.append(norm_obj)

    obj_normalized = np.array(obj_normalized)

    # Calculate HV with proper reference point
    qm = QualityMetrics()
    ref_point = np.array([0, 0, 1.0])  # For normalized objectives
    hv = qm.hypervolume(obj_normalized, ref_point)

    return hv

print("="*70)
print("MOVNS ADVANCED VS MOEA/D - HV COMPARISON")
print("="*70)

package = 'fastapi'
iterations = 15

# Test MOVNS Advanced
print(f"\n1. MOVNS Advanced ({iterations} iterations)")
print("-"*70)

movns = MOVNS_Advanced(
    package,
    archive_size=100,
    max_iterations=iterations,
    track_metrics=True
)

start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

# Calculate HV properly
movns_hv = calculate_hv_for_algorithm(movns, movns_solutions)

# Get best objectives
best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    if -obj[0] > best_movns['lu']:
        best_movns['lu'] = -obj[0]
    if -obj[1] > best_movns['ss']:
        best_movns['ss'] = -obj[1]
    if obj[2] < best_movns['rss']:
        best_movns['rss'] = obj[2]

print(f"\nResults:")
print(f"  Solutions: {len(movns_solutions)}")
print(f"  Hypervolume: {movns_hv:.4f}")
print(f"  Time: {movns_time:.1f}s")
print(f"  Best LU: {best_movns['lu']:.0f}")
print(f"  Best SS: {best_movns['ss']:.4f}")
print(f"  Best RSS: {best_movns['rss']}")

# Test MOEA/D with same iterations
print(f"\n2. MOEA/D ({iterations} generations)")
print("-"*70)

moead = MOEAD_Normalized(
    package,
    pop_size=100,
    max_gen=iterations
)

start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

# Calculate HV properly
moead_hv = calculate_hv_for_algorithm(moead, moead_solutions)

# Get best objectives
best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    if -obj[0] > best_moead['lu']:
        best_moead['lu'] = -obj[0]
    if -obj[1] > best_moead['ss']:
        best_moead['ss'] = -obj[1]
    if obj[2] < best_moead['rss']:
        best_moead['rss'] = obj[2]

print(f"\nResults:")
print(f"  Solutions: {len(moead_solutions)}")
print(f"  Hypervolume: {moead_hv:.4f}")
print(f"  Time: {moead_time:.1f}s")
print(f"  Best LU: {best_moead['lu']:.0f}")
print(f"  Best SS: {best_moead['ss']:.4f}")
print(f"  Best RSS: {best_moead['rss']}")

# Final comparison
print(f"\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

print(f"\nHypervolume:")
print(f"  MOVNS Advanced: {movns_hv:.4f}")
print(f"  MOEA/D:         {moead_hv:.4f}")

if moead_hv > 0:
    ratio = movns_hv / moead_hv
    print(f"\n  Ratio: {ratio*100:.1f}%")

    if ratio > 1.0:
        print(f"\n✓ SUCCESS: MOVNS Advanced BEATS MOEA/D by {(ratio-1)*100:.1f}%!")
        print(f"  With only {iterations} iterations, MOVNS Advanced achieved superior performance")
    elif ratio > 0.95:
        print(f"\n○ COMPETITIVE: MOVNS Advanced at {ratio*100:.1f}% of MOEA/D")
    else:
        print(f"\n✗ MOEA/D still superior by {(1-ratio)*100:.1f}%")
else:
    if movns_hv > 0:
        print(f"\n✓ MOVNS Advanced has HV={movns_hv:.4f} while MOEA/D has HV=0")
        print(f"  MOVNS Advanced clearly superior!")

print(f"\nBest Objectives:")
print(f"  Linked Usage: MOVNS={best_movns['lu']:.0f}, MOEA/D={best_moead['lu']:.0f}")
print(f"  Semantic Sim: MOVNS={best_movns['ss']:.4f}, MOEA/D={best_moead['ss']:.4f}")
print(f"  Set Size:     MOVNS={best_movns['rss']}, MOEA/D={best_moead['rss']}")

print(f"\nExecution Time:")
print(f"  MOVNS: {movns_time:.1f}s")
print(f"  MOEA/D: {moead_time:.1f}s")
print(f"  Speed ratio: {moead_time/movns_time:.1f}x")