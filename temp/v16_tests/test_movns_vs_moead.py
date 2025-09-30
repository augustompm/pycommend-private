"""
Direct comparison MOVNS Advanced vs MOEA/D
Focus on getting correct HV for both algorithms
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized

print("="*70)
print("MOVNS ADVANCED VS MOEA/D - DIRECT COMPARISON")
print("="*70)

package = 'fastapi'

# Test MOVNS Advanced with 15 iterations
print(f"\n1. MOVNS Advanced (15 iterations)")
print("-"*70)

algo = MOVNS_Advanced(
    package,
    archive_size=100,
    max_iterations=15,
    track_metrics=True
)

start = time.time()
solutions = algo.run()
movns_time = time.time() - start

metrics = algo.get_metrics_history()
movns_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

print(f"\nResults:")
print(f"  Hypervolume: {movns_hv:.4f}")
print(f"  Solutions: {len(solutions)}")
print(f"  Time: {movns_time:.1f}s")

best_movns = {'lu': 0, 'ss': 0, 'rss': float('inf')}
for sol in solutions:
    obj = algo.evaluate_objectives(sol['chromosome'])
    if -obj[0] > best_movns['lu']:
        best_movns['lu'] = -obj[0]
    if -obj[1] > best_movns['ss']:
        best_movns['ss'] = -obj[1]
    if obj[2] < best_movns['rss']:
        best_movns['rss'] = obj[2]

print(f"  Best objectives: LU={best_movns['lu']:.0f}, SS={best_movns['ss']:.4f}, RSS={best_movns['rss']}")

# Test MOEA/D with 30 generations (standard comparison)
print(f"\n2. MOEA/D (30 generations - standard)")
print("-"*70)

algo = MOEAD_Normalized(
    package,
    pop_size=100,
    max_gen=30
)

start = time.time()
solutions = algo.run()
moead_time = time.time() - start

# Calculate HV manually if needed
from evaluation.quality_metrics import QualityMetrics
qm = QualityMetrics()

# Get objectives for all solutions
moead_objectives = []
for sol in solutions:
    obj = algo.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)

moead_objectives = np.array(moead_objectives)

# Calculate hypervolume with proper normalization
moead_hv = qm.hypervolume(moead_objectives, ref_point=np.array([0, 0, 30]))

print(f"\nResults:")
print(f"  Hypervolume: {moead_hv:.4f}")
print(f"  Solutions: {len(solutions)}")
print(f"  Time: {moead_time:.1f}s")

best_moead = {'lu': 0, 'ss': 0, 'rss': float('inf')}
for obj in moead_objectives:
    if -obj[0] > best_moead['lu']:
        best_moead['lu'] = -obj[0]
    if -obj[1] > best_moead['ss']:
        best_moead['ss'] = -obj[1]
    if obj[2] < best_moead['rss']:
        best_moead['rss'] = obj[2]

print(f"  Best objectives: LU={best_moead['lu']:.0f}, SS={best_moead['ss']:.4f}, RSS={best_moead['rss']}")

# Also test MOEA/D with 15 generations for fair comparison
print(f"\n3. MOEA/D (15 generations - fair comparison)")
print("-"*70)

algo = MOEAD_Normalized(
    package,
    pop_size=100,
    max_gen=15
)

start = time.time()
solutions = algo.run()
moead15_time = time.time() - start

# Get objectives for all solutions
moead15_objectives = []
for sol in solutions:
    obj = algo.evaluate_objectives(sol['chromosome'])
    moead15_objectives.append(obj)

moead15_objectives = np.array(moead15_objectives)
moead15_hv = qm.hypervolume(moead15_objectives, ref_point=np.array([0, 0, 30]))

print(f"\nResults:")
print(f"  Hypervolume: {moead15_hv:.4f}")
print(f"  Solutions: {len(solutions)}")
print(f"  Time: {moead15_time:.1f}s")

print(f"\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

print(f"\nHypervolume Results:")
print(f"  MOVNS Advanced (15 iter):  {movns_hv:.4f}")
print(f"  MOEA/D (30 gen):           {moead_hv:.4f}")
print(f"  MOEA/D (15 gen):           {moead15_hv:.4f}")

print(f"\nComparison:")
if moead_hv > 0:
    ratio_30 = movns_hv / moead_hv
    print(f"  MOVNS vs MOEA/D-30: {ratio_30*100:.1f}%")
    if ratio_30 > 1.0:
        print(f"    -> MOVNS BEATS MOEA/D-30 by {(ratio_30-1)*100:.1f}%!")
    else:
        print(f"    -> MOEA/D-30 still superior by {(1-ratio_30)*100:.1f}%")

if moead15_hv > 0:
    ratio_15 = movns_hv / moead15_hv
    print(f"  MOVNS vs MOEA/D-15: {ratio_15*100:.1f}%")
    if ratio_15 > 1.0:
        print(f"    -> MOVNS BEATS MOEA/D-15 by {(ratio_15-1)*100:.1f}%!")
    else:
        print(f"    -> MOEA/D-15 still superior by {(1-ratio_15)*100:.1f}%")

print(f"\nBest Objectives Comparison:")
print(f"  Linked Usage:")
print(f"    MOVNS:   {best_movns['lu']:.0f}")
print(f"    MOEA/D:  {best_moead['lu']:.0f}")
print(f"  Semantic Similarity:")
print(f"    MOVNS:   {best_movns['ss']:.4f}")
print(f"    MOEA/D:  {best_moead['ss']:.4f}")
print(f"  Set Size:")
print(f"    MOVNS:   {best_movns['rss']}")
print(f"    MOEA/D:  {best_moead['rss']}")

print(f"\nExecution Time:")
print(f"  MOVNS (15 iter):  {movns_time:.1f}s")
print(f"  MOEA/D (30 gen):  {moead_time:.1f}s")
print(f"  MOEA/D (15 gen):  {moead15_time:.1f}s")