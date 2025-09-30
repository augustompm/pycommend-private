"""
Test MOVNS Advanced vs MOEA/D with 20 iterations
Middle ground between quick and standard test
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized


print("="*70)
print("MOVNS ADVANCED VS MOEA/D - 20 ITERATIONS")
print("="*70)

package = 'fastapi'

print(f"\n1. Testing MOVNS Advanced (20 iterations)")
print("-"*70)

algo = MOVNS_Advanced(
    package,
    archive_size=100,
    max_iterations=20,
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

best_lu = 0
best_ss = 0
best_rss = float('inf')
for sol in solutions:
    obj = algo.evaluate_objectives(sol['chromosome'])
    if -obj[0] > best_lu:
        best_lu = -obj[0]
    if -obj[1] > best_ss:
        best_ss = -obj[1]
    if obj[2] < best_rss:
        best_rss = obj[2]

print(f"  Best LU: {best_lu:.0f}")
print(f"  Best SS: {best_ss:.4f}")
print(f"  Best RSS: {best_rss}")

print(f"\n2. Testing MOVNS v2 Baseline (20 iterations)")
print("-"*70)

algo = MOVNS_V2(
    package,
    archive_size=100,
    max_iterations=20,
    track_metrics=True
)

start = time.time()
solutions = algo.run()
v2_time = time.time() - start

metrics = algo.get_metrics_history()
v2_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

print(f"\nResults:")
print(f"  Hypervolume: {v2_hv:.4f}")
print(f"  Solutions: {len(solutions)}")
print(f"  Time: {v2_time:.1f}s")

print(f"\n3. Testing MOEA/D (20 generations)")
print("-"*70)

algo = MOEAD_Normalized(
    package,
    pop_size=100,
    max_gen=20
)

start = time.time()
solutions = algo.run()
moead_time = time.time() - start

metrics = algo.get_metrics_history()
moead_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

print(f"\nResults:")
print(f"  Hypervolume: {moead_hv:.4f}")
print(f"  Solutions: {len(solutions)}")
print(f"  Time: {moead_time:.1f}s")

best_lu = 0
best_ss = 0
best_rss = float('inf')
for sol in solutions:
    obj = algo.evaluate_objectives(sol['chromosome'])
    if -obj[0] > best_lu:
        best_lu = -obj[0]
    if -obj[1] > best_ss:
        best_ss = -obj[1]
    if obj[2] < best_rss:
        best_rss = obj[2]

print(f"  Best LU: {best_lu:.0f}")
print(f"  Best SS: {best_ss:.4f}")
print(f"  Best RSS: {best_rss}")

print(f"\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

print(f"\nHypervolume:")
print(f"  MOVNS Advanced: {movns_hv:.4f}")
print(f"  MOVNS v2:       {v2_hv:.4f}")
print(f"  MOEA/D:         {moead_hv:.4f}")

if moead_hv > 0:
    adv_ratio = movns_hv / moead_hv
    v2_ratio = v2_hv / moead_hv

    print(f"\nRelative to MOEA/D:")
    print(f"  MOVNS Advanced: {adv_ratio*100:.1f}%")
    print(f"  MOVNS v2:       {v2_ratio*100:.1f}%")

    if adv_ratio > 1.0:
        print(f"\nSUCCESS: MOVNS Advanced beats MOEA/D by {(adv_ratio-1)*100:.1f}%")
    else:
        print(f"\nMOEA/D still superior by {(1-adv_ratio)*100:.1f}%")

if v2_hv > 0:
    improvement = movns_hv / v2_hv
    print(f"\nMOVNS Advanced vs v2: {improvement*100:.1f}% ({(improvement-1)*100:+.1f}%)")

print(f"\nExecution Time:")
print(f"  MOVNS Advanced: {movns_time:.1f}s")
print(f"  MOVNS v2:       {v2_time:.1f}s")
print(f"  MOEA/D:         {moead_time:.1f}s")