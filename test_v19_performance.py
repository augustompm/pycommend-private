"""
Test v19: Performance comparison
MOVNS v19 (speed optimized) vs MOVNS v18 vs MOEA/D v18
Focus on speed and quality metrics
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v19 import MOVNS_V19
from optimizer.movns_v18 import MOVNS_V18
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("V19 PERFORMANCE TEST")
print("MOVNS v19 (optimized) vs v18 vs MOEA/D v18")
print("="*70)

# Test parameters
package_name = 'fastapi'
iterations = 30  # Reduced for speed test
archive_size = 50

# 1. Test MOVNS v19 (speed optimized)
print(f"\n1. MOVNS v19 (Speed Optimized)")
print("-"*70)

movns19 = MOVNS_V19(package_name, archive_size=archive_size, max_iterations=iterations, track_metrics=True)
start = time.time()
movns19_solutions = movns19.run()
movns19_time = time.time() - start

movns19_objectives = np.array([movns19.evaluate_objectives(sol['chromosome'])
                               for sol in movns19_solutions])

# Get metrics
movns19_metrics = movns19.get_metrics_history()
if movns19_metrics and 'hypervolume' in movns19_metrics and movns19_metrics['hypervolume']:
    movns19_hv = movns19_metrics['hypervolume'][-1]
else:
    qm = QualityMetrics()
    movns19_hv = qm.hypervolume(movns19_objectives, ref_point=[0, 0, 15]) if len(movns19_objectives) > 0 else 0

print(f"\nResults:")
print(f"  Time: {movns19_time:.2f}s")
print(f"  Solutions: {len(movns19_solutions)}")
print(f"  HV: {movns19_hv:.4f}")
print(f"  Cache hit rate: {movns19.cache_hits/(movns19.cache_hits+movns19.cache_misses)*100:.1f}%")
print(f"  Total evaluations cached: {len(movns19.objective_cache)}")

# 2. Test MOVNS v18 (baseline)
print(f"\n2. MOVNS v18 (Baseline)")
print("-"*70)

movns18 = MOVNS_V18(package_name, archive_size=archive_size, max_iterations=iterations, track_metrics=True)
start = time.time()
movns18_solutions = movns18.run()
movns18_time = time.time() - start

movns18_objectives = np.array([movns18.evaluate_objectives(sol['chromosome'])
                               for sol in movns18_solutions])

# Get metrics
movns18_metrics = movns18.get_metrics_history()
if movns18_metrics and 'hypervolume' in movns18_metrics and movns18_metrics['hypervolume']:
    movns18_hv = movns18_metrics['hypervolume'][-1]
else:
    qm = QualityMetrics()
    movns18_hv = qm.hypervolume(movns18_objectives, ref_point=[0, 0, 15]) if len(movns18_objectives) > 0 else 0

print(f"\nResults:")
print(f"  Time: {movns18_time:.2f}s")
print(f"  Solutions: {len(movns18_solutions)}")
print(f"  HV: {movns18_hv:.4f}")

# 3. Test MOEA/D v18 (degraded)
print(f"\n3. MOEA/D v18 (Degraded)")
print("-"*70)

moead = MOEAD_V18(package_name, pop_size=50, max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

moead_objectives = np.array([moead.evaluate_objectives(sol['chromosome'])
                             for sol in moead_solutions])

# Get metrics
moead_metrics = moead.get_metrics_history()
if moead_metrics and 'hypervolume' in moead_metrics and moead_metrics['hypervolume']:
    moead_hv = moead_metrics['hypervolume'][-1]
else:
    qm = QualityMetrics()
    moead_hv = qm.hypervolume(moead_objectives, ref_point=[0, 0, 15]) if len(moead_objectives) > 0 else 0

print(f"\nResults:")
print(f"  Time: {moead_time:.2f}s")
print(f"  Solutions: {len(moead_solutions)}")
print(f"  HV: {moead_hv:.4f}")

# 4. Calculate epsilon-indicator
print(f"\n4. Epsilon-Indicator Analysis")
print("-"*70)

if len(movns19_objectives) > 0 and len(moead_objectives) > 0:
    combined = np.vstack([movns19_objectives, moead_objectives])
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
    movns19_epsilon = qm_eps.epsilon_indicator(movns19_objectives, reference_set)
    moead_epsilon = qm_eps.epsilon_indicator(moead_objectives, reference_set)

    print(f"MOVNS v19 epsilon: {movns19_epsilon:.4f}")
    print(f"MOEA/D v18 epsilon: {moead_epsilon:.4f}")

# Summary
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

print("\n1. EXECUTION TIME")
print("-"*70)
print(f"MOVNS v19: {movns19_time:.2f}s")
print(f"MOVNS v18: {movns18_time:.2f}s")
print(f"MOEA/D v18: {moead_time:.2f}s")

if movns18_time > 0:
    speedup = movns18_time / movns19_time
    print(f"\nv19 Speedup over v18: {speedup:.2f}x")
    print(f"Time saved: {movns18_time - movns19_time:.2f}s ({(1 - movns19_time/movns18_time)*100:.1f}% reduction)")

print("\n2. HYPERVOLUME (higher is better)")
print("-"*70)
print(f"MOVNS v19: {movns19_hv:.4f}")
print(f"MOVNS v18: {movns18_hv:.4f}")
print(f"MOEA/D v18: {moead_hv:.4f}")

# Determine winner
best_hv = max(movns19_hv, movns18_hv, moead_hv)
if best_hv == movns19_hv:
    print("WINNER: MOVNS v19 (optimized)")
elif best_hv == movns18_hv:
    print("WINNER: MOVNS v18")
else:
    print("WINNER: MOEA/D v18")

print("\n3. QUALITY vs SPEED TRADE-OFF")
print("-"*70)
print(f"MOVNS v19 maintains {(movns19_hv/movns18_hv)*100:.1f}% of v18 quality")
print(f"while being {speedup:.2f}x faster")

print("\n" + "="*70)
print("CONCLUSION:")
if speedup > 1.5 and movns19_hv >= movns18_hv * 0.95:
    print("v19 optimization successful! Significant speedup with minimal quality loss.")
elif speedup > 1.5:
    print("v19 is faster but with some quality trade-off.")
else:
    print("v19 optimization impact is minimal on this test size.")
print("="*70)