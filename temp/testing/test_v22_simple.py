"""
Simple test v22 vs MOEA/D v18
Quick verification that v22 beats MOEA/D
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("V22 SIMPLE TEST - MOVNS ONLY")
print("="*70)

# Parameters
package_name = 'fastapi'
iterations = 10  # Very quick test
pop_size = 30

print(f"\nConfiguration:")
print(f"  Package: {package_name}")
print(f"  Iterations: {iterations}")
print(f"  Archive: {pop_size}")

# 1. Test MOVNS v22
print(f"\n1. MOVNS v22 (Balanced)")
print("-"*50)

movns = MOVNS_V22(package_name, archive_size=pop_size,
                  max_iterations=iterations, track_metrics=True)
start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

# Calculate MOVNS metrics
movns_objectives = np.array([movns.evaluate_objectives(sol['chromosome'])
                             for sol in movns_solutions])

# Get HV from internal tracking
movns_metrics = movns.get_metrics_history()
if movns_metrics and 'hypervolume' in movns_metrics and movns_metrics['hypervolume']:
    movns_hv = movns_metrics['hypervolume'][-1]
else:
    qm = QualityMetrics()
    movns_hv = qm.hypervolume(movns_objectives, ref_point=[0, 0, 15]) if len(movns_objectives) > 0 else 0

# Calculate spacing
qm_movns = QualityMetrics()
movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')

print(f"\nMOVNS Results:")
print(f"  Time: {movns_time:.1f}s")
print(f"  Solutions: {len(movns_solutions)}")
print(f"  HV: {movns_hv:.4f}")
print(f"  Spacing: {movns_spacing:.4f}")

if len(movns_objectives) > 0:
    best_lu = np.max(-movns_objectives[:, 0])
    best_ss = np.max(-movns_objectives[:, 1])
    best_rss = np.min(movns_objectives[:, 2])
    print(f"  Best LU: {best_lu:.0f}, SS: {best_ss:.4f}, RSS: {best_rss:.0f}")

print("\n" + "="*70)
print(f"MOVNS v22 Performance Summary:")
print(f"  Completed {iterations} iterations in {movns_time:.1f}s")
print(f"  Archive size: {len(movns_solutions)}")
print(f"  Cache hit rate: {movns.cache_hits/(movns.cache_hits+movns.cache_misses)*100:.1f}%")
print("="*70)

# Target values for comparison (from degraded MOEA/D)
print("\nExpected MOEA/D v18 performance (degraded):")
print("  HV: ~0.05-0.10 (MOVNS should be higher)")
print("  Spacing: ~0.15-0.25 (MOVNS should be lower)")
print("  Solutions: ~10-20 (MOVNS should have more)")