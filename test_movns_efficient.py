"""
Test MOVNS Efficient vs MOEA/D
Complete test without simplifications
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_efficient import MOVNS_Efficient
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOVNS EFFICIENT vs MOEA/D - COMPLETE TEST")
print("="*70)

iterations = 50

# 1. MOVNS Efficient
print("\n1. MOVNS Efficient")
print("-"*70)

movns = MOVNS_Efficient('fastapi', archive_size=100, max_iterations=iterations, track_metrics=True)
start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

# Get metrics from MOVNS internal tracking
movns_metrics = movns.get_metrics_history()
movns_hv = 0
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    movns_hv = movns_metrics['hypervolume'][-1]

# Calculate spacing with separate QualityMetrics instance
movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

qm_movns = QualityMetrics()
movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')

print(f"Time: {movns_time:.1f}s")
print(f"Solutions: {len(movns_solutions)}")
print(f"HV: {movns_hv:.4f}")
print(f"Spacing: {movns_spacing:.4f}")

if len(movns_objectives) > 0:
    best_lu = np.max(-movns_objectives[:, 0])
    best_ss = np.max(-movns_objectives[:, 1])
    best_rss = np.min(movns_objectives[:, 2])
    print(f"Best LU: {best_lu:.0f}")
    print(f"Best SS: {best_ss:.4f}")
    print(f"Best RSS: {best_rss:.0f}")

# 2. MOEA/D
print("\n2. MOEA/D Normalized")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

# Get metrics from MOEA/D internal tracking
moead_metrics = moead.get_metrics_history()
moead_hv = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv = moead_metrics['hypervolume'][-1]

# Calculate spacing with separate QualityMetrics instance
moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

qm_moead = QualityMetrics()
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

print(f"Time: {moead_time:.1f}s")
print(f"Solutions: {len(moead_solutions)}")
print(f"HV: {moead_hv:.4f}")
print(f"Spacing: {moead_spacing:.4f}")

if len(moead_objectives) > 0:
    best_lu = np.max(-moead_objectives[:, 0])
    best_ss = np.max(-moead_objectives[:, 1])
    best_rss = np.min(moead_objectives[:, 2])
    print(f"Best LU: {best_lu:.0f}")
    print(f"Best SS: {best_ss:.4f}")
    print(f"Best RSS: {best_rss:.0f}")

# 3. RESULTS
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

movns_wins = 0
moead_wins = 0

print("\n1. HYPERVOLUME (higher is better)")
print(f"   MOVNS: {movns_hv:.4f}")
print(f"   MOEA/D: {moead_hv:.4f}")
if movns_hv > moead_hv:
    print("   WINNER: MOVNS")
    movns_wins += 1
else:
    print("   WINNER: MOEA/D")
    moead_wins += 1

print("\n2. SPACING (lower is better)")
print(f"   MOVNS: {movns_spacing:.4f}")
print(f"   MOEA/D: {moead_spacing:.4f}")
if movns_spacing < moead_spacing:
    print("   WINNER: MOVNS")
    movns_wins += 1
else:
    print("   WINNER: MOEA/D")
    moead_wins += 1

print("\n3. ARCHIVE ANALYSIS")
print(f"   MOVNS: {len(movns_solutions)} solutions")
print(f"   MOEA/D: {len(moead_solutions)} solutions")
print(f"   Target: 80-100 solutions for MOVNS")

if 80 <= len(movns_solutions) <= 100:
    print("   MOVNS archive size: ACHIEVED")
else:
    print(f"   MOVNS archive size: NOT ACHIEVED (got {len(movns_solutions)})")

print("\n" + "="*70)
if movns_wins == 2:
    print("SUCCESS: MOVNS WINS BOTH METRICS")
    print("Objective achieved!")
elif movns_wins == 1:
    print(f"PARTIAL: MOVNS wins {movns_wins}/2 metrics")
    print("Need to improve further")
else:
    print("FAILURE: MOVNS loses both metrics")
    print("Major adjustments needed")

print("="*70)