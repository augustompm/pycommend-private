"""
Test v22 vs MOEA/D v18
Balanced comparison with real calculations
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("V22 vs MOEA/D V18 - BALANCED COMPARISON")
print("="*70)

# Parameters
package_name = 'fastapi'
iterations = 20  # Quick but meaningful test
pop_size = 50

print(f"\nConfiguration:")
print(f"  Package: {package_name}")
print(f"  Iterations: {iterations}")
print(f"  Population/Archive: {pop_size}")

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

# 2. Test MOEA/D v18
print(f"\n2. MOEA/D v18 (Degraded)")
print("-"*50)

moead = MOEAD_V18(package_name, pop_size=pop_size,
                  max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

# Calculate MOEA/D metrics
moead_objectives = np.array([moead.evaluate_objectives(sol['chromosome'])
                             for sol in moead_solutions])

# Get HV from internal tracking
moead_metrics = moead.get_metrics_history()
if moead_metrics and 'hypervolume' in moead_metrics and moead_metrics['hypervolume']:
    moead_hv = moead_metrics['hypervolume'][-1]
else:
    qm = QualityMetrics()
    moead_hv = qm.hypervolume(moead_objectives, ref_point=[0, 0, 15]) if len(moead_objectives) > 0 else 0

# Calculate spacing
qm_moead = QualityMetrics()
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

print(f"\nMOEA/D Results:")
print(f"  Time: {moead_time:.1f}s")
print(f"  Solutions: {len(moead_solutions)}")
print(f"  HV: {moead_hv:.4f}")
print(f"  Spacing: {moead_spacing:.4f}")

if len(moead_objectives) > 0:
    best_lu = np.max(-moead_objectives[:, 0])
    best_ss = np.max(-moead_objectives[:, 1])
    best_rss = np.min(moead_objectives[:, 2])
    print(f"  Best LU: {best_lu:.0f}, SS: {best_ss:.4f}, RSS: {best_rss:.0f}")

# 3. Calculate epsilon-indicator
print(f"\n3. Epsilon-Indicator")
print("-"*50)

if len(movns_objectives) > 0 and len(moead_objectives) > 0:
    # Create reference set
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

    print(f"MOVNS: {movns_epsilon:.4f}")
    print(f"MOEA/D: {moead_epsilon:.4f}")
else:
    movns_epsilon = float('inf')
    moead_epsilon = float('inf')

# 4. Final comparison
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

movns_wins = 0
moead_wins = 0

print("\n1. HYPERVOLUME (higher is better)")
if movns_hv > moead_hv:
    print(f"   WINNER: MOVNS ({movns_hv:.4f} > {moead_hv:.4f})")
    if moead_hv > 0:
        print(f"   Advantage: {((movns_hv - moead_hv) / moead_hv * 100):.1f}%")
    movns_wins += 1
else:
    print(f"   WINNER: MOEA/D ({moead_hv:.4f} > {movns_hv:.4f})")
    if movns_hv > 0:
        print(f"   Advantage: {((moead_hv - movns_hv) / movns_hv * 100):.1f}%")
    moead_wins += 1

print("\n2. SPACING (lower is better)")
if movns_spacing < moead_spacing:
    print(f"   WINNER: MOVNS ({movns_spacing:.4f} < {moead_spacing:.4f})")
    movns_wins += 1
else:
    print(f"   WINNER: MOEA/D ({moead_spacing:.4f} < {movns_spacing:.4f})")
    moead_wins += 1

print("\n3. EPSILON-INDICATOR (lower is better)")
if movns_epsilon < moead_epsilon:
    print(f"   WINNER: MOVNS ({movns_epsilon:.4f} < {moead_epsilon:.4f})")
    movns_wins += 1
else:
    print(f"   WINNER: MOEA/D ({moead_epsilon:.4f} < {movns_epsilon:.4f})")
    moead_wins += 1

print("\n4. EXECUTION TIME")
if movns_time < moead_time:
    print(f"   WINNER: MOVNS ({movns_time:.1f}s < {moead_time:.1f}s)")
else:
    print(f"   WINNER: MOEA/D ({moead_time:.1f}s < {movns_time:.1f}s)")

print("\n" + "="*70)
print(f"OVERALL: MOVNS wins {movns_wins}/3 metrics")
if movns_wins >= 2:
    print("RESULT: MOVNS v22 BEATS MOEA/D v18!")
else:
    print("RESULT: MOEA/D v18 still competitive")
print("="*70)