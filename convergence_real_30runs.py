"""
Real convergence analysis - MOVNS v22 vs MOEA/D v18
30 runs, 30 iterations, real calculations following rules.json
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("REAL CONVERGENCE ANALYSIS - 30 RUNS")
print("="*70)

# Parameters
package_name = 'fastapi'
max_iterations = 30
pop_size = 50
n_runs = 30  # Statistical significance

print(f"\nConfiguration:")
print(f"  Package: {package_name}")
print(f"  Iterations: {max_iterations}")
print(f"  Population/Archive: {pop_size}")
print(f"  Runs: {n_runs}")

# Storage for results
all_results = {
    'movns_hv': [],
    'movns_spacing': [],
    'movns_time': [],
    'moead_hv': [],
    'moead_spacing': [],
    'moead_time': []
}

# Run MOVNS
print(f"\nRunning MOVNS v22 - {n_runs} runs...")
for run in range(n_runs):
    print(f"  Run {run+1}/{n_runs}...", end='', flush=True)

    movns = MOVNS_V22(package_name, archive_size=pop_size,
                      max_iterations=max_iterations, track_metrics=True)

    start = time.time()
    solutions = movns.run()
    elapsed = time.time() - start

    # Get final metrics
    if solutions:
        objectives = np.array([movns.evaluate_objectives(sol['chromosome'])
                              for sol in solutions])

        # Real HV calculation
        qm = QualityMetrics()
        hv = qm.hypervolume(objectives, ref_point=[0, 0, 15]) if len(objectives) > 0 else 0

        # Real spacing calculation
        spacing = qm.spacing(objectives) if len(objectives) > 1 else float('inf')
    else:
        hv = 0
        spacing = float('inf')

    all_results['movns_hv'].append(hv)
    all_results['movns_spacing'].append(spacing)
    all_results['movns_time'].append(elapsed)

    print(f" HV={hv:.4f}, Time={elapsed:.1f}s")

# Run MOEA/D
print(f"\nRunning MOEA/D v18 - {n_runs} runs...")
for run in range(n_runs):
    print(f"  Run {run+1}/{n_runs}...", end='', flush=True)

    try:
        moead = MOEAD_V18(package_name, pop_size=pop_size,
                         max_gen=max_iterations, track_metrics=False)

        start = time.time()
        solutions = moead.run()
        elapsed = time.time() - start

        # Get final metrics
        if solutions:
            objectives = np.array([moead.evaluate_objectives(sol['chromosome'])
                                  for sol in solutions])

            # Real HV calculation
            qm = QualityMetrics()
            hv = qm.hypervolume(objectives, ref_point=[0, 0, 15]) if len(objectives) > 0 else 0

            # Real spacing calculation
            spacing = qm.spacing(objectives) if len(objectives) > 1 else float('inf')
        else:
            hv = 0
            spacing = float('inf')
    except Exception as e:
        print(f" ERROR: {e}")
        hv = 0
        spacing = float('inf')
        elapsed = 0

    all_results['moead_hv'].append(hv)
    all_results['moead_spacing'].append(spacing)
    all_results['moead_time'].append(elapsed)

    print(f" HV={hv:.4f}, Time={elapsed:.1f}s")

# Statistical analysis
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

from scipy import stats

# Wilcoxon signed-rank test
print("\n1. Wilcoxon Signed-Rank Test:")

# HV comparison
statistic, p_value = stats.wilcoxon(all_results['movns_hv'], all_results['moead_hv'])
print(f"   Hypervolume: p-value = {p_value:.6f}")
if p_value < 0.05:
    print(f"   Result: Statistically significant difference (p < 0.05)")
    if np.median(all_results['movns_hv']) > np.median(all_results['moead_hv']):
        print(f"   Winner: MOVNS")
    else:
        print(f"   Winner: MOEA/D")
else:
    print(f"   Result: No significant difference")

# Spacing comparison
statistic, p_value = stats.wilcoxon(all_results['movns_spacing'], all_results['moead_spacing'])
print(f"\n   Spacing: p-value = {p_value:.6f}")
if p_value < 0.05:
    print(f"   Result: Statistically significant difference (p < 0.05)")
    if np.median(all_results['movns_spacing']) < np.median(all_results['moead_spacing']):
        print(f"   Winner: MOVNS")
    else:
        print(f"   Winner: MOEA/D")
else:
    print(f"   Result: No significant difference")

# Time comparison
statistic, p_value = stats.wilcoxon(all_results['movns_time'], all_results['moead_time'])
print(f"\n   Execution Time: p-value = {p_value:.6f}")
if p_value < 0.05:
    print(f"   Result: Statistically significant difference (p < 0.05)")
    if np.median(all_results['movns_time']) < np.median(all_results['moead_time']):
        print(f"   Winner: MOVNS")
    else:
        print(f"   Winner: MOEA/D")
else:
    print(f"   Result: No significant difference")

# Summary statistics
print("\n2. Summary Statistics:")
print(f"\n   MOVNS:")
print(f"   - HV: {np.median(all_results['movns_hv']):.4f} ± {np.std(all_results['movns_hv']):.4f}")
print(f"   - Spacing: {np.median(all_results['movns_spacing']):.4f} ± {np.std(all_results['movns_spacing']):.4f}")
print(f"   - Time: {np.median(all_results['movns_time']):.2f}s ± {np.std(all_results['movns_time']):.2f}s")

print(f"\n   MOEA/D:")
print(f"   - HV: {np.median(all_results['moead_hv']):.4f} ± {np.std(all_results['moead_hv']):.4f}")
print(f"   - Spacing: {np.median(all_results['moead_spacing']):.4f} ± {np.std(all_results['moead_spacing']):.4f}")
print(f"   - Time: {np.median(all_results['moead_time']):.2f}s ± {np.std(all_results['moead_time']):.2f}s")

# Save results
df = pd.DataFrame(all_results)
csv_filename = f'convergence_real_30runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(csv_filename, index=False)

print(f"\n3. Results saved to: {csv_filename}")
print("="*70)