"""
Test v20: Run 3 times and calculate averages
Monitor execution time and status in real-time
"""

import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v20 import MOVNS_V20
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("V20 TIMING TEST - 3 RUNS")
print("="*70)

# Configuration
package_name = 'fastapi'
iterations = 10  # Quick test with 10 iterations
archive_size = 50
n_runs = 3

# Storage for results
times = []
hvs = []
solutions_counts = []
cache_rates = []

for run in range(1, n_runs + 1):
    print(f"\n{'='*70}")
    print(f"RUN {run}/{n_runs} - Started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    # Create new instance for each run
    movns = MOVNS_V20(package_name, archive_size=archive_size,
                      max_iterations=iterations, track_metrics=True)

    # Run and time
    start_time = time.time()
    print(f"\nStarting execution...")

    try:
        solutions = movns.run()
        elapsed = time.time() - start_time

        print(f"\nRun {run} completed in {elapsed:.2f}s")

        # Calculate metrics
        if solutions:
            objectives = np.array([movns.evaluate_objectives(sol['chromosome'])
                                  for sol in solutions])

            # Get HV from internal tracking or calculate
            if hasattr(movns, 'metrics_history') and movns.metrics_history and 'hypervolume' in movns.metrics_history:
                hv = movns.metrics_history['hypervolume'][-1] if movns.metrics_history['hypervolume'] else 0
            else:
                qm = QualityMetrics()
                hv = qm.hypervolume(objectives, ref_point=[0, 0, 15]) if len(objectives) > 0 else 0

            # Cache statistics
            total_evals = movns.cache_hits + movns.cache_misses
            cache_rate = (movns.cache_hits / total_evals * 100) if total_evals > 0 else 0

            # Store results
            times.append(elapsed)
            hvs.append(hv)
            solutions_counts.append(len(solutions))
            cache_rates.append(cache_rate)

            # Print run summary
            print(f"\nRun {run} Summary:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Solutions: {len(solutions)}")
            print(f"  HV: {hv:.4f}")
            print(f"  Cache hit rate: {cache_rate:.1f}%")
            print(f"  Cache size: {len(movns.objective_cache)}")

            # Sample solution
            if solutions:
                sample = solutions[0]
                print(f"  Sample packages: {sample['packages'][:3]}")

        else:
            print(f"Run {run} failed - no solutions found")
            times.append(elapsed)
            hvs.append(0)
            solutions_counts.append(0)
            cache_rates.append(0)

    except Exception as e:
        print(f"Run {run} error: {e}")
        elapsed = time.time() - start_time
        times.append(elapsed)
        hvs.append(0)
        solutions_counts.append(0)
        cache_rates.append(0)

# Calculate and display statistics
print("\n" + "="*70)
print("FINAL STATISTICS (3 RUNS)")
print("="*70)

print("\n1. EXECUTION TIME")
print("-"*40)
print(f"Times: {[f'{t:.2f}s' for t in times]}")
print(f"Average: {np.mean(times):.2f}s")
print(f"Std Dev: {np.std(times):.2f}s")
print(f"Min: {np.min(times):.2f}s")
print(f"Max: {np.max(times):.2f}s")

print("\n2. HYPERVOLUME")
print("-"*40)
print(f"HVs: {[f'{h:.4f}' for h in hvs]}")
print(f"Average: {np.mean(hvs):.4f}")
print(f"Std Dev: {np.std(hvs):.4f}")

print("\n3. ARCHIVE SIZE")
print("-"*40)
print(f"Sizes: {solutions_counts}")
print(f"Average: {np.mean(solutions_counts):.1f}")

print("\n4. CACHE PERFORMANCE")
print("-"*40)
print(f"Hit rates: {[f'{r:.1f}%' for r in cache_rates]}")
print(f"Average: {np.mean(cache_rates):.1f}%")

print("\n" + "="*70)
print("CONCLUSION")
print("-"*40)

avg_time = np.mean(times)
if avg_time < 10:
    print(f"✓ FAST: Average {avg_time:.2f}s for {iterations} iterations")
    print(f"  Estimated for 50 iterations: {avg_time * 5:.1f}s")
elif avg_time < 30:
    print(f"⚠ MODERATE: Average {avg_time:.2f}s for {iterations} iterations")
    print(f"  Estimated for 50 iterations: {avg_time * 5:.1f}s")
else:
    print(f"✗ SLOW: Average {avg_time:.2f}s for {iterations} iterations")
    print(f"  Needs further optimization")

if np.mean(cache_rates) > 50:
    print(f"✓ Good cache performance: {np.mean(cache_rates):.1f}% hit rate")
else:
    print(f"⚠ Cache could be better: {np.mean(cache_rates):.1f}% hit rate")

print("="*70)