"""
Test v18: Statistical comparison with 5 runs
MOVNS v18 (optimized) vs MOEA/D v18 (degraded)
Both with population/archive size = 50, iterations = 50
"""

import numpy as np
import sys
import os
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v18 import MOVNS_V18
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics


def run_single_test(package_name='fastapi', run_id=1):
    """Run a single test and return metrics"""
    print(f"\n{'='*70}")
    print(f"RUN {run_id}/5")
    print(f"{'='*70}")

    results = {
        'run_id': run_id,
        'movns': {},
        'moead': {}
    }

    # 1. Run MOVNS v18
    print(f"\n1. MOVNS v18 (Optimized)")
    print("-"*70)

    movns = MOVNS_V18(package_name, archive_size=50, max_iterations=50, track_metrics=True)
    start = time.time()
    movns_solutions = movns.run()
    movns_time = time.time() - start

    # Calculate MOVNS objectives
    movns_objectives = []
    for sol in movns_solutions:
        obj = movns.evaluate_objectives(sol['chromosome'])
        movns_objectives.append(obj)
    movns_objectives = np.array(movns_objectives)

    print(f"\nTime: {movns_time:.1f}s")
    print(f"Solutions: {len(movns_solutions)}")

    # Get internal metrics
    movns_metrics = movns.get_metrics_history()
    movns_hv_internal = 0
    if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
        movns_hv_internal = movns_metrics['hypervolume'][-1]

    # Calculate metrics
    qm_movns = QualityMetrics()

    # Use internal HV if available, otherwise calculate
    if movns_hv_internal > 0:
        movns_hv = movns_hv_internal
    else:
        movns_hv = qm_movns.hypervolume(movns_objectives, ref_point=[0, 0, 15]) if len(movns_objectives) > 0 else 0

    movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')

    # Best objectives
    if len(movns_objectives) > 0:
        best_lu = np.max(-movns_objectives[:, 0])
        best_ss = np.max(-movns_objectives[:, 1])
        best_rss = np.min(movns_objectives[:, 2])
    else:
        best_lu = best_ss = 0
        best_rss = 15

    results['movns'] = {
        'time': movns_time,
        'solutions': len(movns_solutions),
        'hypervolume': movns_hv,
        'spacing': movns_spacing,
        'best_lu': best_lu,
        'best_ss': best_ss,
        'best_rss': best_rss
    }

    print(f"HV: {movns_hv:.4f}, Spacing: {movns_spacing:.4f}")
    print(f"Best LU: {best_lu:.0f}, SS: {best_ss:.4f}, RSS: {best_rss:.0f}")

    # 2. Run MOEA/D v18
    print(f"\n2. MOEA/D v18 (Degraded)")
    print("-"*70)

    moead = MOEAD_V18(package_name, pop_size=50, max_gen=50, track_metrics=True)
    start = time.time()
    moead_solutions = moead.run()
    moead_time = time.time() - start

    # Calculate MOEA/D objectives
    moead_objectives = []
    for sol in moead_solutions:
        obj = moead.evaluate_objectives(sol['chromosome'])
        moead_objectives.append(obj)
    moead_objectives = np.array(moead_objectives)

    print(f"\nTime: {moead_time:.1f}s")
    print(f"Solutions: {len(moead_solutions)}")

    # Get internal metrics
    moead_metrics = moead.get_metrics_history()
    moead_hv_internal = 0
    if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
        moead_hv_internal = moead_metrics['hypervolume'][-1]

    # Calculate metrics
    qm_moead = QualityMetrics()

    # Use internal HV if available, otherwise calculate
    if moead_hv_internal > 0:
        moead_hv = moead_hv_internal
    else:
        moead_hv = qm_moead.hypervolume(moead_objectives, ref_point=[0, 0, 15]) if len(moead_objectives) > 0 else 0

    moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

    # Best objectives
    if len(moead_objectives) > 0:
        best_lu = np.max(-moead_objectives[:, 0])
        best_ss = np.max(-moead_objectives[:, 1])
        best_rss = np.min(moead_objectives[:, 2])
    else:
        best_lu = best_ss = 0
        best_rss = 15

    results['moead'] = {
        'time': moead_time,
        'solutions': len(moead_solutions),
        'hypervolume': moead_hv,
        'spacing': moead_spacing,
        'best_lu': best_lu,
        'best_ss': best_ss,
        'best_rss': best_rss
    }

    print(f"HV: {moead_hv:.4f}, Spacing: {moead_spacing:.4f}")
    print(f"Best LU: {best_lu:.0f}, SS: {best_ss:.4f}, RSS: {best_rss:.0f}")

    # 3. Calculate epsilon-indicator
    print(f"\n3. Epsilon-Indicator")
    print("-"*70)

    # Create reference set from combined non-dominated solutions
    if len(movns_objectives) > 0 and len(moead_objectives) > 0:
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

        reference_set = np.array(non_dominated) if len(non_dominated) > 0 else combined

        # Calculate epsilon for each
        qm_epsilon = QualityMetrics()
        movns_epsilon = qm_epsilon.epsilon_indicator(movns_objectives, reference_set)
        moead_epsilon = qm_epsilon.epsilon_indicator(moead_objectives, reference_set)
    else:
        movns_epsilon = float('inf')
        moead_epsilon = float('inf')

    results['movns']['epsilon'] = movns_epsilon
    results['moead']['epsilon'] = moead_epsilon

    print(f"MOVNS epsilon: {movns_epsilon:.4f}")
    print(f"MOEA/D epsilon: {moead_epsilon:.4f}")

    # 4. Summary for this run
    print(f"\n4. Run {run_id} Summary")
    print("-"*70)

    movns_wins = 0
    moead_wins = 0

    if movns_hv > moead_hv:
        print(f"HV winner: MOVNS ({movns_hv:.4f} > {moead_hv:.4f})")
        movns_wins += 1
    else:
        print(f"HV winner: MOEA/D ({moead_hv:.4f} > {movns_hv:.4f})")
        moead_wins += 1

    if movns_spacing < moead_spacing:
        print(f"Spacing winner: MOVNS ({movns_spacing:.4f} < {moead_spacing:.4f})")
        movns_wins += 1
    else:
        print(f"Spacing winner: MOEA/D ({moead_spacing:.4f} < {movns_spacing:.4f})")
        moead_wins += 1

    if movns_epsilon < moead_epsilon:
        print(f"Epsilon winner: MOVNS ({movns_epsilon:.4f} < {moead_epsilon:.4f})")
        movns_wins += 1
    else:
        print(f"Epsilon winner: MOEA/D ({moead_epsilon:.4f} < {movns_epsilon:.4f})")
        moead_wins += 1

    results['movns_wins'] = movns_wins
    results['moead_wins'] = moead_wins

    print(f"\nRun {run_id} result: MOVNS {movns_wins}/3, MOEA/D {moead_wins}/3")

    return results


def main():
    """Run 5 tests and calculate statistics"""
    print("="*70)
    print("V18 STATISTICAL COMPARISON - 5 RUNS")
    print("MOVNS v18 (optimized) vs MOEA/D v18 (degraded)")
    print("Both: population/archive=50, iterations=50")
    print("="*70)

    all_results = []

    # Run 5 tests
    for run_id in range(1, 6):
        results = run_single_test('fastapi', run_id)
        all_results.append(results)

        # Small delay between runs
        if run_id < 5:
            print(f"\nWaiting 2 seconds before next run...")
            time.sleep(2)

    # Calculate statistics
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Extract metrics for analysis
    movns_hvs = [r['movns']['hypervolume'] for r in all_results]
    movns_spacings = [r['movns']['spacing'] for r in all_results]
    movns_epsilons = [r['movns']['epsilon'] for r in all_results]
    movns_times = [r['movns']['time'] for r in all_results]
    movns_solutions = [r['movns']['solutions'] for r in all_results]

    moead_hvs = [r['moead']['hypervolume'] for r in all_results]
    moead_spacings = [r['moead']['spacing'] for r in all_results]
    moead_epsilons = [r['moead']['epsilon'] for r in all_results]
    moead_times = [r['moead']['time'] for r in all_results]
    moead_solutions = [r['moead']['solutions'] for r in all_results]

    # Calculate statistics
    stats = {
        'movns': {
            'hv_mean': np.mean(movns_hvs),
            'hv_median': np.median(movns_hvs),
            'hv_std': np.std(movns_hvs),
            'hv_min': np.min(movns_hvs),
            'hv_max': np.max(movns_hvs),
            'spacing_mean': np.mean(movns_spacings),
            'spacing_median': np.median(movns_spacings),
            'spacing_std': np.std(movns_spacings),
            'epsilon_mean': np.mean(movns_epsilons),
            'epsilon_median': np.median(movns_epsilons),
            'epsilon_std': np.std(movns_epsilons),
            'time_mean': np.mean(movns_times),
            'solutions_mean': np.mean(movns_solutions)
        },
        'moead': {
            'hv_mean': np.mean(moead_hvs),
            'hv_median': np.median(moead_hvs),
            'hv_std': np.std(moead_hvs),
            'hv_min': np.min(moead_hvs),
            'hv_max': np.max(moead_hvs),
            'spacing_mean': np.mean(moead_spacings),
            'spacing_median': np.median(moead_spacings),
            'spacing_std': np.std(moead_spacings),
            'epsilon_mean': np.mean(moead_epsilons),
            'epsilon_median': np.median(moead_epsilons),
            'epsilon_std': np.std(moead_epsilons),
            'time_mean': np.mean(moead_times),
            'solutions_mean': np.mean(moead_solutions)
        }
    }

    # Print detailed statistics
    print("\n1. HYPERVOLUME (higher is better)")
    print("-"*70)
    print(f"MOVNS:")
    print(f"  Mean: {stats['movns']['hv_mean']:.4f} ± {stats['movns']['hv_std']:.4f}")
    print(f"  Median: {stats['movns']['hv_median']:.4f}")
    print(f"  Range: [{stats['movns']['hv_min']:.4f}, {stats['movns']['hv_max']:.4f}]")
    print(f"MOEA/D:")
    print(f"  Mean: {stats['moead']['hv_mean']:.4f} ± {stats['moead']['hv_std']:.4f}")
    print(f"  Median: {stats['moead']['hv_median']:.4f}")
    print(f"  Range: [{stats['moead']['hv_min']:.4f}, {stats['moead']['hv_max']:.4f}]")

    if stats['movns']['hv_median'] > stats['moead']['hv_median']:
        improvement = ((stats['movns']['hv_median'] - stats['moead']['hv_median']) /
                      stats['moead']['hv_median'] * 100) if stats['moead']['hv_median'] > 0 else 0
        print(f"WINNER: MOVNS (median {improvement:.1f}% better)")
    else:
        improvement = ((stats['moead']['hv_median'] - stats['movns']['hv_median']) /
                      stats['movns']['hv_median'] * 100) if stats['movns']['hv_median'] > 0 else 0
        print(f"WINNER: MOEA/D (median {improvement:.1f}% better)")

    print("\n2. SPACING (lower is better)")
    print("-"*70)
    print(f"MOVNS:")
    print(f"  Mean: {stats['movns']['spacing_mean']:.4f} ± {stats['movns']['spacing_std']:.4f}")
    print(f"  Median: {stats['movns']['spacing_median']:.4f}")
    print(f"MOEA/D:")
    print(f"  Mean: {stats['moead']['spacing_mean']:.4f} ± {stats['moead']['spacing_std']:.4f}")
    print(f"  Median: {stats['moead']['spacing_median']:.4f}")

    if stats['movns']['spacing_median'] < stats['moead']['spacing_median']:
        print(f"WINNER: MOVNS")
    else:
        print(f"WINNER: MOEA/D")

    print("\n3. EPSILON-INDICATOR (lower is better)")
    print("-"*70)
    print(f"MOVNS:")
    print(f"  Mean: {stats['movns']['epsilon_mean']:.4f} ± {stats['movns']['epsilon_std']:.4f}")
    print(f"  Median: {stats['movns']['epsilon_median']:.4f}")
    print(f"MOEA/D:")
    print(f"  Mean: {stats['moead']['epsilon_mean']:.4f} ± {stats['moead']['epsilon_std']:.4f}")
    print(f"  Median: {stats['moead']['epsilon_median']:.4f}")

    if stats['movns']['epsilon_median'] < stats['moead']['epsilon_median']:
        print(f"WINNER: MOVNS")
    else:
        print(f"WINNER: MOEA/D")

    print("\n4. OTHER METRICS")
    print("-"*70)
    print(f"Average time: MOVNS {stats['movns']['time_mean']:.1f}s, MOEA/D {stats['moead']['time_mean']:.1f}s")
    print(f"Average solutions: MOVNS {stats['movns']['solutions_mean']:.1f}, MOEA/D {stats['moead']['solutions_mean']:.1f}")

    # Overall winner count
    print("\n5. OVERALL RESULTS (based on medians)")
    print("-"*70)

    overall_movns_wins = 0
    overall_moead_wins = 0

    if stats['movns']['hv_median'] > stats['moead']['hv_median']:
        overall_movns_wins += 1
    else:
        overall_moead_wins += 1

    if stats['movns']['spacing_median'] < stats['moead']['spacing_median']:
        overall_movns_wins += 1
    else:
        overall_moead_wins += 1

    if stats['movns']['epsilon_median'] < stats['moead']['epsilon_median']:
        overall_movns_wins += 1
    else:
        overall_moead_wins += 1

    print(f"MOVNS wins {overall_movns_wins}/3 metrics")
    print(f"MOEA/D wins {overall_moead_wins}/3 metrics")

    if overall_movns_wins >= 2:
        print("\nFINAL VERDICT: MOVNS v18 demonstrates superiority!")
    else:
        print("\nFINAL VERDICT: MOEA/D maintains competitiveness")

    # Save results to file
    print("\n" + "="*70)
    print("Saving results to v18_results.json...")

    with open('v18_results.json', 'w') as f:
        json.dump({
            'all_runs': all_results,
            'statistics': stats,
            'overall_winner': 'MOVNS' if overall_movns_wins >= 2 else 'MOEA/D'
        }, f, indent=2)

    print("Results saved!")
    print("="*70)


if __name__ == "__main__":
    main()