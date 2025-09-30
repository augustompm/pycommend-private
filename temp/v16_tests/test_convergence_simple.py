"""
Simplified Convergence Detection Test
Compare MOVNS Advanced and MOEA/D when they converge
Proper HV calculation with normalization
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


def calculate_hv(solutions, algo):
    """Calculate proper hypervolume"""
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

    normalized = []
    for obj in objectives:
        norm_obj = algo.normalize_objectives(obj)
        normalized.append(norm_obj)

    normalized = np.array(normalized)

    qm = QualityMetrics()
    ref_point = np.array([0, 0, 1.0])
    hv = qm.hypervolume(normalized, ref_point)

    return hv


def run_with_convergence_detection(algo_class, package, no_improvement_limit=10, **kwargs):
    """Run algorithm with convergence detection"""

    algo_name = algo_class.__name__

    print(f"\n{'='*60}")
    print(f"Running {algo_name}")
    print(f"Convergence: {no_improvement_limit} iterations without improvement")
    print(f"{'='*60}")

    algo = algo_class(package, track_metrics=True, **kwargs)

    start_time = time.time()

    solutions = []
    hv_history = []
    best_hv = 0
    no_improvement = 0
    convergence_iter = None
    convergence_hv = 0

    if 'MOVNS' in algo_name:
        max_iter = kwargs.get('max_iterations', 100)
        check_interval = 2
    else:
        max_iter = kwargs.get('max_gen', 100)
        check_interval = 2

    solutions = algo.run()

    metrics = algo.get_metrics_history()

    if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0:
        hv_history = metrics['hypervolume']

        for i, hv in enumerate(hv_history):
            if hv > best_hv:
                best_hv = hv
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= no_improvement_limit and convergence_iter is None:
                convergence_iter = i * check_interval
                convergence_hv = hv
                print(f"Convergence detected at iteration {convergence_iter}")
                print(f"Convergence HV: {convergence_hv:.4f}")
                break

    if convergence_iter is None:
        convergence_iter = max_iter
        convergence_hv = hv_history[-1] if hv_history else 0

    elapsed_time = time.time() - start_time

    actual_hv = calculate_hv(solutions, algo)

    best_objectives = {'lu': 0, 'ss': 0, 'rss': float('inf')}
    for sol in solutions:
        obj = algo.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_objectives['lu']:
            best_objectives['lu'] = -obj[0]
        if -obj[1] > best_objectives['ss']:
            best_objectives['ss'] = -obj[1]
        if obj[2] < best_objectives['rss']:
            best_objectives['rss'] = obj[2]

    return {
        'algorithm': algo_name,
        'convergence_iteration': convergence_iter,
        'convergence_hv': convergence_hv,
        'actual_hv': actual_hv,
        'best_hv': best_hv,
        'time': elapsed_time,
        'archive_size': len(solutions),
        'best_objectives': best_objectives,
        'hv_history': hv_history
    }


def main():
    """Main convergence comparison"""

    print("\n" + "="*70)
    print("CONVERGENCE DETECTION COMPARISON")
    print("="*70)

    package = 'fastapi'
    no_improvement = 10

    movns_results = run_with_convergence_detection(
        MOVNS_Advanced,
        package,
        no_improvement,
        archive_size=100,
        max_iterations=50
    )

    moead_results = run_with_convergence_detection(
        MOEAD_Normalized,
        package,
        no_improvement,
        pop_size=100,
        max_gen=50
    )

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n1. Convergence Point:")
    print(f"   MOVNS Advanced: Iteration {movns_results['convergence_iteration']}")
    print(f"   MOEA/D:         Generation {moead_results['convergence_iteration']}")

    print(f"\n2. Hypervolume at Convergence:")
    print(f"   MOVNS Advanced: {movns_results['convergence_hv']:.4f}")
    print(f"   MOEA/D:         {moead_results['convergence_hv']:.4f}")

    print(f"\n3. Actual Final HV (calculated):")
    print(f"   MOVNS Advanced: {movns_results['actual_hv']:.4f}")
    print(f"   MOEA/D:         {moead_results['actual_hv']:.4f}")

    if moead_results['actual_hv'] > 0:
        ratio = movns_results['actual_hv'] / moead_results['actual_hv']
        print(f"   Ratio: {ratio*100:.1f}%")
        if ratio > 1.0:
            print(f"   MOVNS superior by {(ratio-1)*100:.1f}%")
        else:
            print(f"   MOEA/D superior by {(1-ratio)*100:.1f}%")

    print(f"\n4. Archive Size:")
    print(f"   MOVNS Advanced: {movns_results['archive_size']} solutions")
    print(f"   MOEA/D:         {moead_results['archive_size']} solutions")

    print(f"\n5. Execution Time:")
    print(f"   MOVNS Advanced: {movns_results['time']:.1f}s")
    print(f"   MOEA/D:         {moead_results['time']:.1f}s")
    print(f"   Speed ratio: {moead_results['time']/movns_results['time']:.1f}x")

    print(f"\n6. Best Objectives:")
    print(f"   MOVNS Advanced:")
    print(f"     LU:  {movns_results['best_objectives']['lu']:.0f}")
    print(f"     SS:  {movns_results['best_objectives']['ss']:.4f}")
    print(f"     RSS: {movns_results['best_objectives']['rss']}")
    print(f"   MOEA/D:")
    print(f"     LU:  {moead_results['best_objectives']['lu']:.0f}")
    print(f"     SS:  {moead_results['best_objectives']['ss']:.4f}")
    print(f"     RSS: {moead_results['best_objectives']['rss']}")

    print(f"\n7. Convergence Speed:")
    if movns_results['convergence_iteration'] < moead_results['convergence_iteration']:
        faster = 'MOVNS Advanced'
        diff = moead_results['convergence_iteration'] - movns_results['convergence_iteration']
    else:
        faster = 'MOEA/D'
        diff = movns_results['convergence_iteration'] - moead_results['convergence_iteration']

    print(f"   {faster} converges {diff} iterations faster")

    print(f"\n8. HV History (last 5):")
    if len(movns_results['hv_history']) >= 5:
        movns_last5 = [f"{h:.4f}" for h in movns_results['hv_history'][-5:]]
        print(f"   MOVNS: {movns_last5}")
    if len(moead_results['hv_history']) >= 5:
        moead_last5 = [f"{h:.4f}" for h in moead_results['hv_history'][-5:]]
        print(f"   MOEA/D: {moead_last5}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    winner_quality = 'MOVNS Advanced' if movns_results['actual_hv'] > moead_results['actual_hv'] else 'MOEA/D'
    winner_speed = 'MOVNS Advanced' if movns_results['convergence_iteration'] < moead_results['convergence_iteration'] else 'MOEA/D'

    print(f"\nQuality winner: {winner_quality} (HV={movns_results['actual_hv'] if winner_quality == 'MOVNS Advanced' else moead_results['actual_hv']:.4f})")
    print(f"Speed winner: {winner_speed} (converges at iteration {movns_results['convergence_iteration'] if winner_speed == 'MOVNS Advanced' else moead_results['convergence_iteration']})")


if __name__ == "__main__":
    main()