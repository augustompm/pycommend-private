"""
Parameter optimization for convergence improvement
Grid search to find best parameters for each algorithm
"""

import numpy as np
import sys
import os
import json
import pickle
from itertools import product

sys.path.append('pycommend-code/src')
os.chdir('pycommend-code')


def evaluate_parameters(algo_class, algo_name, package, params, iterations=30):
    """
    Evaluate a set of parameters for convergence quality
    """
    try:
        if algo_name == 'MOVNS':
            from optimizer.movns_vns import MOVNS_VNS
            algo = MOVNS_VNS(package,
                           archive_size=params.get('archive_size', 100),
                           max_iterations=iterations,
                           track_metrics=True)

        elif algo_name == 'MOEAD_Improved':
            from optimizer.moead_vns_improved import MOEAD_VNS_Improved
            algo = MOEAD_VNS_Improved(package,
                                     pop_size=params.get('pop_size', 100),
                                     max_gen=iterations,
                                     update_rate=params.get('update_rate', 0.2),
                                     elitism_rate=params.get('elitism_rate', 0.1),
                                     acceptance_threshold=params.get('acceptance_threshold', 0.99),
                                     track_metrics=True)

        elif algo_name == 'NSGA2':
            from optimizer.nsga2_vns import NSGA2_VNS
            algo = NSGA2_VNS(package,
                           pop_size=params.get('pop_size', 100),
                           max_gen=iterations,
                           track_metrics=True)

        solutions = algo.run()
        metrics = algo.get_metrics_history()

        if metrics and 'hypervolume' in metrics and metrics['hypervolume']:
            hv = metrics['hypervolume']

            initial_hv = hv[0] if hv[0] > 0 else 0.001
            final_hv = hv[-1]
            improvement = (final_hv - initial_hv) / initial_hv

            monotonic_steps = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
            monotonic_rate = monotonic_steps / (len(hv) - 1) if len(hv) > 1 else 0

            final_stability = np.std(hv[-5:]) if len(hv) >= 5 else 1.0

            score = improvement * 0.5 + monotonic_rate * 0.3 + (1 - min(final_stability, 1)) * 0.2

            return {
                'score': score,
                'improvement': improvement,
                'monotonic_rate': monotonic_rate,
                'final_hv': final_hv,
                'stability': final_stability,
                'hv_history': hv
            }

    except Exception as e:
        print(f"Error evaluating {algo_name} with params {params}: {e}")

    return None


def grid_search_moead():
    """
    Grid search for MOEA/D improved parameters
    """
    print("\nGrid Search for MOEA/D Improved Parameters")
    print("="*60)

    param_grid = {
        'update_rate': [0.15, 0.2, 0.25, 0.3],
        'elitism_rate': [0.05, 0.1, 0.15],
        'acceptance_threshold': [0.97, 0.98, 0.99]
    }

    best_score = -float('inf')
    best_params = {}
    results = []

    param_combinations = list(product(
        param_grid['update_rate'],
        param_grid['elitism_rate'],
        param_grid['acceptance_threshold']
    ))

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for update_rate, elitism_rate, acceptance_threshold in param_combinations:
        params = {
            'update_rate': update_rate,
            'elitism_rate': elitism_rate,
            'acceptance_threshold': acceptance_threshold,
            'pop_size': 100
        }

        print(f"\nTesting: update={update_rate}, elitism={elitism_rate}, accept={acceptance_threshold}")

        result = evaluate_parameters(None, 'MOEAD_Improved', 'fastapi', params, iterations=20)

        if result:
            results.append({'params': params, 'result': result})

            print(f"  Score: {result['score']:.4f}")
            print(f"  Improvement: {result['improvement']*100:.1f}%")
            print(f"  Monotonic rate: {result['monotonic_rate']*100:.1f}%")

            if result['score'] > best_score:
                best_score = result['score']
                best_params = params

    print("\n" + "="*60)
    print("BEST PARAMETERS FOR MOEA/D:")
    print(f"  Update rate: {best_params.get('update_rate')}")
    print(f"  Elitism rate: {best_params.get('elitism_rate')}")
    print(f"  Acceptance threshold: {best_params.get('acceptance_threshold')}")
    print(f"  Score: {best_score:.4f}")

    return best_params, results


def test_convergence_all_algorithms(best_moead_params=None):
    """
    Test convergence for all algorithms with optimized parameters
    """
    print("\n" + "="*60)
    print("TESTING ALL ALGORITHMS WITH OPTIMIZED PARAMETERS")
    print("="*60)

    results = {}

    print("\n1. Testing MOVNS...")
    movns_result = evaluate_parameters(None, 'MOVNS', 'fastapi',
                                      {'archive_size': 100}, iterations=30)
    if movns_result:
        results['MOVNS'] = movns_result
        print(f"  Improvement: {movns_result['improvement']*100:.1f}%")
        print(f"  Monotonic rate: {movns_result['monotonic_rate']*100:.1f}%")

    if best_moead_params:
        print("\n2. Testing MOEA/D Improved...")
        moead_result = evaluate_parameters(None, 'MOEAD_Improved', 'fastapi',
                                          best_moead_params, iterations=30)
        if moead_result:
            results['MOEAD_Improved'] = moead_result
            print(f"  Improvement: {moead_result['improvement']*100:.1f}%")
            print(f"  Monotonic rate: {moead_result['monotonic_rate']*100:.1f}%")

    print("\n3. Testing NSGA-II...")
    nsga2_result = evaluate_parameters(None, 'NSGA2', 'fastapi',
                                      {'pop_size': 100}, iterations=30)
    if nsga2_result:
        results['NSGA2'] = nsga2_result
        print(f"  Improvement: {nsga2_result['improvement']*100:.1f}%")
        print(f"  Monotonic rate: {nsga2_result['monotonic_rate']*100:.1f}%")

    return results


def plot_optimized_convergence(results):
    """
    Plot convergence curves with optimized parameters
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    colors = {'MOVNS': 'blue', 'MOEAD_Improved': 'orange', 'NSGA2': 'green'}

    plt.subplot(2, 2, 1)
    for algo_name, result in results.items():
        if 'hv_history' in result:
            hv = result['hv_history']
            plt.plot(range(len(hv)), hv, label=f"{algo_name} (+{result['improvement']*100:.0f}%)",
                    color=colors.get(algo_name, 'gray'), linewidth=2)

    plt.xlabel('Iteration/Generation')
    plt.ylabel('Hypervolume')
    plt.title('Optimized Convergence - Raw HV')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for algo_name, result in results.items():
        if 'hv_history' in result:
            hv = result['hv_history']
            if len(hv) > 0 and hv[0] > 0:
                normalized = [(v - hv[0]) / hv[0] * 100 for v in hv]
                plt.plot(range(len(normalized)), normalized,
                        label=f"{algo_name} (mono={result['monotonic_rate']*100:.0f}%)",
                        color=colors.get(algo_name, 'gray'), linewidth=2)

    plt.xlabel('Iteration/Generation')
    plt.ylabel('Improvement (%)')
    plt.title('Relative Improvement from Initial')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    algo_names = list(results.keys())
    improvements = [results[a]['improvement']*100 for a in algo_names]
    monotonic_rates = [results[a]['monotonic_rate']*100 for a in algo_names]

    x = np.arange(len(algo_names))
    width = 0.35

    bars1 = plt.bar(x - width/2, improvements, width, label='HV Improvement (%)',
                   color=['blue', 'orange', 'green'][:len(algo_names)], alpha=0.7)
    bars2 = plt.bar(x + width/2, monotonic_rates, width, label='Monotonic Rate (%)',
                   color=['blue', 'orange', 'green'][:len(algo_names)], alpha=0.5)

    plt.xlabel('Algorithm')
    plt.ylabel('Percentage')
    plt.title('Convergence Quality Metrics')
    plt.xticks(x, algo_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    for algo_name, result in results.items():
        if 'hv_history' in result:
            hv = result['hv_history']
            if len(hv) > 1:
                improvements_per_iter = [hv[i] - hv[i-1] for i in range(1, len(hv))]
                cumulative = np.cumsum(improvements_per_iter)
                plt.plot(range(len(cumulative)), cumulative,
                        label=f"{algo_name}",
                        color=colors.get(algo_name, 'gray'), linewidth=2)

    plt.xlabel('Iteration/Generation')
    plt.ylabel('Cumulative HV Gain')
    plt.title('Cumulative Improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Parameter Optimization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../optimized_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_results(best_params, test_results):
    """
    Save optimization results
    """
    results = {
        'best_moead_params': best_params,
        'test_results': test_results,
        'recommendations': {
            'MOEAD': {
                'update_rate': best_params.get('update_rate', 0.25),
                'elitism_rate': best_params.get('elitism_rate', 0.1),
                'acceptance_threshold': best_params.get('acceptance_threshold', 0.98)
            },
            'MOVNS': {
                'archive_size': 100,
                'max_iterations': 50
            },
            'NSGA2': {
                'pop_size': 100,
                'max_gen': 50
            }
        }
    }

    with open('../optimized_parameters.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to optimized_parameters.json")

    return results


def main():
    print("PARAMETER OPTIMIZATION FOR CONVERGENCE")
    print("="*60)

    best_moead_params, grid_results = grid_search_moead()

    test_results = test_convergence_all_algorithms(best_moead_params)

    if test_results:
        plot_optimized_convergence(test_results)

    saved_results = save_results(best_moead_params,
                                {k: {kk: vv for kk, vv in v.items() if kk != 'hv_history'}
                                 for k, v in test_results.items()})

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("\nRecommended parameters saved to optimized_parameters.json")
    print("Convergence plots saved to optimized_convergence.png")

    print("\nSummary of improvements:")
    for algo, result in test_results.items():
        print(f"  {algo}: {result['improvement']*100:.1f}% HV improvement, "
              f"{result['monotonic_rate']*100:.1f}% monotonic")


if __name__ == "__main__":
    main()