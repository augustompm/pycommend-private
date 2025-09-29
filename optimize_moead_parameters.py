"""
Parameter optimization for MOEA/D-VNS Normalized
Find optimal parameters for best convergence
"""

import numpy as np
import sys
import os
from itertools import product
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
from optimizer.moead_vns_normalized import MOEAD_VNS_Normalized


def test_parameters(params_dict, test_package='fastapi', generations=20, verbose=False):
    """
    Test a specific parameter configuration
    """
    try:
        moead = MOEAD_VNS_Normalized(
            test_package,
            pop_size=params_dict['pop_size'],
            n_neighbors=params_dict['n_neighbors'],
            max_gen=generations,
            decomposition=params_dict['decomposition'],
            theta=params_dict['theta'],
            track_metrics=True
        )

        start_time = time.time()
        solutions = moead.run()
        exec_time = time.time() - start_time

        metrics = moead.get_metrics_history()

        if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 1:
            hv = metrics['hypervolume']
            initial = hv[0] if hv[0] > 0 else 0.001
            final = hv[-1]
            improvement = (final - initial) / initial * 100

            monotonic = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
            monotonic_rate = monotonic / (len(hv) - 1) * 100

            result = {
                'params': params_dict,
                'improvement': improvement,
                'final_hv': final,
                'monotonic_rate': monotonic_rate,
                'n_solutions': len(solutions),
                'exec_time': exec_time,
                'success': improvement > 0
            }

            if verbose:
                print(f"  HV Improvement: {improvement:+.1f}%")
                print(f"  Monotonic Rate: {monotonic_rate:.1f}%")
                print(f"  Solutions: {len(solutions)}, Time: {exec_time:.1f}s")

            return result

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return {'params': params_dict, 'improvement': -100, 'success': False}

    return {'params': params_dict, 'improvement': 0, 'success': False}


def grid_search():
    """
    Perform grid search for optimal parameters
    """
    print("MOEA/D PARAMETER OPTIMIZATION")
    print("="*60)

    param_grid = {
        'pop_size': [50, 100],
        'n_neighbors': [10, 20, 30],
        'decomposition': ['tchebycheff', 'weighted'],
        'theta': [2.0, 5.0, 10.0]
    }

    param_combinations = list(product(
        param_grid['pop_size'],
        param_grid['n_neighbors'],
        param_grid['decomposition'],
        param_grid['theta']
    ))

    print(f"Testing {len(param_combinations)} parameter combinations...")
    print("="*60)

    results = []

    for i, (pop_size, n_neighbors, decomposition, theta) in enumerate(param_combinations, 1):
        params = {
            'pop_size': pop_size,
            'n_neighbors': min(n_neighbors, pop_size - 1),
            'decomposition': decomposition,
            'theta': theta
        }

        print(f"\n[{i}/{len(param_combinations)}] Testing:")
        print(f"  pop_size={pop_size}, n_neighbors={n_neighbors}")
        print(f"  decomposition={decomposition}, theta={theta}")

        result = test_parameters(params, generations=15, verbose=True)
        results.append(result)

        if result['success']:
            print(f"  STATUS: CONVERGING ✓")
        else:
            print(f"  STATUS: DIVERGING ✗")

    results.sort(key=lambda x: x['improvement'], reverse=True)

    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)

    for i, result in enumerate(results[:5], 1):
        p = result['params']
        print(f"\n{i}. Improvement: {result['improvement']:+.1f}%")
        print(f"   pop_size={p['pop_size']}, n_neighbors={p['n_neighbors']}")
        print(f"   decomposition={p['decomposition']}, theta={p['theta']}")
        print(f"   Final HV: {result.get('final_hv', 0):.4f}")
        print(f"   Monotonic: {result.get('monotonic_rate', 0):.1f}%")
        print(f"   Time: {result.get('exec_time', 0):.1f}s")

    return results


def quick_test():
    """
    Quick test with default and optimized parameters
    """
    print("QUICK PARAMETER TEST")
    print("="*60)

    configs = [
        {
            'name': 'Default',
            'params': {
                'pop_size': 100,
                'n_neighbors': 20,
                'decomposition': 'tchebycheff',
                'theta': 5.0
            }
        },
        {
            'name': 'Optimized-1',
            'params': {
                'pop_size': 50,
                'n_neighbors': 10,
                'decomposition': 'tchebycheff',
                'theta': 2.0
            }
        },
        {
            'name': 'Optimized-2',
            'params': {
                'pop_size': 100,
                'n_neighbors': 30,
                'decomposition': 'weighted',
                'theta': 10.0
            }
        }
    ]

    for config in configs:
        print(f"\nTesting {config['name']}:")
        print("-"*40)
        result = test_parameters(config['params'], generations=20, verbose=True)

        if result['success']:
            print(f"RESULT: CONVERGING ({result['improvement']:+.1f}%)")
        else:
            print(f"RESULT: DIVERGING ({result['improvement']:+.1f}%)")


def main():
    """
    Main optimization function
    """
    import argparse

    parser = argparse.ArgumentParser(description='Optimize MOEA/D parameters')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--full', action='store_true', help='Run full grid search')

    args = parser.parse_args()

    if args.full:
        results = grid_search()

        with open('moead_optimization_results.txt', 'w') as f:
            f.write("MOEA/D Parameter Optimization Results\n")
            f.write("="*60 + "\n\n")

            for result in results[:10]:
                p = result['params']
                f.write(f"Improvement: {result['improvement']:+.1f}%\n")
                f.write(f"  pop_size={p['pop_size']}, n_neighbors={p['n_neighbors']}\n")
                f.write(f"  decomposition={p['decomposition']}, theta={p['theta']}\n")
                f.write(f"  Final HV: {result.get('final_hv', 0):.4f}\n")
                f.write(f"  Monotonic: {result.get('monotonic_rate', 0):.1f}%\n")
                f.write("\n")

        print(f"\nResults saved to moead_optimization_results.txt")

    else:
        quick_test()


if __name__ == "__main__":
    main()