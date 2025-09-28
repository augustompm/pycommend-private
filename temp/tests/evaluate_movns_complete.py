"""
Complete evaluation of MOVNS implementation following rules.json
No shortcuts, real execution, comprehensive analysis
"""

import sys
import os
import time
import numpy as np
import json
from datetime import datetime

sys.path.append('E:/pycommend/pycommend-code/src/optimizer')
sys.path.append('E:/pycommend/pycommend-code/src/evaluation')

from movns_vns import MOVNS_VNS
from moead_vns import MOEAD_VNS
from nsga2_vns import NSGA2_VNS


def evaluate_algorithm_real(algorithm_class, package, config):
    """
    Real evaluation without shortcuts
    """
    start_time = time.time()

    if algorithm_class.__name__ == 'MOVNS_VNS':
        algo = algorithm_class(
            package,
            archive_size=config['pop_size'],
            max_iterations=config['max_gen'],
            track_metrics=True
        )
    else:
        algo = algorithm_class(
            package,
            pop_size=config['pop_size'],
            max_gen=config['max_gen'],
            track_metrics=True
        )

    try:
        solutions = algo.run()
        execution_time = time.time() - start_time

        metrics = algo.get_metrics_history()

        result = {
            'algorithm': algorithm_class.__name__,
            'package': package,
            'execution_time': execution_time,
            'num_solutions': len(solutions),
            'solutions': []
        }

        if solutions:
            for i, sol in enumerate(solutions[:5]):
                result['solutions'].append({
                    'packages': sol['packages'][:10],
                    'objectives': sol['objectives']
                })

        if metrics:
            result['metrics'] = {
                'hypervolume': metrics.get('hypervolume', [])[-1] if metrics.get('hypervolume') else None,
                'spacing': metrics.get('spacing', [])[-1] if metrics.get('spacing') else None,
                'spread': metrics.get('spread', [])[-1] if metrics.get('spread') else None,
                'diversity': metrics.get('diversity', [])[-1] if metrics.get('diversity') else None
            }

        return result

    except Exception as e:
        return {
            'algorithm': algorithm_class.__name__,
            'package': package,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def main():
    """
    Main evaluation following rules.json
    """
    print("="*80)
    print("MOVNS COMPLETE EVALUATION - No Shortcuts")
    print(f"Timestamp: {datetime.now()}")
    print("="*80)

    test_packages = ['numpy', 'pandas', 'flask', 'django', 'scikit-learn']

    config = {
        'pop_size': 30,
        'max_gen': 10
    }

    results = {
        'metadata': {
            'timestamp': str(datetime.now()),
            'config': config,
            'packages': test_packages
        },
        'results': {}
    }

    algorithms = [
        ('MOVNS', MOVNS_VNS),
        ('MOEA/D', MOEAD_VNS),
        ('NSGA-II', NSGA2_VNS)
    ]

    for package in test_packages:
        print(f"\nEvaluating package: {package}")
        print("-"*60)

        results['results'][package] = {}

        for algo_name, algo_class in algorithms:
            print(f"\n  Running {algo_name}...")
            result = evaluate_algorithm_real(algo_class, package, config)
            results['results'][package][algo_name] = result

            if 'error' in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Time: {result['execution_time']:.2f}s")
                print(f"    Solutions: {result['num_solutions']}")
                if 'metrics' in result and result['metrics'].get('hypervolume'):
                    print(f"    Hypervolume: {result['metrics']['hypervolume']:.4f}")

    output_file = f"E:/pycommend/temp/tests/movns_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    for package in test_packages:
        print(f"\nPackage: {package}")
        pkg_results = results['results'][package]

        if all(algo in pkg_results for algo in ['MOVNS', 'MOEA/D', 'NSGA-II']):
            movns = pkg_results['MOVNS']
            moead = pkg_results['MOEA/D']
            nsga2 = pkg_results['NSGA-II']

            if 'error' not in movns and 'error' not in moead:
                print(f"  Time comparison:")
                print(f"    MOVNS: {movns['execution_time']:.2f}s")
                print(f"    MOEA/D: {moead['execution_time']:.2f}s")
                print(f"    NSGA-II: {nsga2['execution_time']:.2f}s")

                if movns['execution_time'] < moead['execution_time']:
                    speedup = (moead['execution_time'] / movns['execution_time'] - 1) * 100
                    print(f"    MOVNS is {speedup:.1f}% faster than MOEA/D")

                if 'metrics' in movns and 'metrics' in moead:
                    movns_hv = movns['metrics'].get('hypervolume', 0)
                    moead_hv = moead['metrics'].get('hypervolume', 0)
                    nsga2_hv = nsga2['metrics'].get('hypervolume', 0) if 'metrics' in nsga2 else 0

                    if movns_hv and moead_hv:
                        print(f"  Hypervolume comparison:")
                        print(f"    MOVNS: {movns_hv:.4f}")
                        print(f"    MOEA/D: {moead_hv:.4f}")
                        print(f"    NSGA-II: {nsga2_hv:.4f}")

                        if movns_hv > moead_hv:
                            improvement = (movns_hv / moead_hv - 1) * 100
                            print(f"    MOVNS has {improvement:.1f}% better hypervolume than MOEA/D")

    print(f"\nResults saved to: {output_file}")
    print("\nEvaluation complete - No shortcuts taken")


if __name__ == '__main__':
    main()