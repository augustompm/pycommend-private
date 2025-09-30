"""
Test MOVNS Advanced vs MOEA/D
Extended runtime, no simplifications, aggressive optimization
"""

import numpy as np
import sys
import os
import time
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized


def run_comparison(package_name: str = 'fastapi', extended: bool = True) -> Dict:
    """
    Run comprehensive comparison of algorithms
    """
    results = {}

    print("="*70)
    print("MOVNS ADVANCED VS MOEA/D COMPARISON")
    print("="*70)
    print(f"Testing package: {package_name}")
    print(f"Extended runtime: {extended}")
    print()

    if extended:
        movns_iterations = 100
        movns_archive = 200
        moead_generations = 100
        moead_pop_size = 200
    else:
        movns_iterations = 30
        movns_archive = 100
        moead_generations = 30
        moead_pop_size = 100

    print("-"*70)
    print("1. Testing MOVNS Advanced (Aggressive Multi-Method)")
    print("-"*70)

    algo = MOVNS_Advanced(
        package_name,
        archive_size=movns_archive,
        max_iterations=movns_iterations,
        track_metrics=True
    )

    start_time = time.time()
    solutions = algo.run()
    advanced_time = time.time() - start_time

    metrics = algo.get_metrics_history()
    advanced_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

    best_lu = 0
    best_ss = 0
    best_rss = float('inf')
    for sol in solutions:
        obj = algo.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_lu:
            best_lu = -obj[0]
        if -obj[1] > best_ss:
            best_ss = -obj[1]
        if obj[2] < best_rss:
            best_rss = obj[2]

    results['movns_advanced'] = {
        'hypervolume': advanced_hv,
        'time': advanced_time,
        'solutions': len(solutions),
        'best_lu': best_lu,
        'best_ss': best_ss,
        'best_rss': best_rss,
        'hv_history': metrics.get('hypervolume', [])
    }

    print(f"Results:")
    print(f"  Hypervolume: {advanced_hv:.4f}")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {advanced_time:.1f}s")
    print(f"  Best LU: {best_lu:.0f}")
    print(f"  Best SS: {best_ss:.4f}")
    print(f"  Best RSS: {best_rss}")
    print()

    print("-"*70)
    print("2. Testing MOVNS v2 (Baseline)")
    print("-"*70)

    algo = MOVNS_V2(
        package_name,
        archive_size=movns_archive//2,
        max_iterations=movns_iterations//2,
        track_metrics=True
    )

    start_time = time.time()
    solutions = algo.run()
    v2_time = time.time() - start_time

    metrics = algo.get_metrics_history()
    v2_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

    results['movns_v2'] = {
        'hypervolume': v2_hv,
        'time': v2_time,
        'solutions': len(solutions),
        'hv_history': metrics.get('hypervolume', [])
    }

    print(f"Results:")
    print(f"  Hypervolume: {v2_hv:.4f}")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {v2_time:.1f}s")
    print()

    print("-"*70)
    print("3. Testing MOEA/D (Target to beat)")
    print("-"*70)

    algo = MOEAD_Normalized(
        package_name,
        pop_size=moead_pop_size,
        max_gen=moead_generations
    )

    start_time = time.time()
    solutions = algo.run()
    moead_time = time.time() - start_time

    metrics = algo.get_metrics_history()
    moead_hv = metrics['hypervolume'][-1] if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0 else 0

    best_lu = 0
    best_ss = 0
    best_rss = float('inf')
    for sol in solutions:
        obj = algo.evaluate_objectives(sol['chromosome'])
        if -obj[0] > best_lu:
            best_lu = -obj[0]
        if -obj[1] > best_ss:
            best_ss = -obj[1]
        if obj[2] < best_rss:
            best_rss = obj[2]

    results['moead'] = {
        'hypervolume': moead_hv,
        'time': moead_time,
        'solutions': len(solutions),
        'best_lu': best_lu,
        'best_ss': best_ss,
        'best_rss': best_rss,
        'hv_history': metrics.get('hypervolume', [])
    }

    print(f"Results:")
    print(f"  Hypervolume: {moead_hv:.4f}")
    print(f"  Solutions: {len(solutions)}")
    print(f"  Time: {moead_time:.1f}s")
    print(f"  Best LU: {best_lu:.0f}")
    print(f"  Best SS: {best_ss:.4f}")
    print(f"  Best RSS: {best_rss}")
    print()

    print("="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print(f"\nHypervolume Performance:")
    print(f"  MOVNS Advanced: {results['movns_advanced']['hypervolume']:.4f}")
    print(f"  MOVNS v2:       {results['movns_v2']['hypervolume']:.4f}")
    print(f"  MOEA/D:         {results['moead']['hypervolume']:.4f}")

    if results['moead']['hypervolume'] > 0:
        advanced_ratio = results['movns_advanced']['hypervolume'] / results['moead']['hypervolume']
        v2_ratio = results['movns_v2']['hypervolume'] / results['moead']['hypervolume']

        print(f"\nRelative Performance:")
        print(f"  MOVNS Advanced vs MOEA/D: {advanced_ratio*100:.1f}%")
        print(f"  MOVNS v2 vs MOEA/D:       {v2_ratio*100:.1f}%")

        if advanced_ratio > 1.0:
            print(f"\n✓ SUCCESS: MOVNS Advanced beats MOEA/D by {(advanced_ratio-1)*100:.1f}%")
        elif advanced_ratio > 0.95:
            print(f"\n○ COMPETITIVE: MOVNS Advanced at {advanced_ratio*100:.1f}% of MOEA/D")
        else:
            print(f"\n✗ MOEA/D still superior by {(1-advanced_ratio)*100:.1f}%")

    if results['movns_v2']['hypervolume'] > 0:
        improvement = results['movns_advanced']['hypervolume'] / results['movns_v2']['hypervolume']
        print(f"\nMOVNS Advanced vs v2: {improvement*100:.1f}% ({(improvement-1)*100:+.1f}%)")

    print(f"\nBest Objectives Comparison:")
    print(f"  Linked Usage:")
    print(f"    MOVNS Advanced: {results['movns_advanced']['best_lu']:.0f}")
    print(f"    MOEA/D:         {results['moead']['best_lu']:.0f}")
    print(f"  Semantic Similarity:")
    print(f"    MOVNS Advanced: {results['movns_advanced']['best_ss']:.4f}")
    print(f"    MOEA/D:         {results['moead']['best_ss']:.4f}")
    print(f"  Set Size:")
    print(f"    MOVNS Advanced: {results['movns_advanced']['best_rss']}")
    print(f"    MOEA/D:         {results['moead']['best_rss']}")

    print(f"\nExecution Time:")
    print(f"  MOVNS Advanced: {results['movns_advanced']['time']:.1f}s")
    print(f"  MOVNS v2:       {results['movns_v2']['time']:.1f}s")
    print(f"  MOEA/D:         {results['moead']['time']:.1f}s")

    print(f"\nArchive/Population Size:")
    print(f"  MOVNS Advanced: {results['movns_advanced']['solutions']}")
    print(f"  MOVNS v2:       {results['movns_v2']['solutions']}")
    print(f"  MOEA/D:         {results['moead']['solutions']}")

    return results


def test_multiple_packages(packages: List[str] = None, extended: bool = False):
    """
    Test on multiple packages
    """
    if packages is None:
        packages = ['fastapi', 'scikit-learn', 'prophet']

    all_results = {}

    for package in packages:
        print(f"\n{'='*70}")
        print(f"Testing package: {package.upper()}")
        print(f"{'='*70}\n")

        try:
            results = run_comparison(package, extended=extended)
            all_results[package] = results
        except Exception as e:
            print(f"Error testing {package}: {e}")
            continue

    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL PACKAGES")
    print(f"{'='*70}\n")

    movns_wins = 0
    moead_wins = 0

    for package, results in all_results.items():
        print(f"\n{package}:")
        adv_hv = results['movns_advanced']['hypervolume']
        moead_hv = results['moead']['hypervolume']

        if moead_hv > 0:
            ratio = adv_hv / moead_hv
            print(f"  MOVNS Advanced: {adv_hv:.4f}")
            print(f"  MOEA/D:         {moead_hv:.4f}")
            print(f"  Ratio:          {ratio*100:.1f}%")

            if adv_hv > moead_hv:
                print(f"  Winner:         MOVNS Advanced (+{(ratio-1)*100:.1f}%)")
                movns_wins += 1
            else:
                print(f"  Winner:         MOEA/D (+{(1-ratio)*100:.1f}%)")
                moead_wins += 1

    print(f"\nOverall Score: MOVNS Advanced {movns_wins} - {moead_wins} MOEA/D")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=str, default='fastapi',
                       help='Package to test')
    parser.add_argument('--extended', action='store_true',
                       help='Use extended runtime (100 iterations)')
    parser.add_argument('--multiple', action='store_true',
                       help='Test multiple packages')

    args = parser.parse_args()

    if args.multiple:
        test_multiple_packages(extended=args.extended)
    else:
        run_comparison(args.package, extended=args.extended)