"""
Compare NSGA-II and MOEA/D performance
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizer.nsga2 import NSGA2
from optimizer.moead import MOEAD


def run_comparison(package_name, pop_size=50, generations=30):
    """Run both algorithms and compare results"""

    results = {}

    print(f"\n{'='*60}")
    print(f"Comparing algorithms for package: {package_name}")
    print(f"Population: {pop_size}, Generations: {generations}")
    print('='*60)

    # Run NSGA-II
    print("\n[NSGA-II] Starting...")
    start_time = time.time()
    try:
        nsga2 = NSGA2(package_name, pop_size=pop_size, max_gen=generations)
        nsga2_solutions = nsga2.run()
        nsga2_time = time.time() - start_time
        nsga2_df = nsga2.format_results(nsga2_solutions)

        results['nsga2'] = {
            'time': nsga2_time,
            'n_solutions': len(nsga2_solutions),
            'n_unique': len(nsga2_df),
            'avg_f1': nsga2_df['linked_usage'].mean(),
            'avg_f2': nsga2_df['semantic_similarity'].mean(),
            'avg_f3': nsga2_df['size'].mean(),
            'best_f1': nsga2_df['linked_usage'].max(),
            'best_f2': nsga2_df['semantic_similarity'].max(),
            'min_f3': nsga2_df['size'].min()
        }
        print(f"[NSGA-II] Completed in {nsga2_time:.2f}s")
        print(f"[NSGA-II] Solutions: {len(nsga2_solutions)} total, {len(nsga2_df)} unique")

    except Exception as e:
        print(f"[NSGA-II] Error: {e}")
        results['nsga2'] = None

    # Run MOEA/D
    print("\n[MOEA/D] Starting...")
    start_time = time.time()
    try:
        moead = MOEAD(package_name, pop_size=pop_size, n_neighbors=15, max_gen=generations)
        moead_solutions = moead.run()
        moead_time = time.time() - start_time
        moead_df = moead.format_results(moead_solutions)

        results['moead'] = {
            'time': moead_time,
            'n_solutions': len(moead_solutions),
            'n_unique': len(moead_df),
            'avg_f1': moead_df['linked_usage'].mean(),
            'avg_f2': moead_df['semantic_similarity'].mean(),
            'avg_f3': moead_df['size'].mean(),
            'best_f1': moead_df['linked_usage'].max(),
            'best_f2': moead_df['semantic_similarity'].max(),
            'min_f3': moead_df['size'].min()
        }
        print(f"[MOEA/D] Completed in {moead_time:.2f}s")
        print(f"[MOEA/D] Solutions: {len(moead_solutions)} total, {len(moead_df)} unique")

    except Exception as e:
        print(f"[MOEA/D] Error: {e}")
        results['moead'] = None

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    if results['nsga2'] and results['moead']:
        # Time comparison
        time_diff = results['moead']['time'] - results['nsga2']['time']
        time_pct = (results['moead']['time'] / results['nsga2']['time'] - 1) * 100

        print(f"\nExecution Time:")
        print(f"  NSGA-II: {results['nsga2']['time']:.2f}s")
        print(f"  MOEA/D:  {results['moead']['time']:.2f}s")
        print(f"  Difference: {time_diff:+.2f}s ({time_pct:+.1f}%)")

        # Solution diversity
        print(f"\nSolution Diversity:")
        print(f"  NSGA-II: {results['nsga2']['n_unique']} unique solutions")
        print(f"  MOEA/D:  {results['moead']['n_unique']} unique solutions")

        # Objective quality
        print(f"\nObjective Quality (Averages):")
        print(f"  F1 (Linked Usage):")
        print(f"    NSGA-II: {results['nsga2']['avg_f1']:.2f}")
        print(f"    MOEA/D:  {results['moead']['avg_f1']:.2f}")

        print(f"  F2 (Semantic Similarity):")
        print(f"    NSGA-II: {results['nsga2']['avg_f2']:.4f}")
        print(f"    MOEA/D:  {results['moead']['avg_f2']:.4f}")

        print(f"  F3 (Set Size):")
        print(f"    NSGA-II: {results['nsga2']['avg_f3']:.2f}")
        print(f"    MOEA/D:  {results['moead']['avg_f3']:.2f}")

        # Best solutions
        print(f"\nBest Solutions:")
        print(f"  Max F1: NSGA-II={results['nsga2']['best_f1']:.2f}, MOEA/D={results['moead']['best_f1']:.2f}")
        print(f"  Max F2: NSGA-II={results['nsga2']['best_f2']:.4f}, MOEA/D={results['moead']['best_f2']:.4f}")
        print(f"  Min F3: NSGA-II={results['nsga2']['min_f3']:.0f}, MOEA/D={results['moead']['min_f3']:.0f}")

        # Winner determination
        print("\n" + "="*60)
        print("WINNER BY METRIC:")

        winners = {}
        if results['moead']['time'] < results['nsga2']['time']:
            winners['speed'] = 'MOEA/D'
        else:
            winners['speed'] = 'NSGA-II'

        if results['moead']['n_unique'] > results['nsga2']['n_unique']:
            winners['diversity'] = 'MOEA/D'
        else:
            winners['diversity'] = 'NSGA-II'

        if results['moead']['best_f1'] > results['nsga2']['best_f1']:
            winners['f1'] = 'MOEA/D'
        else:
            winners['f1'] = 'NSGA-II'

        if results['moead']['best_f2'] > results['nsga2']['best_f2']:
            winners['f2'] = 'MOEA/D'
        else:
            winners['f2'] = 'NSGA-II'

        print(f"  Speed: {winners['speed']}")
        print(f"  Diversity: {winners['diversity']}")
        print(f"  Best F1: {winners['f1']}")
        print(f"  Best F2: {winners['f2']}")

        # Overall winner
        moead_wins = sum(1 for w in winners.values() if w == 'MOEA/D')
        nsga2_wins = sum(1 for w in winners.values() if w == 'NSGA-II')

        print(f"\nOVERALL: ", end="")
        if moead_wins > nsga2_wins:
            print(f"MOEA/D wins ({moead_wins}-{nsga2_wins})")
        elif nsga2_wins > moead_wins:
            print(f"NSGA-II wins ({nsga2_wins}-{moead_wins})")
        else:
            print(f"TIE ({moead_wins}-{nsga2_wins})")

    return results


def main():
    """Run comparisons for multiple packages"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare NSGA-II and MOEA/D')
    parser.add_argument('--packages', nargs='+', default=['numpy', 'pandas', 'requests'],
                       help='Packages to test')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=30, help='Generations')

    args = parser.parse_args()

    all_results = {}

    for package in args.packages:
        results = run_comparison(package, args.pop_size, args.generations)
        all_results[package] = results

    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    for package, results in all_results.items():
        if results['nsga2'] and results['moead']:
            print(f"\n{package}:")
            print(f"  NSGA-II: {results['nsga2']['time']:.2f}s, {results['nsga2']['n_unique']} solutions")
            print(f"  MOEA/D:  {results['moead']['time']:.2f}s, {results['moead']['n_unique']} solutions")

    return 0


if __name__ == "__main__":
    main()