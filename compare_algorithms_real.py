"""
Real comparison between NSGA-II and MOEA/D for PyCommend VNS
Following rules.json: No shortcuts, no artificial speedups
"""

import sys
import os
import numpy as np
import time
from datetime import datetime

os.chdir('pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from nsga2_vns import NSGA2_VNS
from moead_vns import MOEAD_VNS
from quality_metrics import QualityMetrics


def run_single_test(package_name, algorithm='both'):
    """
    Run a single test with real parameters
    No shortcuts as per rules.json
    """
    print(f"\n{'='*70}")
    print(f"REAL TEST: {package_name.upper()}")
    print(f"Parameters: Population=100, Generations=50")
    print("="*70)

    results = {}

    if algorithm in ['both', 'nsga2']:
        print("\nRunning NSGA-II (this will take time)...")
        start_time = time.time()

        nsga2 = NSGA2_VNS(package_name, pop_size=100, max_gen=50)
        nsga2_solutions = nsga2.run()

        nsga2_time = time.time() - start_time
        print(f"NSGA-II completed in {nsga2_time:.2f} seconds")
        print(f"Found {len(nsga2_solutions)} Pareto-optimal solutions")

        nsga2_objectives = np.array([sol['objectives'] for sol in nsga2_solutions])

        results['nsga2'] = {
            'solutions': nsga2_solutions,
            'objectives': nsga2_objectives,
            'time': nsga2_time,
            'n_solutions': len(nsga2_solutions)
        }

    if algorithm in ['both', 'moead']:
        print("\nRunning MOEA/D (this will take time)...")
        start_time = time.time()

        moead = MOEAD_VNS(package_name, pop_size=100, n_neighbors=20,
                         max_gen=50, decomposition='tchebycheff')
        moead_solutions = moead.run()

        moead_time = time.time() - start_time
        print(f"MOEA/D completed in {moead_time:.2f} seconds")
        print(f"Found {len(moead_solutions)} Pareto-optimal solutions")

        moead_objectives = np.array([sol['objectives'] for sol in moead_solutions])

        results['moead'] = {
            'solutions': moead_solutions,
            'objectives': moead_objectives,
            'time': moead_time,
            'n_solutions': len(moead_solutions)
        }

    return results


def calculate_all_metrics(objectives):
    """
    Calculate all quality metrics
    """
    metrics = QualityMetrics()

    # Filter out solutions with inf values
    valid_mask = ~np.any(np.isinf(objectives), axis=1)
    clean_objectives = objectives[valid_mask]

    if len(clean_objectives) == 0:
        return None

    results = metrics.evaluate_all(clean_objectives)

    # Add epsilon indicator
    def epsilon_indicator_self(objectives):
        """Epsilon indicator against ideal point"""
        if len(objectives) == 0:
            return float('inf')
        ideal = np.min(objectives, axis=0)
        max_diff = np.max(objectives - ideal, axis=1)
        return np.mean(max_diff)

    results['epsilon_indicator'] = epsilon_indicator_self(clean_objectives)

    return results


def print_detailed_comparison(results):
    """
    Print detailed comparison with all metrics
    """
    print("\n" + "="*70)
    print("DETAILED METRICS COMPARISON")
    print("="*70)

    if 'nsga2' in results:
        nsga2_metrics = calculate_all_metrics(results['nsga2']['objectives'])

        print("\nNSGA-II Metrics:")
        print("-"*35)
        if nsga2_metrics:
            print(f"Hypervolume:        {nsga2_metrics['hypervolume']:.4f}")
            print(f"Spacing:            {nsga2_metrics['spacing']:.4f}")
            print(f"Spread (Δ):         {nsga2_metrics['spread']:.4f}")
            print(f"Diversity:          {nsga2_metrics['diversity']:.4f}")
            print(f"Maximum Spread:     {nsga2_metrics['maximum_spread']:.4f}")
            print(f"ε-indicator:        {nsga2_metrics['epsilon_indicator']:.4f}")
            print(f"Non-dominated:      {nsga2_metrics['n_nondominated']}")
            print(f"Execution time:     {results['nsga2']['time']:.2f}s")

    if 'moead' in results:
        moead_metrics = calculate_all_metrics(results['moead']['objectives'])

        print("\nMOEA/D Metrics:")
        print("-"*35)
        if moead_metrics:
            print(f"Hypervolume:        {moead_metrics['hypervolume']:.4f}")
            print(f"Spacing:            {moead_metrics['spacing']:.4f}")
            print(f"Spread (Δ):         {moead_metrics['spread']:.4f}")
            print(f"Diversity:          {moead_metrics['diversity']:.4f}")
            print(f"Maximum Spread:     {moead_metrics['maximum_spread']:.4f}")
            print(f"ε-indicator:        {moead_metrics['epsilon_indicator']:.4f}")
            print(f"Non-dominated:      {moead_metrics['n_nondominated']}")
            print(f"Execution time:     {results['moead']['time']:.2f}s")

    if 'nsga2' in results and 'moead' in results and nsga2_metrics and moead_metrics:
        print("\n" + "="*70)
        print("WINNER BY METRIC")
        print("="*70)

        comparisons = [
            ('Hypervolume', nsga2_metrics['hypervolume'], moead_metrics['hypervolume'], True),
            ('Spacing', nsga2_metrics['spacing'], moead_metrics['spacing'], False),
            ('Spread (Δ)', nsga2_metrics['spread'], moead_metrics['spread'], False),
            ('Diversity', nsga2_metrics['diversity'], moead_metrics['diversity'], True),
            ('ε-indicator', nsga2_metrics['epsilon_indicator'], moead_metrics['epsilon_indicator'], False),
            ('Execution Time', results['nsga2']['time'], results['moead']['time'], False),
        ]

        nsga2_wins = 0
        moead_wins = 0

        for metric_name, nsga2_val, moead_val, higher_better in comparisons:
            if higher_better:
                if nsga2_val > moead_val:
                    winner = "NSGA-II ↑"
                    nsga2_wins += 1
                elif moead_val > nsga2_val:
                    winner = "MOEA/D ↑"
                    moead_wins += 1
                else:
                    winner = "TIE"
            else:
                if nsga2_val < moead_val:
                    winner = "NSGA-II ↓"
                    nsga2_wins += 1
                elif moead_val < nsga2_val:
                    winner = "MOEA/D ↓"
                    moead_wins += 1
                else:
                    winner = "TIE"

            print(f"{metric_name:<20}: {winner}")

        print(f"\nOverall Score: NSGA-II {nsga2_wins} - {moead_wins} MOEA/D")


def show_best_recommendations(results):
    """
    Show best recommendations from each algorithm
    """
    print("\n" + "="*70)
    print("BEST RECOMMENDATIONS")
    print("="*70)

    if 'nsga2' in results:
        nsga2 = NSGA2_VNS('dummy', pop_size=1, max_gen=1)
        nsga2.package_names = nsga2.package_names  # Load package names

        best_nsga2 = min(results['nsga2']['solutions'],
                        key=lambda x: x['objectives'][0])

        indices = np.where(best_nsga2['chromosome'] == 1)[0]
        packages = [nsga2.package_names[idx] for idx in indices]

        print("\nNSGA-II Best Solution:")
        print(f"Packages ({len(packages)}): {', '.join(packages[:10])}")
        print(f"LU: {-best_nsga2['objectives'][0]:.1f}")
        print(f"SS: {-best_nsga2['objectives'][1]:.3f}")
        print(f"RSS: {best_nsga2['objectives'][2]:.1f}")

    if 'moead' in results:
        moead = MOEAD_VNS('dummy', pop_size=1, max_gen=1)
        moead.package_names = moead.package_names  # Load package names

        best_moead = min(results['moead']['solutions'],
                        key=lambda x: x['objectives'][0])

        indices = np.where(best_moead['chromosome'] == 1)[0]
        packages = [moead.package_names[idx] for idx in indices]

        print("\nMOEA/D Best Solution:")
        print(f"Packages ({len(packages)}): {', '.join(packages[:10])}")
        print(f"LU: {-best_moead['objectives'][0]:.1f}")
        print(f"SS: {-best_moead['objectives'][1]:.3f}")
        print(f"RSS: {best_moead['objectives'][2]:.1f}")


def main():
    """
    Main execution following rules.json
    No shortcuts, real execution
    """
    print("="*70)
    print("REAL ALGORITHM COMPARISON")
    print("Following rules.json - No shortcuts")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    print("\nSelect test mode:")
    print("1. Quick test (1 package, both algorithms)")
    print("2. Full test (3 packages, both algorithms)")
    print("3. Single algorithm test")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == '1':
        package = input("Enter package name (e.g., fastapi): ").strip()
        results = run_single_test(package, 'both')
        print_detailed_comparison(results)
        show_best_recommendations(results)

    elif choice == '2':
        packages = ['fastapi', 'scikit-learn', 'prophet']
        for package in packages:
            results = run_single_test(package, 'both')
            print_detailed_comparison(results)
            show_best_recommendations(results)

    elif choice == '3':
        package = input("Enter package name: ").strip()
        algorithm = input("Enter algorithm (nsga2/moead): ").strip()
        results = run_single_test(package, algorithm)
        print_detailed_comparison(results)
        show_best_recommendations(results)

    else:
        print("Invalid choice")
        return

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == '__main__':
    # Check if running in background
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto mode for testing
        results = run_single_test('fastapi', 'both')
        print_detailed_comparison(results)
        show_best_recommendations(results)
    else:
        main()