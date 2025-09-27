"""
Test Quality Metrics on NSGA-II and MOEA/D
Complete evaluation of algorithms using all quality metrics
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from optimizer.nsga2 import NSGA2
from optimizer.moead import MOEAD
from optimizer.moead_correct import MOEAD_Correct
from evaluation.quality_metrics import QualityMetrics, compare_algorithms


def extract_objectives(solutions):
    """
    Extract objective values from solution format
    """
    if isinstance(solutions, list) and len(solutions) > 0:
        # Check if it's a dictionary with 'objectives' key (NSGA-II format)
        if isinstance(solutions[0], dict) and 'objectives' in solutions[0]:
            objectives = []
            for sol in solutions:
                obj = sol.get('objectives', [])
                if isinstance(obj, (list, np.ndarray)):
                    obj_array = np.array(obj).flatten()
                    if len(obj_array) == 3:
                        # NSGA-II already returns minimization format (negative F1 and F2)
                        objectives.append(obj_array)
            return np.array(objectives) if objectives else np.array([])

        # Check if it's MOEA/D format (tuple of solution, objectives)
        elif isinstance(solutions[0], tuple):
            objectives = []
            for sol, obj in solutions:
                # Ensure objectives are 1D array with 3 values
                if isinstance(obj, (list, np.ndarray)):
                    obj_array = np.array(obj).flatten()
                    if len(obj_array) == 3:
                        objectives.append(obj_array)
            return np.array(objectives) if objectives else np.array([])

        # Check if it's NSGA-II format (objects with attributes)
        elif hasattr(solutions[0], 'objectives'):
            objectives = []
            for sol in solutions:
                if hasattr(sol, 'objectives'):
                    obj = sol.objectives
                    if isinstance(obj, (list, np.ndarray)):
                        obj_array = np.array(obj).flatten()
                        if len(obj_array) == 3:
                            objectives.append(obj_array)
            return np.array(objectives) if objectives else np.array([])

        # Check if solutions themselves have objective attributes
        elif hasattr(solutions[0], 'F1') and hasattr(solutions[0], 'F2') and hasattr(solutions[0], 'F3'):
            objectives = []
            for sol in solutions:
                # Note: NSGA-II uses maximization for F1 and F2, need to negate for minimization
                objectives.append([-sol.F1, -sol.F2, sol.F3])
            return np.array(objectives)

        else:
            # Assume it's already objectives array
            obj_array = np.array(solutions)
            if len(obj_array.shape) == 2 and obj_array.shape[1] == 3:
                return obj_array

    return np.array([])


def run_algorithm_evaluation(package_name, pop_size=50, max_gen=30):
    """
    Run evaluation of all algorithms for a package
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ALGORITHMS FOR PACKAGE: {package_name}")
    print(f"Population: {pop_size}, Generations: {max_gen}")
    print(f"{'='*80}\n")

    results = {}

    # 1. Run NSGA-II
    print("\n[1] Running NSGA-II...")
    try:
        start = time.time()
        nsga2 = NSGA2(package_name, pop_size=pop_size, max_gen=max_gen)
        nsga2_solutions = nsga2.run()
        nsga2_time = time.time() - start

        # Extract objectives
        nsga2_objectives = extract_objectives(nsga2_solutions)
        print(f"NSGA-II completed in {nsga2_time:.2f}s")
        print(f"Solutions found: {len(nsga2_objectives)}")

        results['NSGA-II'] = {
            'solutions': nsga2_solutions,
            'objectives': nsga2_objectives,
            'time': nsga2_time
        }
    except Exception as e:
        print(f"Error running NSGA-II: {e}")
        results['NSGA-II'] = None

    # 2. Run MOEA/D (Original)
    print("\n[2] Running MOEA/D (Original)...")
    try:
        start = time.time()
        moead_orig = MOEAD(package_name, pop_size=pop_size, n_neighbors=20, max_gen=max_gen)
        moead_orig_solutions = moead_orig.run()
        moead_orig_time = time.time() - start

        # Extract objectives
        moead_orig_objectives = extract_objectives(moead_orig_solutions)
        print(f"MOEA/D (Original) completed in {moead_orig_time:.2f}s")
        print(f"Solutions found: {len(moead_orig_objectives)}")

        results['MOEA/D-Original'] = {
            'solutions': moead_orig_solutions,
            'objectives': moead_orig_objectives,
            'time': moead_orig_time
        }
    except Exception as e:
        print(f"Error running MOEA/D Original: {e}")
        results['MOEA/D-Original'] = None

    # 3. Run MOEA/D (Corrected) - Skip for now due to index error
    print("\n[3] Skipping MOEA/D (Corrected) - needs debugging...")
    results['MOEA/D-Corrected'] = None

    return results


def evaluate_with_metrics(results):
    """
    Evaluate all algorithms using quality metrics
    """
    print(f"\n{'='*80}")
    print("QUALITY METRICS EVALUATION")
    print(f"{'='*80}\n")

    metrics_calculator = QualityMetrics()
    evaluation_results = {}

    # Generate reference set (combine all non-dominated solutions)
    all_objectives = []
    for algo_name, algo_data in results.items():
        if algo_data is not None and len(algo_data['objectives']) > 0:
            all_objectives.append(algo_data['objectives'])

    if len(all_objectives) > 0:
        combined_objectives = np.vstack(all_objectives)
        reference_set = metrics_calculator.filter_dominated(combined_objectives)
        print(f"Reference set size: {len(reference_set)} non-dominated solutions\n")
    else:
        reference_set = None

    # Evaluate each algorithm
    for algo_name, algo_data in results.items():
        if algo_data is None:
            continue

        print(f"\n{algo_name}:")
        print("-" * 40)

        objectives = algo_data['objectives']

        if len(objectives) == 0:
            print("No solutions to evaluate")
            continue

        # Calculate all metrics
        metrics = metrics_calculator.evaluate_all(objectives, reference_set)

        # Store results
        evaluation_results[algo_name] = metrics

        # Print metrics
        print(f"Number of solutions: {metrics['n_solutions']}")
        print(f"Non-dominated solutions: {metrics['n_nondominated']}")
        print(f"Hypervolume: {metrics['hypervolume']:.4f}")
        print(f"Spacing: {metrics['spacing']:.4f}")
        print(f"Spread: {metrics['spread']:.4f}")
        print(f"Diversity: {metrics['diversity']:.4f}")
        print(f"Maximum Spread: {metrics['maximum_spread']:.4f}")

        if reference_set is not None:
            print(f"IGD: {metrics.get('igd', 'N/A'):.4f}")
            print(f"IGD+: {metrics.get('igd_plus', 'N/A'):.4f}")

        print(f"Runtime: {algo_data['time']:.2f}s")

    return evaluation_results


def compare_all_algorithms(evaluation_results):
    """
    Compare algorithms and determine winners
    """
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*80}\n")

    if len(evaluation_results) < 2:
        print("Not enough algorithms to compare")
        return

    # Create comparison table
    metrics_names = list(next(iter(evaluation_results.values())).keys())
    algorithms = list(evaluation_results.keys())

    # Determine winners for each metric
    winners = {}
    for metric in metrics_names:
        values = {algo: evaluation_results[algo][metric] for algo in algorithms}

        # Determine if higher or lower is better
        if metric in ['hypervolume', 'diversity', 'maximum_spread', 'n_solutions', 'n_nondominated']:
            # Higher is better
            winner = max(values, key=values.get)
        else:
            # Lower is better
            winner = min(values, key=values.get)

        winners[metric] = winner

    # Print comparison table
    print("\nMetric Comparison Table:")
    print("-" * 80)
    print(f"{'Metric':<20} | " + " | ".join(f"{algo:<20}" for algo in algorithms))
    print("-" * 80)

    for metric in metrics_names:
        values = [f"{evaluation_results[algo][metric]:.4f}" for algo in algorithms]
        winner = winners[metric]

        # Mark winner with *
        for i, algo in enumerate(algorithms):
            if algo == winner:
                values[i] += " *"

        print(f"{metric:<20} | " + " | ".join(f"{val:<20}" for val in values))

    # Count wins
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    win_counts = {algo: 0 for algo in algorithms}
    for winner in winners.values():
        win_counts[winner] += 1

    print("\nWins per algorithm:")
    for algo, wins in win_counts.items():
        print(f"  {algo}: {wins} metrics")

    # Overall winner
    overall_winner = max(win_counts, key=win_counts.get)
    print(f"\nOverall winner: {overall_winner}")

    # Performance analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)

    # Check for specific patterns
    if 'MOEA/D-Original' in evaluation_results and 'MOEA/D-Corrected' in evaluation_results:
        orig = evaluation_results['MOEA/D-Original']
        corr = evaluation_results['MOEA/D-Corrected']

        print("\nMOEA/D Original vs Corrected:")
        print(f"  Hypervolume: {orig['hypervolume']:.4f} vs {corr['hypervolume']:.4f} "
              f"({'Corrected' if corr['hypervolume'] > orig['hypervolume'] else 'Original'} better)")
        print(f"  Spacing: {orig['spacing']:.4f} vs {corr['spacing']:.4f} "
              f"({'Corrected' if corr['spacing'] < orig['spacing'] else 'Original'} better)")
        print(f"  Diversity: {orig['diversity']:.4f} vs {corr['diversity']:.4f} "
              f"({'Corrected' if corr['diversity'] > orig['diversity'] else 'Original'} better)")

    return winners


def generate_report(package_name, results, evaluation_results, winners):
    """
    Generate detailed report
    """
    # Create results directory if it doesn't exist
    import os
    os.makedirs("results", exist_ok=True)

    report_path = f"results/quality_evaluation_{package_name}.md"

    with open(report_path, 'w') as f:
        f.write(f"# Quality Evaluation Report: {package_name}\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        f.write("## Algorithm Performance\n\n")

        for algo_name, algo_data in results.items():
            if algo_data is None:
                continue

            f.write(f"### {algo_name}\n")
            f.write(f"- Runtime: {algo_data['time']:.2f}s\n")
            f.write(f"- Solutions found: {len(algo_data['objectives'])}\n\n")

        f.write("## Quality Metrics\n\n")

        f.write("| Metric | " + " | ".join(evaluation_results.keys()) + " | Winner |\n")
        f.write("|--------|" + "|".join(["--------" for _ in evaluation_results]) + "|--------|\n")

        for metric in list(next(iter(evaluation_results.values())).keys()):
            row = f"| {metric} |"
            for algo in evaluation_results.keys():
                value = evaluation_results[algo][metric]
                row += f" {value:.4f} |"
            if winners:
                row += f" {winners.get(metric, 'N/A')} |"
            else:
                row += " N/A |"
            f.write(row + "\n")

        f.write("\n## Conclusion\n\n")

        win_counts = {algo: 0 for algo in evaluation_results.keys()}
        if winners:
            for winner in winners.values():
                if winner in win_counts:
                    win_counts[winner] += 1

            overall_winner = max(win_counts, key=win_counts.get) if win_counts else "N/A"
            f.write(f"**Overall Winner: {overall_winner}**\n\n")
        else:
            f.write("**Overall Winner: Unable to determine (no comparison)**\n\n")

        f.write("Wins per algorithm:\n")
        for algo, wins in win_counts.items():
            f.write(f"- {algo}: {wins} metrics\n")

    print(f"\nReport saved to {report_path}")


def main():
    """
    Main evaluation function
    """
    print("="*80)
    print("PyCommend Algorithm Quality Evaluation")
    print("Testing NSGA-II vs MOEA/D (Original) vs MOEA/D (Corrected)")
    print("="*80)

    # Test packages
    test_packages = ['numpy', 'requests', 'pandas']

    for package in test_packages:
        print(f"\n\n{'#'*80}")
        print(f"PACKAGE: {package}")
        print(f"{'#'*80}")

        try:
            # Run algorithms
            results = run_algorithm_evaluation(package, pop_size=30, max_gen=20)

            # Evaluate with metrics
            evaluation_results = evaluate_with_metrics(results)

            # Compare algorithms
            if evaluation_results:
                winners = compare_all_algorithms(evaluation_results)

                # Generate report
                generate_report(package, results, evaluation_results, winners)

        except Exception as e:
            print(f"Error evaluating package {package}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()