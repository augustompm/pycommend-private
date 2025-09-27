"""
Simplified test of quality metrics on just one package
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
from evaluation.quality_metrics import QualityMetrics


def test_single_package(package_name="numpy"):
    """
    Test evaluation for a single package
    """
    print(f"\nTesting package: {package_name}")
    print("="*60)

    pop_size = 20
    max_gen = 10

    # 1. Run NSGA-II
    print("\n1. Running NSGA-II...")
    nsga2 = NSGA2(package_name, pop_size=pop_size, max_gen=max_gen)
    nsga2_solutions = nsga2.run()

    # Extract objectives from NSGA-II
    nsga2_objectives = []
    for sol in nsga2_solutions:
        if isinstance(sol, dict) and 'objectives' in sol:
            obj = sol['objectives']
            if len(obj) == 3:
                nsga2_objectives.append(obj)

    nsga2_objectives = np.array(nsga2_objectives) if nsga2_objectives else np.array([])
    print(f"NSGA-II: {len(nsga2_objectives)} solutions found")

    # 2. Run MOEA/D
    print("\n2. Running MOEA/D...")
    moead = MOEAD(package_name, pop_size=pop_size, n_neighbors=10, max_gen=max_gen)
    moead_solutions = moead.run()

    # Extract objectives from MOEA/D
    moead_objectives = []
    for sol, obj in moead_solutions:
        if len(obj) == 3:
            moead_objectives.append(obj)

    moead_objectives = np.array(moead_objectives) if moead_objectives else np.array([])
    print(f"MOEA/D: {len(moead_objectives)} solutions found")

    # 3. Calculate metrics
    print("\n3. Calculating quality metrics...")
    print("-"*60)

    metrics = QualityMetrics()

    if len(nsga2_objectives) > 0:
        print("\nNSGA-II Metrics:")
        nsga2_metrics = metrics.evaluate_all(nsga2_objectives)
        for key, value in nsga2_metrics.items():
            if key not in ['igd', 'igd_plus']:  # Skip metrics that need reference
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if len(moead_objectives) > 0:
        print("\nMOEA/D Metrics:")
        moead_metrics = metrics.evaluate_all(moead_objectives)
        for key, value in moead_metrics.items():
            if key not in ['igd', 'igd_plus']:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # 4. Compare
    if len(nsga2_objectives) > 0 and len(moead_objectives) > 0:
        print("\n4. Comparison:")
        print("-"*60)

        # Key metrics comparison
        metrics_to_compare = ['hypervolume', 'spacing', 'spread', 'diversity']

        for metric in metrics_to_compare:
            nsga2_val = nsga2_metrics.get(metric, 0)
            moead_val = moead_metrics.get(metric, 0)

            if metric in ['hypervolume', 'diversity']:
                # Higher is better
                winner = "NSGA-II" if nsga2_val > moead_val else "MOEA/D"
            else:
                # Lower is better
                winner = "NSGA-II" if nsga2_val < moead_val else "MOEA/D"

            print(f"{metric:15s}: NSGA-II={nsga2_val:.4f}, MOEA/D={moead_val:.4f} -> {winner}")

    print("\n" + "="*60)
    print("Test completed!")


if __name__ == "__main__":
    test_single_package("numpy")