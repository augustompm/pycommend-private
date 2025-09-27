"""
Test script for updated MOEA/D implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimizer.moead import MOEAD
from evaluation.quality_metrics import QualityMetrics
import numpy as np
import matplotlib.pyplot as plt


def test_moead_implementation():
    """Test updated MOEA/D with different decomposition methods"""

    package_name = "numpy"

    print("=" * 80)
    print("Testing Updated MOEA/D Implementation")
    print("=" * 80)

    # Test configurations
    configs = [
        {"decomposition": "tchebycheff", "name": "Tchebycheff"},
        {"decomposition": "weighted_sum", "name": "Weighted Sum"},
        {"decomposition": "pbi", "name": "PBI"}
    ]

    results_by_method = {}

    for config in configs:
        print(f"\n\nTesting with {config['name']} decomposition...")
        print("-" * 40)

        try:
            # Run MOEA/D with specific decomposition
            moead = MOEAD(
                package_name=package_name,
                pop_size=50,
                n_neighbors=10,
                max_gen=30,
                decomposition=config["decomposition"],
                cr=1.0,
                f_scale=0.5,
                eta_m=20
            )

            # Run optimization
            solutions = moead.run()

            # Extract objectives
            objectives = np.array([obj for _, obj in solutions])

            # Convert back to maximization for F1 and F2 for analysis
            objectives_display = objectives.copy()
            objectives_display[:, 0] = -objectives_display[:, 0]  # F1
            objectives_display[:, 1] = -objectives_display[:, 1]  # F2

            results_by_method[config["name"]] = {
                "solutions": solutions,
                "objectives": objectives,
                "objectives_display": objectives_display
            }

            print(f"\nResults for {config['name']}:")
            print(f"Number of solutions: {len(solutions)}")
            print(f"F1 range: [{objectives_display[:, 0].min():.4f}, {objectives_display[:, 0].max():.4f}]")
            print(f"F2 range: [{objectives_display[:, 1].min():.4f}, {objectives_display[:, 1].max():.4f}]")
            print(f"F3 range: [{objectives[:, 2].min():.0f}, {objectives[:, 2].max():.0f}]")

        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue

    # Calculate quality metrics
    print("\n" + "=" * 80)
    print("Quality Metrics Comparison")
    print("=" * 80)

    metrics_calc = QualityMetrics()

    for method_name, results in results_by_method.items():
        print(f"\n{method_name}:")
        print("-" * 40)

        # Calculate metrics
        objectives = results["objectives"]

        # Hypervolume (using normalized objectives)
        hv = metrics_calc.hypervolume(objectives)
        print(f"Hypervolume: {hv:.4f}")

        # Spacing (uniformity)
        spacing = metrics_calc.spacing(objectives)
        print(f"Spacing: {spacing:.4f}")

        # Spread
        spread = metrics_calc.spread(objectives)
        print(f"Spread: {spread:.4f}")

        # Diversity
        diversity = metrics_calc.diversity(objectives)
        print(f"Diversity: {diversity:.4f}")

        # Number of non-dominated solutions
        non_dom = metrics_calc.filter_dominated(objectives)
        print(f"Non-dominated solutions: {len(non_dom)} / {len(objectives)}")

    # Visualization
    if len(results_by_method) > 0:
        print("\n" + "=" * 80)
        print("Creating visualization...")

        fig = plt.figure(figsize=(15, 5))

        for idx, (method_name, results) in enumerate(results_by_method.items(), 1):
            ax = fig.add_subplot(1, 3, idx, projection='3d')

            obj_display = results["objectives_display"]

            # Plot solutions
            ax.scatter(obj_display[:, 0], obj_display[:, 1], obj_display[:, 2],
                      c='blue', marker='o', alpha=0.6)

            ax.set_xlabel('F1: Linked Usage')
            ax.set_ylabel('F2: Semantic Similarity')
            ax.set_zlabel('F3: Set Size')
            ax.set_title(f'{method_name} Decomposition')

            # Set viewing angle
            ax.view_init(elev=20, azim=45)

        plt.suptitle(f'MOEA/D Results for {package_name}')
        plt.tight_layout()

        # Save figure
        output_file = 'results/moead_comparison.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
        plt.close()

    # Test algorithm parameters
    print("\n" + "=" * 80)
    print("Testing Algorithm Parameters")
    print("=" * 80)

    # Test with different parameters
    param_tests = [
        {"cr": 0.5, "f_scale": 0.5, "name": "Low CR"},
        {"cr": 1.0, "f_scale": 0.8, "name": "High F"},
        {"cr": 0.9, "f_scale": 0.5, "name": "Balanced"}
    ]

    for params in param_tests:
        print(f"\nTesting {params['name']} (CR={params['cr']}, F={params['f_scale']})...")

        try:
            moead = MOEAD(
                package_name=package_name,
                pop_size=30,
                n_neighbors=10,
                max_gen=20,
                decomposition="tchebycheff",
                cr=params["cr"],
                f_scale=params["f_scale"]
            )

            solutions = moead.run()
            print(f"Solutions found: {len(solutions)}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Testing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_moead_implementation()