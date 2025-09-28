"""
Test NSGA-II with IGD+ metric tracking
"""

import sys
import os
import numpy as np

os.chdir('pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from nsga2_vns import NSGA2_VNS


def test_with_metrics():
    """Test NSGA-II with quality metrics tracking"""

    print("="*70)
    print("TESTING NSGA-II WITH IGD+ TRACKING")
    print("="*70)

    # Test with smaller population/generations for speed
    packages = ['fastapi', 'scikit-learn']

    for package in packages:
        print(f"\n{'='*70}")
        print(f"Testing package: {package.upper()}")
        print("="*70)

        # Run with metrics tracking enabled
        nsga2 = NSGA2_VNS(package, pop_size=20, max_gen=20, track_metrics=True)
        solutions = nsga2.run()

        # Get metrics history
        metrics_history = nsga2.get_metrics_history()

        if metrics_history:
            print("\n" + "-"*70)
            print("METRICS EVOLUTION")
            print("-"*70)

            # Show evolution of metrics
            if metrics_history['hypervolume']:
                hv_values = metrics_history['hypervolume']
                print(f"Hypervolume: {hv_values[0]:.4f} -> {hv_values[-1]:.4f}")
                if len(hv_values) > 1:
                    improvement = (hv_values[-1] - hv_values[0]) / abs(hv_values[0]) * 100
                    print(f"  Improvement: {improvement:+.1f}%")

            if metrics_history['igd_plus']:
                igd_values = [v for v in metrics_history['igd_plus'] if v is not None]
                if igd_values:
                    print(f"IGD+: {igd_values[0]:.4f} -> {igd_values[-1]:.4f}")
                    if len(igd_values) > 1:
                        reduction = (igd_values[0] - igd_values[-1]) / igd_values[0] * 100
                        print(f"  Reduction: {reduction:.1f}% (lower is better)")

            if metrics_history['spacing']:
                sp_values = metrics_history['spacing']
                print(f"Spacing: {sp_values[0]:.4f} -> {sp_values[-1]:.4f}")

            if metrics_history['diversity']:
                div_values = metrics_history['diversity']
                print(f"Diversity: {div_values[0]:.4f} -> {div_values[-1]:.4f}")

        # Show best solution
        print("\n" + "-"*70)
        print("BEST SOLUTION")
        print("-"*70)

        recommendations = nsga2.get_recommendations(solutions)
        if recommendations:
            best = recommendations[0]
            print(f"Packages ({best['size']}): {', '.join(best['packages'][:10])}")
            print(f"LU: {best['linked_usage']:.1f}")
            print(f"SS: {best['semantic_similarity']:.3f}")


def compare_with_without_metrics():
    """Compare performance with and without metrics tracking"""

    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: WITH vs WITHOUT METRICS")
    print("="*70)

    import time

    package = 'numpy'

    # Run without metrics
    print(f"\nRunning without metrics tracking...")
    start = time.time()
    nsga2_no_metrics = NSGA2_VNS(package, pop_size=50, max_gen=30, track_metrics=False)
    solutions_no_metrics = nsga2_no_metrics.run()
    time_no_metrics = time.time() - start

    # Run with metrics
    print(f"\nRunning with metrics tracking...")
    start = time.time()
    nsga2_with_metrics = NSGA2_VNS(package, pop_size=50, max_gen=30, track_metrics=True)
    solutions_with_metrics = nsga2_with_metrics.run()
    time_with_metrics = time.time() - start

    # Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"Without metrics: {time_no_metrics:.2f}s, {len(solutions_no_metrics)} solutions")
    print(f"With metrics: {time_with_metrics:.2f}s, {len(solutions_with_metrics)} solutions")
    overhead = (time_with_metrics - time_no_metrics) / time_no_metrics * 100
    print(f"Overhead: {overhead:.1f}%")

    # Show metrics summary if available
    metrics_history = nsga2_with_metrics.get_metrics_history()
    if metrics_history and metrics_history['igd_plus']:
        igd_values = [v for v in metrics_history['igd_plus'] if v is not None]
        if igd_values:
            print(f"\nIGD+ convergence: {len(igd_values)} measurements")
            print(f"Initial IGD+: {igd_values[0]:.4f}")
            print(f"Final IGD+: {igd_values[-1]:.4f}")
            print(f"Best IGD+: {min(igd_values):.4f}")


if __name__ == '__main__':
    test_with_metrics()
    compare_with_without_metrics()

    print("\n" + "="*70)
    print("IGD+ TESTING COMPLETE")
    print("="*70)
    print("\nUsage: python -m src.optimizer.nsga2_vns fastapi --metrics")
    print("This will enable IGD+ and other quality metrics tracking during evolution")