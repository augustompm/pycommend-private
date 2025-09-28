"""
Test corrected IGD+ implementation
IGD+ should NOT be zero when measuring convergence
"""

import sys
import os
import numpy as np

os.chdir('pycommend-code')
sys.path.append('src/optimizer')
sys.path.append('src/evaluation')

from nsga2_vns import NSGA2_VNS


def test_igd_plus_corrected():
    """Test IGD+ with proper reference set"""

    print("="*70)
    print("TESTING CORRECTED IGD+ IMPLEMENTATION")
    print("="*70)
    print("\nIGD+ should measure distance to a reference set")
    print("It should NOT be zero unless the Pareto front is perfect\n")

    package = 'fastapi'

    # Run with metrics tracking
    print(f"Testing package: {package}")
    print("-"*70)

    nsga2 = NSGA2_VNS(package, pop_size=30, max_gen=30, track_metrics=True)

    print("\nRunning NSGA-II with proper reference set generation...")
    solutions = nsga2.run()

    # Get metrics history
    metrics_history = nsga2.get_metrics_history()

    if metrics_history and metrics_history['igd_plus']:
        igd_values = [v for v in metrics_history['igd_plus'] if v is not None]

        print("\n" + "="*70)
        print("IGD+ EVOLUTION")
        print("="*70)

        if igd_values:
            print(f"Number of IGD+ measurements: {len(igd_values)}")
            print(f"Initial IGD+: {igd_values[0]:.4f}")
            print(f"Final IGD+: {igd_values[-1]:.4f}")
            print(f"Best IGD+: {min(igd_values):.4f}")
            print(f"Worst IGD+: {max(igd_values):.4f}")

            # Check if IGD+ is properly non-zero
            if all(v == 0 for v in igd_values):
                print("\n⚠️ WARNING: All IGD+ values are zero!")
                print("This indicates the reference set may be identical to the Pareto front")
            else:
                print("\n✓ IGD+ values are properly non-zero")

                # Calculate improvement
                if len(igd_values) > 1 and igd_values[0] > 0:
                    improvement = (igd_values[0] - igd_values[-1]) / igd_values[0] * 100
                    print(f"IGD+ improvement: {improvement:.1f}%")

                    if improvement > 0:
                        print("✓ Algorithm is converging (IGD+ decreasing)")
                    elif improvement < 0:
                        print("⚠️ Algorithm may be diverging (IGD+ increasing)")
                    else:
                        print("→ Algorithm has stabilized")

        # Show reference set info
        if nsga2.reference_set is not None:
            print("\n" + "-"*70)
            print("REFERENCE SET INFO")
            print("-"*70)
            print(f"Reference set size: {len(nsga2.reference_set)}")
            print(f"Reference set shape: {nsga2.reference_set.shape}")

            # Show sample reference points
            print("\nSample reference points (first 3):")
            for i, point in enumerate(nsga2.reference_set[:3]):
                print(f"  Ref {i+1}: LU={-point[0]:.1f}, SS={-point[1]:.3f}, RSS={point[2]:.1f}")

            # Compare with actual solutions
            print("\nSample actual solutions (first 3):")
            for i, sol in enumerate(solutions[:3]):
                obj = sol['objectives']
                print(f"  Sol {i+1}: LU={-obj[0]:.1f}, SS={-obj[1]:.3f}, RSS={obj[2]:.1f}")
    else:
        print("No IGD+ values recorded")

    # Show other metrics for comparison
    if metrics_history:
        print("\n" + "="*70)
        print("OTHER METRICS COMPARISON")
        print("="*70)

        for metric_name in ['hypervolume', 'spacing', 'diversity']:
            if metric_name in metrics_history and metrics_history[metric_name]:
                values = metrics_history[metric_name]
                print(f"{metric_name.capitalize()}:")
                print(f"  Initial: {values[0]:.4f}")
                print(f"  Final: {values[-1]:.4f}")
                if len(values) > 1:
                    change = (values[-1] - values[0]) / abs(values[0]) * 100
                    print(f"  Change: {change:+.1f}%")


if __name__ == '__main__':
    test_igd_plus_corrected()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey points about IGD+:")
    print("1. IGD+ measures distance from current front to reference set")
    print("2. Lower IGD+ means better convergence")
    print("3. IGD+ = 0 only if current front dominates entire reference")
    print("4. Reference set should be well-distributed in objective space")
    print("5. IGD+ decreasing over generations shows improvement")