"""
Test MOVNS Improved vs Original
Compare convergence and performance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir('pycommend-code')


def test_algorithm(algorithm='improved', package='fastapi', iterations=30):
    """
    Test MOVNS versions
    """
    if algorithm == 'improved':
        from optimizer.movns_improved import MOVNS_Improved
        movns = MOVNS_Improved(package, archive_size=50, max_iterations=iterations,
                               track_metrics=True, min_no_improvement=10)
        algo_name = "MOVNS Improved"
    else:
        from optimizer.movns_vns import MOVNS_VNS
        movns = MOVNS_VNS(package, archive_size=50, max_iterations=iterations,
                         track_metrics=True)
        algo_name = "MOVNS Original"

    print(f"\nTesting {algo_name}...")
    print("-"*40)

    start = time.time()
    solutions = movns.run()
    exec_time = time.time() - start

    metrics = movns.get_metrics_history()

    results = {
        'name': algo_name,
        'solutions': len(solutions),
        'time': exec_time,
        'metrics': metrics
    }

    if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 1:
        hv = metrics['hypervolume']
        initial = hv[0] if hv[0] > 0 else 0.001
        final = hv[-1]
        improvement = (final - initial) / initial * 100

        results['initial_hv'] = initial
        results['final_hv'] = final
        results['improvement'] = improvement

        monotonic = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
        monotonic_rate = monotonic / (len(hv) - 1) * 100
        results['monotonic_rate'] = monotonic_rate

        print(f"Initial HV: {initial:.4f}")
        print(f"Final HV: {final:.4f}")
        print(f"Improvement: {improvement:+.1f}%")
        print(f"Monotonic Rate: {monotonic_rate:.0f}%")
        print(f"Solutions: {len(solutions)}, Time: {exec_time:.1f}s")

    return results


def compare_movns_versions():
    """
    Compare improved vs original MOVNS
    """
    print("MOVNS IMPROVEMENT TEST")
    print("="*60)
    print("Comparing: Normalization + Better Early Stopping")
    print("="*60)

    results = []

    # Test improved version
    improved_result = test_algorithm('improved', iterations=30)
    results.append(improved_result)

    # Test original version
    original_result = test_algorithm('original', iterations=30)
    results.append(original_result)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Hypervolume convergence
    ax = axes[0, 0]
    colors = {'MOVNS Improved': 'green', 'MOVNS Original': 'blue'}

    for result in results:
        if 'metrics' in result and result['metrics']:
            hv = result['metrics']['hypervolume']
            label = f"{result['name']} ({result.get('improvement', 0):+.0f}%)"
            ax.plot(range(len(hv)), hv, label=label,
                   color=colors.get(result['name'], 'gray'), linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final performance bars
    ax = axes[0, 1]
    names = [r['name'].replace(' ', '\n') for r in results]
    final_hvs = [r.get('final_hv', 0) for r in results]
    colors_list = ['green' if 'Improved' in r['name'] else 'blue' for r in results]

    bars = ax.bar(range(len(names)), final_hvs, color=colors_list, alpha=0.7)
    ax.set_ylabel('Final Hypervolume')
    ax.set_title('Final Performance')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, hv in zip(bars, final_hvs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{hv:.3f}', ha='center', va='bottom')

    # Plot 3: Improvement percentage
    ax = axes[1, 0]
    improvements = [r.get('improvement', 0) for r in results]
    colors_list = ['green' if imp > 50 else 'orange' if imp > 0 else 'red'
                  for imp in improvements]

    bars = ax.bar(range(len(names)), improvements, color=colors_list, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('HV Improvement (%)')
    ax.set_title('Convergence Improvement')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
               height + (5 if height > 0 else -10),
               f'{imp:+.0f}%', ha='center',
               va='bottom' if height > 0 else 'top')

    # Plot 4: Summary table
    ax = axes[1, 1]
    data_table = []
    for r in results:
        data_table.append([
            r['name'].replace(' ', '\n'),
            f"{r.get('initial_hv', 0):.4f}",
            f"{r.get('final_hv', 0):.4f}",
            f"{r.get('improvement', 0):+.1f}%",
            f"{r.get('monotonic_rate', 0):.0f}%",
            f"{r.get('time', 0):.1f}s"
        ])

    table = ax.table(cellText=data_table,
                    colLabels=['Algorithm', 'Initial HV', 'Final HV',
                              'Improvement', 'Monotonic', 'Time'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.axis('off')
    ax.set_title('Performance Summary', pad=20)

    plt.suptitle('MOVNS Improvement Analysis - With Normalization',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../movns_improvement_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    improved = results[0]
    original = results[1]

    if improved.get('improvement', 0) > original.get('improvement', 0):
        gain = improved['improvement'] - original['improvement']
        print(f"SUCCESS: Improved version better by {gain:.1f} percentage points")
        print(f"  Improved: {improved['improvement']:+.1f}%")
        print(f"  Original: {original['improvement']:+.1f}%")
    else:
        print("No significant improvement detected")

    print(f"\nKey improvements implemented:")
    print("  1. Objective normalization for fair dominance")
    print("  2. Dynamic bounds tracking")
    print("  3. Better early stopping (10 iterations)")
    print("  4. Crowding distance for archive truncation")

    print(f"\nPlot saved to: movns_improvement_test.png")


def main():
    """
    Main test function
    """
    compare_movns_versions()


if __name__ == "__main__":
    main()