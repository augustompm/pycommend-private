"""
Detailed convergence test with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

sys.path.append('pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.moead_vns import MOEAD_VNS
from optimizer.nsga2_vns import NSGA2_VNS

os.chdir('pycommend-code')

def run_detailed_test(package='fastapi', iterations=50):
    """
    Run all algorithms and plot convergence
    """
    results = {}

    algorithms = [
        ('MOVNS', MOVNS_VNS(package, archive_size=100, max_iterations=iterations, track_metrics=True)),
        ('MOEA/D', MOEAD_VNS(package, pop_size=100, max_gen=iterations, track_metrics=True)),
        ('NSGA-II', NSGA2_VNS(package, pop_size=100, max_gen=iterations, track_metrics=True))
    ]

    for name, algo in algorithms:
        print(f"\nRunning {name}...")
        solutions = algo.run()
        metrics = algo.get_metrics_history()

        if metrics:
            results[name] = metrics
            if 'hypervolume' in metrics and metrics['hypervolume']:
                hv = metrics['hypervolume']
                print(f"{name} Results:")
                print(f"  HV points: {len(hv)}")
                print(f"  Initial HV: {hv[0]:.6f}")
                print(f"  Final HV: {hv[-1]:.6f}")
                print(f"  Max HV: {max(hv):.6f}")
                print(f"  Improvement: {((hv[-1] - hv[0]) / (hv[0] + 1e-10)) * 100:.1f}%")

                improvements = sum(1 for i in range(1, len(hv)) if hv[i] > hv[i-1])
                print(f"  Steps with improvement: {improvements}/{len(hv)-1} ({improvements/(len(hv)-1)*100:.1f}%)")

    return results

def plot_convergence(results):
    """
    Create convergence plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'MOVNS': 'blue', 'MOEA/D': 'orange', 'NSGA-II': 'green'}

    ax = axes[0, 0]
    for name, metrics in results.items():
        if 'hypervolume' in metrics and metrics['hypervolume']:
            hv = metrics['hypervolume']
            ax.plot(range(len(hv)), hv, label=name, color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title('Hypervolume Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, metrics in results.items():
        if 'hypervolume' in metrics and metrics['hypervolume']:
            hv = metrics['hypervolume']
            normalized = [(v - hv[0]) / (max(hv) - hv[0] + 1e-10) for v in hv]
            ax.plot(range(len(normalized)), normalized, label=name, color=colors.get(name, 'gray'), linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Normalized HV (0=initial, 1=best)')
    ax.set_title('Normalized Hypervolume Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name, metrics in results.items():
        if 'spacing' in metrics and metrics['spacing']:
            spacing = metrics['spacing']
            ax.plot(range(len(spacing)), spacing, label=name, color=colors.get(name, 'gray'), linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Spacing')
    ax.set_title('Spacing Metric (uniformity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for name, metrics in results.items():
        if 'diversity' in metrics and metrics['diversity']:
            diversity = metrics['diversity']
            ax.plot(range(len(diversity)), diversity, label=name, color=colors.get(name, 'gray'), linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Diversity')
    ax.set_title('Diversity Metric')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Algorithm Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../convergence_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig

def analyze_convergence_quality(results):
    """
    Analyze convergence quality metrics
    """
    print("\n" + "="*60)
    print("CONVERGENCE QUALITY ANALYSIS")
    print("="*60)

    for name, metrics in results.items():
        if 'hypervolume' not in metrics or not metrics['hypervolume']:
            continue

        hv = metrics['hypervolume']

        smoothness = []
        for i in range(1, len(hv)):
            change = abs(hv[i] - hv[i-1])
            smoothness.append(change)

        avg_change = np.mean(smoothness) if smoothness else 0
        std_change = np.std(smoothness) if smoothness else 0

        monotonic_improvements = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
        monotonic_rate = monotonic_improvements / (len(hv) - 1) * 100 if len(hv) > 1 else 0

        final_10_percent = int(len(hv) * 0.1) or 1
        convergence_rate = (hv[-1] - hv[-final_10_percent]) / final_10_percent if final_10_percent > 0 else 0

        print(f"\n{name}:")
        print(f"  Average change per iteration: {avg_change:.6f} (±{std_change:.6f})")
        print(f"  Monotonic improvement rate: {monotonic_rate:.1f}%")
        print(f"  Convergence rate (last 10%): {convergence_rate:.6f}")
        print(f"  Total improvement: {((hv[-1] - hv[0]) / (hv[0] + 1e-10)) * 100:.1f}%")

if __name__ == "__main__":
    print("DETAILED CONVERGENCE ANALYSIS")
    print("="*60)

    results = run_detailed_test('fastapi', iterations=50)

    if results:
        plot_convergence(results)
        analyze_convergence_quality(results)

        with open('../convergence_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\nResults saved to convergence_results.pkl")
        print("Plot saved to convergence_detailed.png")
    else:
        print("No results obtained")