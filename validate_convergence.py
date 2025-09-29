"""
Validate and ensure positive convergence for all algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('pycommend-code/src')
os.chdir('pycommend-code')


def test_algorithm_convergence(algo_name, iterations=30):
    """
    Test an algorithm and verify positive convergence
    """
    print(f"\nTesting {algo_name}...")

    if algo_name == 'MOVNS':
        from optimizer.movns_vns import MOVNS_VNS
        algo = MOVNS_VNS('fastapi', archive_size=100, max_iterations=iterations, track_metrics=True)

    elif algo_name == 'MOEAD_Original':
        from optimizer.moead_vns import MOEAD_VNS
        algo = MOEAD_VNS('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)

    elif algo_name == 'NSGA2':
        from optimizer.nsga2_vns import NSGA2_VNS
        algo = NSGA2_VNS('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)

    solutions = algo.run()
    metrics = algo.get_metrics_history()

    convergence_status = {
        'algorithm': algo_name,
        'converges': False,
        'improvement': 0,
        'monotonic_rate': 0,
        'hv_history': []
    }

    if metrics and 'hypervolume' in metrics and metrics['hypervolume']:
        hv = metrics['hypervolume']
        convergence_status['hv_history'] = hv

        if len(hv) > 1:
            initial = hv[0] if hv[0] > 0 else 0.001
            final = hv[-1]
            improvement = (final - initial) / initial * 100

            monotonic = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
            monotonic_rate = monotonic / (len(hv) - 1) * 100

            convergence_status['converges'] = improvement > 0
            convergence_status['improvement'] = improvement
            convergence_status['monotonic_rate'] = monotonic_rate

            print(f"  Initial HV: {hv[0]:.4f}")
            print(f"  Final HV: {hv[-1]:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Monotonic rate: {monotonic_rate:.1f}%")

            if improvement > 0:
                print("  STATUS: CONVERGING")
            else:
                print("  STATUS: DIVERGING - NEEDS FIX")

    return convergence_status


def create_convergence_fix_for_moead():
    """
    Create a simple fix to ensure MOEA/D converges
    Add archive mechanism similar to MOVNS
    """
    print("\n" + "="*60)
    print("IMPLEMENTING CONVERGENCE FIX FOR MOEA/D")
    print("="*60)

    fix_code = """
# Key changes needed in moead_vns.py:

1. Add archive mechanism:
   - Keep best solutions found so far
   - Never lose good solutions

2. Change update strategy:
   - Always preserve improvements
   - Use archive to guide evolution

3. Add restart mechanism:
   - If no improvement for 5 generations
   - Reinitialize 20% of population from archive
"""

    print(fix_code)

    return fix_code


def plot_convergence_comparison(results):
    """
    Plot convergence comparison
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    colors = {'MOVNS': 'blue', 'MOEAD_Original': 'orange', 'NSGA2': 'green'}

    for result in results:
        algo = result['algorithm']
        hv = result['hv_history']
        if hv:
            label = f"{algo} ({'OK' if result['converges'] else 'FAIL'})"
            plt.plot(range(len(hv)), hv, label=label,
                    color=colors.get(algo, 'gray'), linewidth=2)

    plt.xlabel('Iteration/Generation')
    plt.ylabel('Hypervolume')
    plt.title('Convergence Validation - Raw HV')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)

    algo_names = [r['algorithm'] for r in results]
    improvements = [r['improvement'] for r in results]
    monotonic_rates = [r['monotonic_rate'] for r in results]

    x = np.arange(len(algo_names))
    width = 0.35

    bars1 = plt.bar(x - width/2, improvements, width, label='Improvement (%)')
    bars2 = plt.bar(x + width/2, monotonic_rates, width, label='Monotonic (%)')

    for i, (imp, mono) in enumerate(zip(improvements, monotonic_rates)):
        color1 = 'green' if imp > 0 else 'red'
        color2 = 'green' if mono > 60 else 'orange'
        bars1[i].set_color(color1)
        bars1[i].set_alpha(0.7)
        bars2[i].set_color(color2)
        bars2[i].set_alpha(0.7)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=60, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Algorithm')
    plt.ylabel('Percentage')
    plt.title('Convergence Quality Metrics')
    plt.xticks(x, algo_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Convergence Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../convergence_validation.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("CONVERGENCE VALIDATION")
    print("="*60)

    algorithms = ['MOVNS', 'MOEAD_Original', 'NSGA2']
    results = []

    for algo in algorithms:
        result = test_algorithm_convergence(algo, iterations=20)
        results.append(result)

    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60)

    converging = []
    diverging = []

    for result in results:
        if result['converges']:
            converging.append(result['algorithm'])
        else:
            diverging.append(result['algorithm'])

    print(f"\nConverging algorithms: {converging}")
    print(f"Diverging algorithms: {diverging}")

    if diverging:
        print(f"\nWARNING: {len(diverging)} algorithm(s) need fixing")
        fix_suggestions = create_convergence_fix_for_moead()

    plot_convergence_comparison(results)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("Plot saved to convergence_validation.png")

    return results


if __name__ == "__main__":
    results = main()