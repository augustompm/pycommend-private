"""
Final convergence test for MOEA/D with normalization
Verify positive convergence and generate comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir('pycommend-code')


def test_algorithm_convergence(algorithm='normalized', package='fastapi', generations=30):
    """
    Test convergence of different MOEA/D versions
    """
    if algorithm == 'normalized':
        from optimizer.moead_vns_normalized import MOEAD_VNS_Normalized
        moead = MOEAD_VNS_Normalized(package, pop_size=50, max_gen=generations,
                                     n_neighbors=15, theta=5.0, track_metrics=True)
        algo_name = "MOEA/D Normalized"
    elif algorithm == 'original':
        from optimizer.moead_vns_final import MOEAD_VNS_Final
        moead = MOEAD_VNS_Final(package, pop_size=50, max_gen=generations,
                               n_neighbors=15, track_metrics=True)
        algo_name = "MOEA/D Original"
    else:
        from optimizer.movns_vns import MOVNS_VNS
        movns = MOVNS_VNS(package, archive_size=50, max_iterations=generations,
                         track_metrics=True)
        algo_name = "MOVNS"
        solutions = movns.run()
        metrics = movns.get_metrics_history()
        return algo_name, metrics

    solutions = moead.run()
    metrics = moead.get_metrics_history()
    return algo_name, metrics


def analyze_convergence(metrics, algo_name):
    """
    Analyze convergence metrics
    """
    if not metrics or 'hypervolume' not in metrics:
        return None

    hv = metrics['hypervolume']
    if len(hv) < 2:
        return None

    initial = hv[0] if hv[0] > 0 else 0.001
    final = hv[-1]
    improvement = (final - initial) / initial * 100

    monotonic = sum(1 for i in range(1, len(hv)) if hv[i] >= hv[i-1])
    monotonic_rate = monotonic / (len(hv) - 1) * 100

    avg_step = np.mean([abs(hv[i+1] - hv[i]) for i in range(len(hv)-1)])

    return {
        'initial': initial,
        'final': final,
        'improvement': improvement,
        'monotonic_rate': monotonic_rate,
        'avg_step': avg_step,
        'hv_history': hv,
        'converges': improvement > 0
    }


def plot_comparison(results):
    """
    Create comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'MOEA/D Normalized': 'green', 'MOEA/D Original': 'red', 'MOVNS': 'blue'}

    ax = axes[0, 0]
    for algo_name, data in results.items():
        if data and 'hv_history' in data:
            hv = data['hv_history']
            label = f"{algo_name} ({data['improvement']:+.0f}%)"
            color = colors.get(algo_name, 'gray')
            ax.plot(range(len(hv)), hv, label=label, linewidth=2, color=color)

    ax.set_xlabel('Generation/Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    names = list(results.keys())
    improvements = [r['improvement'] if r else 0 for r in results.values()]
    colors_list = [colors.get(name, 'gray') for name in names]

    bars = ax.bar(range(len(names)), improvements, color=colors_list, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('HV Improvement (%)')
    ax.set_title('Final Performance')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in names])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -10),
               f'{imp:+.0f}%', ha='center', va='bottom' if height > 0 else 'top')

    ax = axes[1, 0]
    for algo_name, data in results.items():
        if data and 'hv_history' in data:
            hv = data['hv_history']
            improvements = [(hv[i+1] - hv[i])/abs(hv[i]+1e-10)*100
                          for i in range(len(hv)-1)]
            ax.plot(range(len(improvements)), improvements,
                   label=algo_name, linewidth=1.5, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Step Improvement (%)')
    ax.set_title('Per-Generation Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    data_table = []
    for algo_name, data in results.items():
        if data:
            data_table.append([
                algo_name.replace(' ', '\n'),
                f"{data['initial']:.4f}",
                f"{data['final']:.4f}",
                f"{data['improvement']:+.1f}%",
                f"{data['monotonic_rate']:.0f}%"
            ])

    if data_table:
        table = ax.table(cellText=data_table,
                        colLabels=['Algorithm', 'Initial HV', 'Final HV', 'Improvement', 'Monotonic'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

    ax.axis('off')
    ax.set_title('Performance Summary', pad=20)

    plt.suptitle('MOEA/D Normalization Fix Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../moead_convergence_final.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Main test function
    """
    print("FINAL CONVERGENCE TEST - MOEA/D WITH NORMALIZATION")
    print("="*60)

    results = {}

    print("\n1. Testing MOEA/D with Normalization...")
    print("-"*40)
    start = time.time()
    algo_name, metrics = test_algorithm_convergence('normalized', generations=25)
    analysis = analyze_convergence(metrics, algo_name)
    results[algo_name] = analysis

    if analysis:
        print(f"Initial HV: {analysis['initial']:.4f}")
        print(f"Final HV: {analysis['final']:.4f}")
        print(f"Improvement: {analysis['improvement']:+.1f}%")
        print(f"Monotonic Rate: {analysis['monotonic_rate']:.0f}%")
        print(f"Status: {'CONVERGING' if analysis['converges'] else 'DIVERGING'}")
    print(f"Time: {time.time()-start:.1f}s")

    print("\n2. Testing Original MOEA/D...")
    print("-"*40)
    start = time.time()
    algo_name, metrics = test_algorithm_convergence('original', generations=25)
    analysis = analyze_convergence(metrics, algo_name)
    results[algo_name] = analysis

    if analysis:
        print(f"Initial HV: {analysis['initial']:.4f}")
        print(f"Final HV: {analysis['final']:.4f}")
        print(f"Improvement: {analysis['improvement']:+.1f}%")
        print(f"Monotonic Rate: {analysis['monotonic_rate']:.0f}%")
        print(f"Status: {'CONVERGING' if analysis['converges'] else 'DIVERGING'}")
    print(f"Time: {time.time()-start:.1f}s")

    print("\n3. Testing MOVNS for Reference...")
    print("-"*40)
    start = time.time()
    algo_name, metrics = test_algorithm_convergence('movns', generations=25)
    analysis = analyze_convergence(metrics, algo_name)
    results[algo_name] = analysis

    if analysis:
        print(f"Initial HV: {analysis['initial']:.4f}")
        print(f"Final HV: {analysis['final']:.4f}")
        print(f"Improvement: {analysis['improvement']:+.1f}%")
        print(f"Monotonic Rate: {analysis['monotonic_rate']:.0f}%")
        print(f"Status: {'CONVERGING' if analysis['converges'] else 'DIVERGING'}")
    print(f"Time: {time.time()-start:.1f}s")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for algo_name, analysis in results.items():
        if analysis:
            status = "CONVERGES" if analysis['converges'] else "DIVERGES"
            print(f"{algo_name:25s}: {status:10s} ({analysis['improvement']:+.1f}%)")

    normalized = results.get('MOEA/D Normalized')
    original = results.get('MOEA/D Original')

    if normalized and original:
        if normalized['converges'] and not original['converges']:
            print("\nSUCCESS: Normalization fixes the convergence problem!")
        elif normalized['converges'] and original['converges']:
            if normalized['improvement'] > original['improvement']:
                print(f"\nSUCCESS: Normalization improves convergence by {normalized['improvement'] - original['improvement']:.1f}%")
        else:
            print("\nWARNING: Results inconclusive")

    plot_comparison(results)
    print("\nPlot saved to: moead_convergence_final.png")


if __name__ == "__main__":
    main()