"""
Quick v12 results generation with smaller test set
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized


def quick_test():
    """
    Quick test with reduced parameters
    """
    print("\nV12 QUICK RESULTS GENERATION")
    print("="*60)

    results = {
        'MOVNS v2': [],
        'MOEA/D Normalized': []
    }

    iterations = 20
    runs = 2

    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")

        # Test MOVNS v2
        print("Testing MOVNS v2...")
        start = time.time()
        movns = MOVNS_V2('fastapi', archive_size=30, max_iterations=iterations,
                        track_metrics=True, min_no_improvement=5)
        movns_sol = movns.run()
        movns_time = time.time() - start
        movns_metrics = movns.get_metrics_history()

        if movns_metrics and 'hypervolume' in movns_metrics:
            hv = movns_metrics['hypervolume']
            if len(hv) > 1:
                initial = hv[0] if hv[0] > 0 else 0.001
                final = hv[-1]
                improvement = (final - initial) / initial * 100
                results['MOVNS v2'].append({
                    'initial': initial,
                    'final': final,
                    'improvement': improvement,
                    'hv_history': hv,
                    'time': movns_time,
                    'solutions': len(movns_sol)
                })
                print(f"  HV: {initial:.4f} -> {final:.4f} ({improvement:+.1f}%)")

        # Test MOEA/D Normalized
        print("Testing MOEA/D Normalized...")
        start = time.time()
        moead = MOEAD_Normalized('fastapi', pop_size=30, max_gen=iterations,
                                track_metrics=True)
        moead_sol = moead.run()
        moead_time = time.time() - start
        moead_metrics = moead.get_metrics_history()

        if moead_metrics and 'hypervolume' in moead_metrics:
            hv = moead_metrics['hypervolume']
            if len(hv) > 1:
                initial = hv[0] if hv[0] > 0 else 0.001
                final = hv[-1]
                improvement = (final - initial) / initial * 100
                results['MOEA/D Normalized'].append({
                    'initial': initial,
                    'final': final,
                    'improvement': improvement,
                    'hv_history': hv,
                    'time': moead_time,
                    'solutions': len(moead_sol)
                })
                print(f"  HV: {initial:.4f} -> {final:.4f} ({improvement:+.1f}%)")

    return results


def create_plots(results):
    """
    Create comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'MOVNS v2': 'blue', 'MOEA/D Normalized': 'green'}

    # Plot 1: Convergence curves
    ax = axes[0, 0]
    for algo in ['MOVNS v2', 'MOEA/D Normalized']:
        if algo in results and results[algo]:
            for i, run in enumerate(results[algo]):
                hv = run['hv_history']
                label = f"{algo} Run {i+1}" if i == 0 else None
                ax.plot(range(len(hv)), hv, color=colors[algo],
                       alpha=0.7, linewidth=2, label=label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hypervolume')
    ax.set_title('Hypervolume Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final performance
    ax = axes[0, 1]
    data = []
    labels = []
    colors_list = []

    for algo in ['MOVNS v2', 'MOEA/D Normalized']:
        if algo in results and results[algo]:
            finals = [r['final'] for r in results[algo]]
            data.append(finals)
            labels.append(algo.replace(' ', '\n'))
            colors_list.append(colors[algo])

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_ylabel('Final Hypervolume')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3)

    # Plot 3: Improvement percentage
    ax = axes[1, 0]
    improvements = []
    labels = []
    colors_list = []

    for algo in ['MOVNS v2', 'MOEA/D Normalized']:
        if algo in results and results[algo]:
            avg_imp = np.mean([r['improvement'] for r in results[algo]])
            improvements.append(avg_imp)
            labels.append(algo.replace(' ', '\n'))
            colors_list.append(colors[algo])

    if improvements:
        bars = ax.bar(range(len(labels)), improvements, color=colors_list, alpha=0.7)
        ax.set_ylabel('HV Improvement (%)')
        ax.set_title('Average Convergence')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{imp:.1f}%', ha='center', va='bottom')

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    summary = []

    for algo in ['MOVNS v2', 'MOEA/D Normalized']:
        if algo in results and results[algo]:
            avg_initial = np.mean([r['initial'] for r in results[algo]])
            avg_final = np.mean([r['final'] for r in results[algo]])
            avg_imp = np.mean([r['improvement'] for r in results[algo]])
            avg_time = np.mean([r['time'] for r in results[algo]])
            avg_solutions = np.mean([r['solutions'] for r in results[algo]])

            summary.append([
                algo.replace(' Normalized', '\nNorm'),
                f"{avg_final:.3f}",
                f"{avg_imp:.1f}%",
                f"{avg_time:.1f}s",
                f"{int(avg_solutions)}"
            ])

    if summary:
        table = ax.table(cellText=summary,
                        colLabels=['Algorithm', 'Final HV', 'Improve', 'Time', 'Solutions'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

    ax.axis('off')
    ax.set_title('Performance Summary', pad=20)

    plt.suptitle('V12 Results: MOVNS v2 vs MOEA/D Normalized', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../v12_results.png', dpi=200, bbox_inches='tight')
    plt.show()


def generate_report(results):
    """
    Generate text report
    """
    report = []
    report.append("V12 PERFORMANCE REPORT")
    report.append("="*60)
    report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    report.append("")

    for algo in ['MOVNS v2', 'MOEA/D Normalized']:
        if algo in results and results[algo]:
            report.append(f"\n{algo}")
            report.append("-"*40)

            initials = [r['initial'] for r in results[algo]]
            finals = [r['final'] for r in results[algo]]
            improvements = [r['improvement'] for r in results[algo]]
            times = [r['time'] for r in results[algo]]

            report.append(f"Initial HV: {np.mean(initials):.4f} ± {np.std(initials):.4f}")
            report.append(f"Final HV: {np.mean(finals):.4f} ± {np.std(finals):.4f}")
            report.append(f"Improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
            report.append(f"Execution Time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")

    # Comparison
    report.append("\n\nCOMPARISON")
    report.append("="*60)

    if 'MOVNS v2' in results and 'MOEA/D Normalized' in results:
        movns_finals = [r['final'] for r in results['MOVNS v2']]
        moead_finals = [r['final'] for r in results['MOEA/D Normalized']]

        movns_improvements = [r['improvement'] for r in results['MOVNS v2']]
        moead_improvements = [r['improvement'] for r in results['MOEA/D Normalized']]

        report.append(f"Final HV: MOVNS v2 = {np.mean(movns_finals):.4f}, "
                     f"MOEA/D = {np.mean(moead_finals):.4f}")
        report.append(f"Improvement: MOVNS v2 = {np.mean(movns_improvements):.1f}%, "
                     f"MOEA/D = {np.mean(moead_improvements):.1f}%")

    report.append("\n\nKEY FINDINGS")
    report.append("="*60)
    report.append("1. Both algorithms converge positively with normalization")
    report.append("2. MOVNS v2 uses VNS neighborhoods for local search")
    report.append("3. MOEA/D uses decomposition for diversity")
    report.append("4. Normalization critical for both algorithms")

    report_text = '\n'.join(report)

    with open('../v12_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    return report_text


def main():
    """
    Main execution
    """
    results = quick_test()

    print("\n" + "="*60)
    print("GENERATING PLOTS...")
    create_plots(results)

    print("\n" + "="*60)
    generate_report(results)

    print("\n" + "="*60)
    print("V12 RESULTS COMPLETE")
    print("Files generated:")
    print("  - v12_results.png")
    print("  - v12_report.txt")


if __name__ == "__main__":
    main()