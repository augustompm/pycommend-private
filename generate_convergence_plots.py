"""
Generate convergence plots and statistical graphs for PyCommend paper
Based on state-of-the-art MOEA papers (2023-2024)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import sys
import os
from datetime import datetime
import seaborn as sns
from scipy import stats

sys.path.append('pycommend-code/src')
from optimizer.movns_vns import MOVNS_VNS
from optimizer.moead import MOEAD

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ConvergencePlotter:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_experiments(self, package='fastapi', n_runs=5, generations=50):
        """
        Run multiple experiments and collect metrics
        """
        results = {
            'MOVNS': {'hypervolume': [], 'igd_plus': [], 'spacing': [], 'diversity': []},
            'MOEA/D': {'hypervolume': [], 'igd_plus': [], 'spacing': [], 'diversity': []},
            'NSGA-II': {'hypervolume': [], 'igd_plus': [], 'spacing': [], 'diversity': []}
        }

        raw_data = {'MOVNS': [], 'MOEA/D': [], 'NSGA-II': []}

        print(f"Running experiments for {package}...")
        print(f"Number of runs: {n_runs}, Generations: {generations}")

        for run_id in range(n_runs):
            print(f"\n--- Run {run_id+1}/{n_runs} ---")

            # MOVNS
            print("Running MOVNS...")
            movns = MOVNS_VNS(package, archive_size=100, max_iterations=generations, track_metrics=True)
            movns_solutions = movns.run()
            movns_metrics = movns.get_metrics_history()
            if movns_metrics:
                raw_data['MOVNS'].append(movns_metrics)
                for key in results['MOVNS']:
                    if key in movns_metrics and movns_metrics[key]:
                        results['MOVNS'][key].append(movns_metrics[key])

            # MOEA/D
            print("Running MOEA/D...")
            moead = MOEAD(package, pop_size=100, max_gen=generations, track_metrics=True)
            moead_solutions = moead.run()
            moead_metrics = moead.get_metrics_history()
            if moead_metrics:
                raw_data['MOEA/D'].append(moead_metrics)
                for key in results['MOEA/D']:
                    if key in moead_metrics and moead_metrics[key]:
                        results['MOEA/D'][key].append(moead_metrics[key])

            # NSGA-II (import from nsga2)
            print("Running NSGA-II...")
            from optimizer.nsga2 import NSGA2
            nsga2 = NSGA2(package, pop_size=100, max_gen=generations, track_metrics=True)
            nsga2_solutions = nsga2.run()
            nsga2_metrics = nsga2.get_metrics_history()
            if nsga2_metrics:
                raw_data['NSGA-II'].append(nsga2_metrics)
                for key in results['NSGA-II']:
                    if key in nsga2_metrics and nsga2_metrics[key]:
                        results['NSGA-II'][key].append(nsga2_metrics[key])

        # Save raw data
        with open(f'{self.output_dir}/raw_data_{package}_{datetime.now():%Y%m%d_%H%M%S}.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

        return results, raw_data

    def plot_hypervolume_convergence(self, results, title="Hypervolume Convergence"):
        """
        Plot hypervolume convergence over generations
        Following best practices from 2023-2024 papers
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = {'MOVNS': '#1f77b4', 'MOEA/D': '#ff7f0e', 'NSGA-II': '#2ca02c'}

        # Plot 1: Mean convergence with confidence intervals
        for algo in ['MOVNS', 'MOEA/D', 'NSGA-II']:
            if algo in results and results[algo]['hypervolume']:
                hv_data = results[algo]['hypervolume']

                # Pad shorter runs to match longest
                max_len = max(len(run) for run in hv_data)
                padded_data = []
                for run in hv_data:
                    if len(run) < max_len:
                        padded_run = run + [run[-1]] * (max_len - len(run))
                    else:
                        padded_run = run
                    padded_data.append(padded_run)

                hv_array = np.array(padded_data)
                mean_hv = np.mean(hv_array, axis=0)
                std_hv = np.std(hv_array, axis=0)
                generations = np.arange(len(mean_hv))

                ax1.plot(generations, mean_hv, label=algo, color=colors[algo], linewidth=2)
                ax1.fill_between(generations,
                                mean_hv - std_hv,
                                mean_hv + std_hv,
                                alpha=0.3, color=colors[algo])

        ax1.set_xlabel('Generation/Iteration', fontsize=12)
        ax1.set_ylabel('Hypervolume', fontsize=12)
        ax1.set_title(f'{title} - Mean with 95% CI', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Box plots at specific generations
        generations_to_plot = [10, 20, 30, 40, 50]
        box_data = []
        positions = []
        labels = []

        pos = 0
        for gen in generations_to_plot:
            for i, algo in enumerate(['MOVNS', 'MOEA/D']):
                if algo in results and results[algo]['hypervolume']:
                    gen_idx = min(gen - 1, len(results[algo]['hypervolume'][0]) - 1)
                    values = [run[gen_idx] if gen_idx < len(run) else run[-1]
                             for run in results[algo]['hypervolume']]
                    box_data.append(values)
                    positions.append(pos)
                    if i == 1:
                        labels.append(f'Gen {gen}')
                    pos += 1
                pos += 0.5

        bp = ax2.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

        # Color boxes by algorithm
        for i, box in enumerate(bp['boxes']):
            algo_idx = i % 3
            algo = ['MOVNS', 'MOEA/D', 'NSGA-II'][algo_idx]
            box.set_facecolor(colors[algo])
            box.set_alpha(0.7)

        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Hypervolume', fontsize=12)
        ax2.set_title('Hypervolume Distribution at Key Generations', fontsize=14, fontweight='bold')
        ax2.set_xticks([p + 1 for p in range(0, len(positions), 4)])
        ax2.set_xticklabels(labels)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[algo], alpha=0.7, label=algo)
                          for algo in ['MOVNS', 'MOEA/D', 'NSGA-II']]
        ax2.legend(handles=legend_elements, fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hypervolume_convergence.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/hypervolume_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pareto_front_evolution(self, raw_data, package='fastapi'):
        """
        Plot Pareto front evolution at different generations
        """
        fig = plt.figure(figsize=(15, 5))

        generations_to_plot = [1, 10, 20, 30, 50]

        for idx, gen in enumerate(generations_to_plot, 1):
            ax = fig.add_subplot(1, 5, idx, projection='3d')

            for algo, color in [('MOVNS', 'blue'), ('MOEA/D', 'orange')]:
                if algo in raw_data and raw_data[algo]:
                    # Get first run's data
                    run_data = raw_data[algo][0]

                    # This would require storing the actual solutions, not just metrics
                    # For now, we'll create a placeholder
                    ax.set_xlabel('LU', fontsize=8)
                    ax.set_ylabel('SS', fontsize=8)
                    ax.set_zlabel('RSS', fontsize=8)
                    ax.set_title(f'Gen {gen}', fontsize=10)

        plt.suptitle(f'Pareto Front Evolution - {package}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pareto_evolution.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self, results):
        """
        Create comprehensive performance comparison plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        metrics = ['hypervolume', 'spacing', 'diversity']
        titles = ['Hypervolume (Higher is Better)', 'Spacing (Lower is Better)',
                 'Diversity (Higher is Better)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            if idx >= 2:
                ax = axes[1, idx-2]
            else:
                ax = axes[0, idx]

            # Collect final values
            final_values = {}
            for algo in ['MOVNS', 'MOEA/D', 'NSGA-II']:
                if algo in results and results[algo][metric]:
                    final_values[algo] = [run[-1] for run in results[algo][metric]
                                         if len(run) > 0]

            # Create violin plot
            if final_values:
                data = []
                labels = []
                for algo, values in final_values.items():
                    data.extend(values)
                    labels.extend([algo] * len(values))

                positions = []
                for i, algo in enumerate(['MOVNS', 'MOEA/D']):
                    if algo in final_values:
                        positions.append(i)

                parts = ax.violinplot([final_values[algo] for algo in final_values.keys()],
                                     positions=positions, showmeans=True, showmedians=True)

                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=11)
                ax.set_xticks(positions)
                ax.set_xticklabels(list(final_values.keys()))
                ax.grid(True, alpha=0.3)

        # Statistical significance test (4th subplot)
        ax = axes[1, 1]

        # Perform Wilcoxon signed-rank test
        significance_results = []
        for metric in ['hypervolume', 'spacing', 'diversity']:
            if 'MOVNS' in results and 'MOEA/D' in results:
                if results['MOVNS'][metric] and results['MOEA/D'][metric]:
                    movns_final = [run[-1] for run in results['MOVNS'][metric] if len(run) > 0]
                    moead_final = [run[-1] for run in results['MOEA/D'][metric] if len(run) > 0]

                    if len(movns_final) == len(moead_final) and len(movns_final) > 0:
                        statistic, p_value = stats.wilcoxon(movns_final, moead_final)
                        significance_results.append({
                            'metric': metric,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })

        # Plot significance results
        y_pos = np.arange(len(significance_results))
        p_values = [r['p_value'] for r in significance_results]
        colors = ['green' if r['significant'] else 'gray' for r in significance_results]

        bars = ax.barh(y_pos, p_values, color=colors)
        ax.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r['metric'] for r in significance_results])
        ax.set_xlabel('p-value', fontsize=11)
        ax.set_title('Statistical Significance (MOVNS vs MOEA/D)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_latex_table(self, results):
        """
        Generate LaTeX table with results for paper
        """
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Multi-Objective Algorithms}
\\label{tab:performance}
\\begin{tabular}{lcccc}
\\hline
Algorithm & HV (mean±std) & Spacing (mean±std) & Diversity (mean±std) & Time (s) \\\\
\\hline
"""

        for algo in ['MOVNS', 'MOEA/D', 'NSGA-II']:
            if algo in results:
                hv_final = [run[-1] for run in results[algo]['hypervolume'] if len(run) > 0]
                spacing_final = [run[-1] for run in results[algo]['spacing'] if len(run) > 0]
                diversity_final = [run[-1] for run in results[algo]['diversity'] if len(run) > 0]

                hv_mean = np.mean(hv_final) if hv_final else 0
                hv_std = np.std(hv_final) if hv_final else 0
                spacing_mean = np.mean(spacing_final) if spacing_final else 0
                spacing_std = np.std(spacing_final) if spacing_final else 0
                diversity_mean = np.mean(diversity_final) if diversity_final else 0
                diversity_std = np.std(diversity_final) if diversity_final else 0

                latex_table += f"{algo} & {hv_mean:.4f}±{hv_std:.4f} & "
                latex_table += f"{spacing_mean:.4f}±{spacing_std:.4f} & "
                latex_table += f"{diversity_mean:.4f}±{diversity_std:.4f} & N/A \\\\\n"

        latex_table += """\\hline
\\end{tabular}
\\end{table}
"""

        with open(f'{self.output_dir}/performance_table.tex', 'w') as f:
            f.write(latex_table)

        print("LaTeX table saved to performance_table.tex")
        return latex_table


def main():
    """
    Main function to generate all plots for the paper
    """
    plotter = ConvergencePlotter('plots')

    # Test packages
    test_packages = ['fastapi', 'scikit-learn', 'pandas']

    all_results = {}

    for package in test_packages:
        print(f"\n{'='*50}")
        print(f"Testing package: {package}")
        print('='*50)

        # Run experiments (reduced for testing)
        results, raw_data = plotter.run_experiments(
            package=package,
            n_runs=3,  # Increase to 30 for final paper
            generations=30  # Increase to 50+ for final paper
        )

        all_results[package] = results

        # Generate plots
        plotter.plot_hypervolume_convergence(results, title=f"Convergence Analysis - {package}")
        plotter.plot_performance_comparison(results)

        # Generate LaTeX table
        plotter.generate_latex_table(results)

    print("\n" + "="*50)
    print("PLOT GENERATION COMPLETE")
    print("="*50)
    print(f"Plots saved in: plots/")
    print("\nGenerated files:")
    print("- hypervolume_convergence.pdf/png")
    print("- performance_comparison.pdf/png")
    print("- performance_table.tex")
    print("\nFor publication, increase n_runs to 30 and generations to 50+")


if __name__ == "__main__":
    main()