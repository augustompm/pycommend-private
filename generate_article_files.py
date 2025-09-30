"""
Generate CSV and PNG files for article
Create convergence data and plots for both MOVNS and MOEA/D
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("Generating Article Data and Plots")
print("="*70)

def normalize_objectives(objectives):
    """Correct normalization for metrics calculation"""
    norm = objectives.copy()
    norm[:, 0] = -norm[:, 0] / 20000  # LU: higher is better
    norm[:, 1] = -norm[:, 1]  # SS: higher is better
    norm[:, 2] = (20 - norm[:, 2]) / 20  # RSS: lower is better (inverted)
    return norm

def run_movns_full(iterations=30):
    """Run MOVNS and track metrics at each iteration"""
    movns = MOVNS_V22('fastapi', archive_size=50, max_iterations=iterations, track_metrics=False)

    # Initialize
    if len(movns.archive) == 0:
        initial = movns.smart_initialization(exploration_rate=0.3)[0]
        initial_obj = movns.evaluate_objectives(initial)
        movns.update_archive(initial, initial_obj)

    metrics = []
    ref_objectives = np.array([sol['objectives'] for sol in movns.archive])

    for iteration in range(iterations):
        # VNS iteration
        if len(movns.archive) > 0:
            current_idx = np.random.randint(len(movns.archive))
            current = movns.archive[current_idx]['chromosome'].copy()
            current_obj = movns.archive[current_idx]['objectives'].copy()

        k = 0
        while k < movns.k_max:
            neighbor = movns.get_neighborhood(current, k)
            neighbor_obj = movns.evaluate_objectives(neighbor)

            if np.random.random() < movns.pls_probability:
                improved_neighbor, improved = movns.simple_local_search(neighbor, movns.pls_max_neighbors)
                if improved:
                    neighbor = improved_neighbor
                    neighbor_obj = movns.evaluate_objectives(neighbor)

            movns.update_archive(neighbor, neighbor_obj)

            if movns.dominates(neighbor_obj, current_obj):
                current = neighbor
                current_obj = neighbor_obj
                k = 0
            else:
                k += 1

        # Calculate metrics
        if len(movns.archive) > 0:
            objectives = np.array([sol['objectives'] for sol in movns.archive])
            qm = QualityMetrics()

            norm = normalize_objectives(objectives)
            hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
            spacing = qm.spacing(objectives)
            epsilon = qm.epsilon_indicator(objectives, reference_set=ref_objectives)

            metrics.append({
                'iteration': iteration + 1,
                'hypervolume': hv,
                'spacing': spacing,
                'epsilon': epsilon,
                'archive_size': len(movns.archive)
            })

    return metrics

def run_moead_full(generations=30):
    """Run MOEA/D and track metrics at each generation"""
    moead = MOEAD_V18('fastapi', pop_size=30, n_neighbors=5, max_gen=generations, theta=1.0, track_metrics=False)

    metrics = []
    initial_objectives = np.array([ind['objectives'] for ind in moead.population])

    for gen in range(generations):
        if gen > 0:
            for i in range(moead.pop_size):
                neighbors = moead.B[i]

                if np.random.rand() < 0.9:
                    parents_idx = np.random.choice(neighbors, 2, replace=False)
                else:
                    parents_idx = np.random.choice(moead.pop_size, 2, replace=False)

                parent1 = moead.population[parents_idx[0]]
                parent2 = moead.population[parents_idx[1]]

                offspring = moead.crossover(parent1['chromosome'], parent2['chromosome'])
                offspring = moead.mutation(offspring)
                off_obj = moead.evaluate_objectives(offspring)

                moead.z = np.minimum(moead.z, off_obj)
                moead.nadir = np.maximum(moead.nadir, off_obj)

                updated = 0
                for j in neighbors:
                    if updated >= moead.theta:
                        break

                    current_fitness = moead.decompose(moead.population[j]['objectives'], moead.weights[j])
                    offspring_fitness = moead.decompose(off_obj, moead.weights[j])

                    if offspring_fitness < current_fitness:
                        moead.population[j] = {
                            'chromosome': offspring.copy(),
                            'objectives': off_obj.copy()
                        }
                        updated += 1

        # Calculate metrics
        objectives = np.array([ind['objectives'] for ind in moead.population])
        qm = QualityMetrics()

        norm = normalize_objectives(objectives)
        hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])
        spacing = qm.spacing(objectives)
        epsilon = qm.epsilon_indicator(objectives, reference_set=initial_objectives)

        # Count non-dominated
        pareto = qm.get_non_dominated_set(objectives)

        metrics.append({
            'generation': gen + 1,
            'hypervolume': hv,
            'spacing': spacing,
            'epsilon': epsilon,
            'population_size': len(moead.population),
            'pareto_size': len(pareto)
        })

    return metrics

print("\nRunning MOVNS (30 iterations)...")
movns_data = run_movns_full(30)

print("Running MOEA/D (30 generations)...")
moead_data = run_moead_full(30)

# Create DataFrames
df_movns = pd.DataFrame(movns_data)
df_moead = pd.DataFrame(moead_data)

# Save CSVs
os.makedirs('../article', exist_ok=True)

df_movns.to_csv('../article/movns_convergence.csv', index=False)
df_moead.to_csv('../article/moead_convergence.csv', index=False)

print("\nCSV files saved:")
print("  - article/movns_convergence.csv")
print("  - article/moead_convergence.csv")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MOVNS vs MOEA/D Convergence Comparison', fontsize=16, fontweight='bold')

# Plot 1: Hypervolume comparison
ax1 = axes[0, 0]
ax1.plot(df_movns['iteration'], df_movns['hypervolume'], 'b-', linewidth=2, label='MOVNS')
ax1.plot(df_moead['generation'], df_moead['hypervolume'], 'r--', linewidth=2, label='MOEA/D')
ax1.set_xlabel('Iteration/Generation')
ax1.set_ylabel('Hypervolume')
ax1.set_title('Hypervolume Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Spacing comparison
ax2 = axes[0, 1]
ax2.plot(df_movns['iteration'], df_movns['spacing'], 'b-', linewidth=2, label='MOVNS')
ax2.plot(df_moead['generation'], df_moead['spacing'], 'r--', linewidth=2, label='MOEA/D')
ax2.set_xlabel('Iteration/Generation')
ax2.set_ylabel('Spacing')
ax2.set_title('Spacing Evolution (lower is better)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Archive/Population size
ax3 = axes[1, 0]
ax3.plot(df_movns['iteration'], df_movns['archive_size'], 'b-', linewidth=2, label='MOVNS Archive')
ax3.plot(df_moead['generation'], df_moead['population_size'], 'r--', linewidth=2, label='MOEA/D Population')
ax3.plot(df_moead['generation'], df_moead['pareto_size'], 'g:', linewidth=2, label='MOEA/D Pareto')
ax3.set_xlabel('Iteration/Generation')
ax3.set_ylabel('Size')
ax3.set_title('Archive/Population Size')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Final comparison bar chart
ax4 = axes[1, 1]
final_movns = df_movns.iloc[-1]
final_moead = df_moead.iloc[-1]

metrics = ['Hypervolume', 'Spacing\n(inverted)', 'Archive Size']
movns_values = [final_movns['hypervolume'], 1/final_movns['spacing'], final_movns['archive_size']/10]
moead_values = [final_moead['hypervolume'], 1/final_moead['spacing'], final_moead['population_size']/10]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, movns_values, width, label='MOVNS', color='blue', alpha=0.7)
bars2 = ax4.bar(x + width/2, moead_values, width, label='MOEA/D', color='red', alpha=0.7)

ax4.set_ylabel('Normalized Values')
ax4.set_title('Final Metrics Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../article/convergence_comparison.png', dpi=150)

print("\nPNG file saved:")
print("  - article/convergence_comparison.png")

# Create summary CSV
summary_data = {
    'Algorithm': ['MOVNS', 'MOEA/D'],
    'Final_HV': [final_movns['hypervolume'], final_moead['hypervolume']],
    'Final_Spacing': [final_movns['spacing'], final_moead['spacing']],
    'Final_Size': [final_movns['archive_size'], final_moead['population_size']],
    'Iterations': [30, 30],
    'HV_Improvement': [
        (final_movns['hypervolume'] / df_movns.iloc[0]['hypervolume'] - 1) * 100,
        (final_moead['hypervolume'] / df_moead.iloc[0]['hypervolume'] - 1) * 100
    ],
    'Spacing_Improvement': [
        (df_movns.iloc[0]['spacing'] / final_movns['spacing'] - 1) * 100,
        (df_moead.iloc[0]['spacing'] / final_moead['spacing'] - 1) * 100
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('../article/comparison_summary.csv', index=False)

print("\nSummary CSV saved:")
print("  - article/comparison_summary.csv")

print("\n" + "="*70)
print("Article files generated successfully!")
print("="*70)
print(f"\nFinal Results:")
print(f"MOVNS: HV={final_movns['hypervolume']:.4f}, Spacing={final_movns['spacing']:.4f}")
print(f"MOEA/D: HV={final_moead['hypervolume']:.4f}, Spacing={final_moead['spacing']:.4f}")
print("\nMOVNS wins in:")
if final_movns['hypervolume'] > final_moead['hypervolume']:
    print(f"  - Hypervolume: {(final_movns['hypervolume']/final_moead['hypervolume']-1)*100:.1f}% better")
if final_movns['spacing'] < final_moead['spacing']:
    print(f"  - Spacing: {(final_moead['spacing']/final_movns['spacing']-1)*100:.1f}% better")