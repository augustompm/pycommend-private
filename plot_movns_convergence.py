"""
Generate MOVNS convergence plots
Track HV, Spacing, and Epsilon indicators over iterations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOVNS Convergence Analysis")
print("="*70)

def normalize_objectives(objectives):
    """Correct normalization for metrics calculation"""
    norm = objectives.copy()
    norm[:, 0] = -norm[:, 0] / 20000  # LU: higher is better
    norm[:, 1] = -norm[:, 1]  # SS: higher is better
    norm[:, 2] = (20 - norm[:, 2]) / 20  # RSS: lower is better (inverted)
    return norm

# Run MOVNS and track metrics at each iteration
iterations = 30
n_runs = 3

all_hv = []
all_spacing = []
all_epsilon = []
all_archive_size = []

for run in range(n_runs):
    print(f"\nRun {run+1}/{n_runs}")

    movns = MOVNS_V22('fastapi', archive_size=50, max_iterations=iterations, track_metrics=False)

    # Initialize
    if len(movns.archive) == 0:
        initial = movns.smart_initialization(exploration_rate=0.3)[0]
        initial_obj = movns.evaluate_objectives(initial)
        movns.update_archive(initial, initial_obj)

    hv_history = []
    spacing_history = []
    epsilon_history = []
    archive_history = []

    # Reference set for epsilon indicator (initial archive)
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

            # Normalize for HV
            norm = normalize_objectives(objectives)
            hv = qm.hypervolume(norm, ref_point=[1.1, 1.1, 1.1])

            # Spacing (raw objectives)
            spacing = qm.spacing(objectives)

            # Epsilon indicator
            epsilon = qm.epsilon_indicator(objectives, reference_set=ref_objectives)

            hv_history.append(hv)
            spacing_history.append(spacing)
            epsilon_history.append(epsilon)
            archive_history.append(len(movns.archive))

        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1}: HV={hv:.4f}, Spacing={spacing:.4f}, Archive={len(movns.archive)}")

    all_hv.append(hv_history)
    all_spacing.append(spacing_history)
    all_epsilon.append(epsilon_history)
    all_archive_size.append(archive_history)

# Calculate mean and std
mean_hv = np.mean(all_hv, axis=0)
std_hv = np.std(all_hv, axis=0)
mean_spacing = np.mean(all_spacing, axis=0)
std_spacing = np.std(all_spacing, axis=0)
mean_epsilon = np.mean(all_epsilon, axis=0)
std_epsilon = np.std(all_epsilon, axis=0)
mean_archive = np.mean(all_archive_size, axis=0)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('MOVNS Convergence Analysis (3 runs)', fontsize=16, fontweight='bold')

x = range(1, iterations + 1)

# Plot 1: Hypervolume
ax1 = axes[0, 0]
ax1.plot(x, mean_hv, 'b-', linewidth=2, label='Mean HV')
ax1.fill_between(x, mean_hv - std_hv, mean_hv + std_hv, alpha=0.3, color='blue')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Hypervolume')
ax1.set_title('Hypervolume Evolution')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Spacing
ax2 = axes[0, 1]
ax2.plot(x, mean_spacing, 'r-', linewidth=2, label='Mean Spacing')
ax2.fill_between(x, mean_spacing - std_spacing, mean_spacing + std_spacing, alpha=0.3, color='red')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Spacing')
ax2.set_title('Spacing Evolution (lower is better)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Epsilon Indicator
ax3 = axes[1, 0]
ax3.plot(x, mean_epsilon, 'g-', linewidth=2, label='Mean Epsilon')
ax3.fill_between(x, mean_epsilon - std_epsilon, mean_epsilon + std_epsilon, alpha=0.3, color='green')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Epsilon Indicator')
ax3.set_title('Epsilon Indicator Evolution')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Archive Size
ax4 = axes[1, 1]
ax4.plot(x, mean_archive, 'm-', linewidth=2, label='Archive Size')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Archive Size')
ax4.set_title('Archive Growth')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Max Archive')
ax4.legend()

plt.tight_layout()
plt.savefig('movns_convergence.png', dpi=150)
print(f"\nConvergence plot saved as 'movns_convergence.png'")

# Print final statistics
print("\n" + "="*70)
print("Final Metrics (iteration 30):")
print("="*70)
print(f"Hypervolume: {mean_hv[-1]:.4f} ± {std_hv[-1]:.4f}")
print(f"Spacing: {mean_spacing[-1]:.4f} ± {std_spacing[-1]:.4f}")
print(f"Epsilon: {mean_epsilon[-1]:.4f} ± {std_epsilon[-1]:.4f}")
print(f"Archive Size: {mean_archive[-1]:.1f}")
print("="*70)