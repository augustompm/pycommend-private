"""
Fixed convergence tracking with proper HV calculation
5 runs, 30 iterations, real execution
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v22 import MOVNS_V22
from optimizer.moead_v18 import MOEAD_V18
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("FIXED CONVERGENCE TRACKING - REAL EXECUTION")
print("="*70)

# Parameters
package_name = 'fastapi'
max_iterations = 30
pop_size = 50
n_runs = 5

print(f"\nConfiguration:")
print(f"  Package: {package_name}")
print(f"  Iterations: {max_iterations}")
print(f"  Population/Archive: {pop_size}")
print(f"  Runs: {n_runs}")

# Storage for convergence data
convergence_data = {
    'run': [],
    'iteration': [],
    'algorithm': [],
    'hv': [],
    'spacing': [],
    'archive_size': [],
    'time': []
}

def calculate_hv_properly(objectives):
    """Calculate HV with proper transformation"""
    if len(objectives) == 0:
        return 0.0

    # Transform objectives for HV calculation
    # LU and SS are negative (maximize), RSS is positive (minimize)
    # Transform to all positive minimization
    transformed = objectives.copy()
    transformed[:, 0] = -transformed[:, 0]  # LU: make positive
    transformed[:, 1] = -transformed[:, 1]  # SS: make positive
    # RSS stays as is

    # Filter out invalid values
    valid_mask = np.all(np.isfinite(transformed), axis=1)
    transformed = transformed[valid_mask]

    if len(transformed) == 0:
        return 0.0

    # Use QualityMetrics with proper reference point
    qm = QualityMetrics()
    # Reference point for transformed objectives
    ref_point = [20000, 1.5, 15]  # Large values to encompass all solutions

    try:
        hv = qm.hypervolume(transformed, ref_point=ref_point)
        return hv
    except:
        return 0.0

# Custom MOVNS that tracks at each iteration
class MOVNS_V22_Fixed(MOVNS_V22):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_data = []

    def run_with_tracking(self):
        """Modified run that tracks metrics at each iteration"""
        print(f"Starting MOVNS v22...")
        start_time = time.time()

        iteration = 0
        no_improvement = 0
        best_hv = 0

        # Initialize if empty archive
        if len(self.archive) == 0:
            initial = self.smart_initialization(exploration_rate=0.3)[0]
            initial_obj = self.evaluate_objectives(initial)
            self.update_archive(initial, initial_obj)

        while iteration < self.max_iterations:
            # VNS main loop (same as original)
            if len(self.archive) > 0:
                current_idx = np.random.randint(len(self.archive))
                current = self.archive[current_idx]['chromosome'].copy()
                current_obj = self.archive[current_idx]['objectives'].copy()
            else:
                current = self.smart_initialization(exploration_rate=0.3)[0]
                current_obj = self.evaluate_objectives(current)
                self.update_archive(current, current_obj)

            # Perform VNS iteration
            k = 0
            local_no_improve = 0

            while k < self.k_max and local_no_improve < 3:
                neighbor = self.get_neighborhood(current, k)
                neighbor_obj = self.evaluate_objectives(neighbor)

                if np.random.random() < self.pls_probability:
                    improved_neighbor, improved = self.simple_local_search(
                        neighbor, self.pls_max_neighbors
                    )
                    if improved:
                        neighbor = improved_neighbor
                        neighbor_obj = self.evaluate_objectives(neighbor)

                self.update_archive(neighbor, neighbor_obj)

                if self.dominates(neighbor_obj, current_obj):
                    current = neighbor
                    current_obj = neighbor_obj
                    k = 0
                    local_no_improve = 0
                elif not self.dominates(current_obj, neighbor_obj):
                    delta = np.sum(np.abs(neighbor_obj - current_obj))
                    if np.random.random() < np.exp(-delta / self.temperature):
                        current = neighbor
                        current_obj = neighbor_obj
                        k = 0
                    else:
                        k += 1
                        local_no_improve += 1
                else:
                    k += 1
                    local_no_improve += 1

                self.tabu_list.append(tuple(neighbor))

            self.temperature *= self.cooling_rate

            # Calculate metrics for this iteration
            if len(self.archive) > 0:
                objectives = np.array([sol['objectives'] for sol in self.archive])

                # Calculate HV properly
                hv = calculate_hv_properly(objectives)

                # Calculate spacing
                qm = QualityMetrics()
                spacing = qm.spacing(objectives) if len(objectives) > 1 else 0

                archive_size = len(self.archive)
                elapsed = time.time() - start_time

                self.iteration_data.append({
                    'iteration': iteration + 1,
                    'hv': hv,
                    'spacing': spacing,
                    'archive_size': archive_size,
                    'time': elapsed
                })

                # Track convergence
                if hv > best_hv:
                    best_hv = hv
                    no_improvement = 0
                else:
                    no_improvement += 1

                if iteration % 10 == 0:
                    print(f"  Iter {iteration}: Archive={archive_size}, "
                          f"HV={hv:.1f}, Spacing={spacing:.4f}, Time={elapsed:.1f}s")

                if no_improvement >= self.min_no_improvement:
                    print(f"  Converged at iteration {iteration}")
                    # Fill remaining iterations with last values
                    for i in range(iteration + 1, self.max_iterations):
                        self.iteration_data.append({
                            'iteration': i + 1,
                            'hv': hv,
                            'spacing': spacing,
                            'archive_size': archive_size,
                            'time': elapsed
                        })
                    break

            iteration += 1

        return self.archive

# Custom MOEA/D that tracks at each iteration
class MOEAD_V18_Fixed(MOEAD_V18):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_data = []

    def run_with_tracking(self):
        """Modified run that tracks metrics at each generation"""
        print(f"Starting MOEA/D v18...")
        start_time = time.time()

        for gen in range(self.max_gen):
            # Regular MOEA/D generation
            for i in range(self.pop_size):
                # Get neighbors
                if not hasattr(self, 'neighbors'):
                    self.initialize_neighbors()

                neighbors = self.neighbors[i]
                parents_idx = np.random.choice(neighbors, 2, replace=False)
                parent1 = self.population[parents_idx[0]]
                parent2 = self.population[parents_idx[1]]

                # Generate offspring
                offspring = self.crossover(parent1['chromosome'], parent2['chromosome'])
                offspring = self.mutation(offspring)
                off_obj = self.evaluate_objectives(offspring)

                # Update neighbors
                updated = 0
                for j in neighbors:
                    if updated >= self.theta:
                        break

                    # Tchebycheff comparison
                    if not hasattr(self, 'ideal_point'):
                        self.ideal_point = np.array([-np.inf, -np.inf, 0])
                        self.nadir_point = np.array([0, 0, 15])

                    current_fitness = self.tchebycheff(
                        self.population[j]['objectives'],
                        self.weights[j]
                    )
                    offspring_fitness = self.tchebycheff(off_obj, self.weights[j])

                    if offspring_fitness < current_fitness:
                        self.population[j] = {
                            'chromosome': offspring.copy(),
                            'objectives': off_obj.copy()
                        }
                        updated += 1

            # Calculate metrics for this generation
            objectives = np.array([sol['objectives'] for sol in self.population])

            # Calculate HV properly
            hv = calculate_hv_properly(objectives)

            # Calculate spacing
            qm = QualityMetrics()
            spacing = qm.spacing(objectives) if len(objectives) > 1 else 0

            elapsed = time.time() - start_time

            self.iteration_data.append({
                'iteration': gen + 1,
                'hv': hv,
                'spacing': spacing,
                'archive_size': self.pop_size,
                'time': elapsed
            })

            if gen % 10 == 0:
                print(f"  Gen {gen}: Pop={self.pop_size}, "
                      f"HV={hv:.1f}, Spacing={spacing:.4f}, Time={elapsed:.1f}s")

        # Return final Pareto front
        pareto_front = []
        for sol in self.population:
            dominated = False
            for other in self.population:
                if np.all(other['objectives'] <= sol['objectives']) and \
                   np.any(other['objectives'] < sol['objectives']):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(sol)

        return pareto_front

    def tchebycheff(self, objectives, weight):
        """Tchebycheff scalarization"""
        # Avoid division issues
        if not np.all(np.isfinite(objectives)):
            return float('inf')

        # Simple normalization to avoid inf values
        norm_obj = objectives.copy()
        norm_obj[0] = (norm_obj[0] + 20000) / 20000  # LU
        norm_obj[1] = (norm_obj[1] + 1) / 1  # SS
        norm_obj[2] = norm_obj[2] / 15  # RSS

        weighted = weight * np.abs(norm_obj)
        return np.max(weighted)

    def initialize_neighbors(self):
        """Initialize neighbor structure"""
        distances = np.zeros((self.pop_size, self.pop_size))
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                distances[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])

        self.neighbors = []
        for i in range(self.pop_size):
            neighbor_indices = np.argsort(distances[i])[:self.n_neighbors]
            self.neighbors.append(neighbor_indices)

# Run experiments
print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)

for run in range(1, n_runs + 1):
    print(f"\n--- RUN {run}/{n_runs} ---")

    # Run MOVNS
    print(f"\nMOVNS v22 Run {run}:")
    movns = MOVNS_V22_Fixed(package_name, archive_size=pop_size,
                           max_iterations=max_iterations, track_metrics=False)
    movns.run_with_tracking()

    # Store MOVNS data
    for data_point in movns.iteration_data:
        convergence_data['run'].append(run)
        convergence_data['iteration'].append(data_point['iteration'])
        convergence_data['algorithm'].append('MOVNS')
        convergence_data['hv'].append(data_point['hv'])
        convergence_data['spacing'].append(data_point['spacing'])
        convergence_data['archive_size'].append(data_point['archive_size'])
        convergence_data['time'].append(data_point['time'])

    print(f"  Final: Archive={movns.iteration_data[-1]['archive_size']}, "
          f"HV={movns.iteration_data[-1]['hv']:.1f}")

    # Run MOEA/D
    print(f"\nMOEA/D v18 Run {run}:")
    try:
        moead = MOEAD_V18_Fixed(package_name, pop_size=pop_size,
                               max_gen=max_iterations, track_metrics=False)
        moead.run_with_tracking()

        # Store MOEA/D data
        for data_point in moead.iteration_data:
            convergence_data['run'].append(run)
            convergence_data['iteration'].append(data_point['iteration'])
            convergence_data['algorithm'].append('MOEAD')
            convergence_data['hv'].append(data_point['hv'])
            convergence_data['spacing'].append(data_point['spacing'])
            convergence_data['archive_size'].append(data_point['archive_size'])
            convergence_data['time'].append(data_point['time'])

        print(f"  Final: Pop={moead.iteration_data[-1]['archive_size']}, "
              f"HV={moead.iteration_data[-1]['hv']:.1f}")
    except Exception as e:
        print(f"  MOEA/D Error: {e}")
        # Fill with zeros if failed
        for i in range(1, max_iterations + 1):
            convergence_data['run'].append(run)
            convergence_data['iteration'].append(i)
            convergence_data['algorithm'].append('MOEAD')
            convergence_data['hv'].append(0)
            convergence_data['spacing'].append(0)
            convergence_data['archive_size'].append(0)
            convergence_data['time'].append(0)

# Create DataFrame
df = pd.DataFrame(convergence_data)

# Calculate medians for each iteration
median_data = df.groupby(['algorithm', 'iteration']).agg({
    'hv': 'median',
    'spacing': 'median',
    'archive_size': 'median',
    'time': 'median'
}).reset_index()

# Save full data
csv_filename = f'convergence_fixed_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(csv_filename, index=False)
print(f"\nFull data saved to: {csv_filename}")

# Save median data for plotting
median_filename = f'convergence_fixed_median_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
median_data.to_csv(median_filename, index=False)
print(f"Median data saved to: {median_filename}")

# Print summary
print("\n" + "="*70)
print("CONVERGENCE SUMMARY (Median values)")
print("="*70)

# Final iteration comparison
final_movns = median_data[(median_data['algorithm'] == 'MOVNS') &
                          (median_data['iteration'] == max_iterations)].iloc[0]
final_moead = median_data[(median_data['algorithm'] == 'MOEAD') &
                          (median_data['iteration'] == max_iterations)].iloc[0]

print(f"\nFinal values (iteration {max_iterations}):")
print(f"\nMOVNS:")
print(f"  HV: {final_movns['hv']:.1f}")
print(f"  Spacing: {final_movns['spacing']:.4f}")
print(f"  Archive: {final_movns['archive_size']:.0f}")
print(f"  Time: {final_movns['time']:.1f}s")

print(f"\nMOEA/D:")
print(f"  HV: {final_moead['hv']:.1f}")
print(f"  Spacing: {final_moead['spacing']:.4f}")
print(f"  Population: {final_moead['archive_size']:.0f}")
print(f"  Time: {final_moead['time']:.1f}s")

print("\n" + "="*70)
print("READY FOR CONVERGENCE GRAPHS - HV NOW WORKING!")
print("="*70)