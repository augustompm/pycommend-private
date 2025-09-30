"""
MOVNS v18 - Optimized Variable Neighborhood Search
- Population size: 50 (matching MOEA/D)
- Removed 2 least contributing neighborhoods
- Enhanced parameters for better performance
- Based on empirical analysis of neighborhood contributions
"""

import numpy as np
import pickle
import random
import time
from typing import List, Dict, Tuple, Set
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics
from optimizer.movns_v2 import MOVNS_V2


class MOVNS_V18(MOVNS_V2):
    """
    Optimized MOVNS with enhanced parameters and reduced neighborhoods
    Population/Archive size: 50 (for fair comparison)
    Removed n3_segment_exchange and n5_diversity_injection (lowest contribution)
    """

    def __init__(self, main_package, archive_size=50, max_iterations=50,
                 track_metrics=True):
        # Initialize with 50 archive size (matching MOEA/D population)
        super().__init__(main_package, archive_size, max_iterations,
                        k_max=4, track_metrics=track_metrics, min_no_improvement=15)

        # Enhanced SA parameters
        self.temperature = 1.5  # Higher initial temperature
        self.cooling_rate = 0.98  # Slower cooling
        self.min_temperature = 0.001

        # Larger tabu list for better diversification
        self.tabu_list = deque(maxlen=100)
        self.tabu_tenure = 30

        # Adjusted learning rates for 4 neighborhoods
        self.learning_rates = np.ones(4) * 0.7  # Higher initial rates
        self.neighborhood_success = np.zeros(4)
        self.neighborhood_calls = np.zeros(4)

        self.pareto_queue = deque()
        self.intensification_memory = []
        self.diversification_memory = set()

        # Optimized adaptive parameters
        self.adaptive_params = {
            'local_search_intensity': 15,  # More intensive
            'perturbation_strength': 0.15,
            'archive_pressure': 0.4,
            'exploration_rate': 0.3  # Less exploration, more exploitation
        }

        print(f"MOVNS v18 initialized with archive_size=50")
        print(f"Using 4 neighborhoods (removed least contributing 2)")

    def get_neighborhood(self, solution: np.ndarray, k: int) -> np.ndarray:
        """
        Get neighborhood with only 4 most effective structures
        Removed: n3_segment_exchange and n5_diversity_injection
        """
        # Map to reduced neighborhood set
        neighborhood_map = {
            0: self.n1_single_flip,
            1: self.n2_multi_flip,
            2: self.n4_smart_adjustment,  # Was n4, now index 2
            3: self.n6_cluster_based     # Was n6, now index 3
        }

        if k in neighborhood_map:
            return neighborhood_map[k](solution)
        return solution

    def n1_single_flip(self, solution: np.ndarray) -> np.ndarray:
        """Single bit flip - fine-tuning"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if len(indices) > self.min_size and len(indices) < self.max_size:
            # Enhanced selection based on contribution
            if np.random.random() < 0.6:  # 60% add
                candidates = self.cooccur_candidates[:30]
                valid = [c for c in candidates if solution[c] == 0]
                if valid:
                    idx = np.random.choice(valid)
                    neighbor[idx] = 1
            else:  # 40% remove
                if len(indices) > self.min_size:
                    idx = np.random.choice(indices)
                    neighbor[idx] = 0

        return neighbor

    def n2_multi_flip(self, solution: np.ndarray) -> np.ndarray:
        """Multi-bit flip - medium perturbation"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        # Flip 2-4 bits with bias toward quality
        num_flips = np.random.randint(2, 5)

        for _ in range(num_flips):
            if np.random.random() < 0.5 and len(indices) < self.max_size:
                # Add from high-quality candidates
                candidates = self.semantic_candidates[:40]
                valid = [c for c in candidates if neighbor[c] == 0]
                if valid:
                    idx = np.random.choice(valid)
                    neighbor[idx] = 1
                    indices = np.where(neighbor == 1)[0]
            elif len(indices) > self.min_size:
                # Remove low contribution
                idx = np.random.choice(indices)
                neighbor[idx] = 0
                indices = np.where(neighbor == 1)[0]

        return neighbor

    def n4_smart_adjustment(self, solution: np.ndarray) -> np.ndarray:
        """Smart domain-specific adjustment"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]
        current_size = len(indices)

        # Target optimal size with higher probability
        if current_size != self.ideal_size:
            target_size = self.ideal_size
            diff = target_size - current_size

            if diff > 0:  # Need to add
                candidates = np.concatenate([
                    self.cooccur_candidates[:20],
                    self.semantic_candidates[:20]
                ])
                candidates = np.unique(candidates)
                valid = [c for c in candidates if neighbor[c] == 0]

                for _ in range(min(diff, len(valid))):
                    if valid:
                        idx = valid.pop(np.random.randint(len(valid)))
                        neighbor[idx] = 1

            elif diff < 0:  # Need to remove
                for _ in range(min(-diff, len(indices) - self.min_size)):
                    current_indices = np.where(neighbor == 1)[0]
                    if len(current_indices) > self.min_size:
                        idx = np.random.choice(current_indices)
                        neighbor[idx] = 0

        return neighbor

    def n6_cluster_based(self, solution: np.ndarray) -> np.ndarray:
        """Cluster-based neighborhood - semantic coherence"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if len(indices) > 0 and hasattr(self, 'clusters'):
            # Find dominant cluster
            cluster_counts = {}
            for idx in indices:
                cluster = self.clusters[idx]
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

            if cluster_counts:
                dominant_cluster = max(cluster_counts, key=cluster_counts.get)

                # Add from same cluster
                if len(indices) < self.max_size:
                    same_cluster = np.where(self.clusters == dominant_cluster)[0]
                    valid = [idx for idx in same_cluster if neighbor[idx] == 0]
                    if valid:
                        to_add = np.random.choice(valid,
                                                min(2, len(valid)), replace=False)
                        for idx in to_add:
                            neighbor[idx] = 1

                # Remove from different clusters
                if len(indices) > self.min_size:
                    different = [idx for idx in indices
                               if self.clusters[idx] != dominant_cluster]
                    if different:
                        to_remove = np.random.choice(different,
                                                   min(1, len(different)), replace=False)
                        for idx in to_remove:
                            neighbor[idx] = 0

        return neighbor

    def pareto_local_search(self, solution: np.ndarray, max_neighbors: int = 30) -> List[np.ndarray]:
        """Enhanced Pareto Local Search with more neighbors"""
        pareto_set = []
        queue = deque([solution])
        evaluated = set()

        while queue and len(evaluated) < max_neighbors:
            current = queue.popleft()
            current_tuple = tuple(current)

            if current_tuple in evaluated:
                continue
            evaluated.add(current_tuple)

            current_obj = self.evaluate_objectives(current)

            # Generate diverse neighbors
            neighbors = []
            for k in range(self.k_max):
                for _ in range(2):  # Multiple neighbors per structure
                    neighbor = self.get_neighborhood(current, k)
                    neighbors.append(neighbor)

            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in evaluated and neighbor_tuple not in self.tabu_list:
                    neighbor_obj = self.evaluate_objectives(neighbor)

                    is_dominated = False
                    for sol, obj in pareto_set:
                        if self.dominates(obj, neighbor_obj):
                            is_dominated = True
                            break

                    if not is_dominated:
                        pareto_set = [(s, o) for s, o in pareto_set
                                     if not self.dominates(neighbor_obj, o)]
                        pareto_set.append((neighbor, neighbor_obj))

                        if self.dominates(neighbor_obj, current_obj) or len(queue) < 30:
                            queue.append(neighbor)

        return [sol for sol, _ in pareto_set]

    def run(self) -> List[Dict]:
        """Main MOVNS loop with optimized parameters"""
        print(f"\nStarting MOVNS v18 with optimized settings...")
        print(f"Settings: {self.max_iterations} iterations, {self.archive_limit} archive size")

        iteration = 0
        no_improvement = 0
        best_hv = 0
        global_best_lu = float('-inf')

        while iteration < self.max_iterations:
            # Archive-based selection with higher pressure
            if len(self.archive) > 0:
                if np.random.random() < self.adaptive_params['exploration_rate']:
                    # Exploration: random solution
                    current_idx = np.random.randint(len(self.archive))
                else:
                    # Exploitation: quality-biased selection
                    if len(self.archive) > 10:
                        archive_objectives = []
                        for sol in self.archive:
                            obj = self.evaluate_objectives(sol['chromosome'])
                            archive_objectives.append(obj)

                        # Select based on hypervolume contribution
                        contributions = []
                        for i, obj in enumerate(archive_objectives):
                            hv_contribution = -obj[0] * 0.5 - obj[1] * 0.3 + (15 - obj[2]) * 0.2
                            contributions.append(hv_contribution)

                        probabilities = np.array(contributions)
                        probabilities = probabilities - probabilities.min() + 1
                        probabilities = probabilities / probabilities.sum()
                        current_idx = np.random.choice(len(self.archive), p=probabilities)
                    else:
                        current_idx = np.random.randint(len(self.archive))

                current_solution = self.archive[current_idx]['chromosome'].copy()
                current_objectives = self.archive[current_idx]['objectives'].copy()
            else:
                current_solution = self.smart_initialization(exploration_rate=0.3)[0]
                current_objectives = self.evaluate_objectives(current_solution)
                self.update_archive(current_solution, current_objectives)

            # VNS main loop with 4 neighborhoods
            k = 0
            while k < self.k_max:
                # Shaking with adaptive neighborhood selection
                if np.random.random() < 0.7:  # 70% adaptive
                    k = self.select_neighborhood_adaptively()

                neighbor = self.get_neighborhood(current_solution, k)
                neighbor_objectives = self.evaluate_objectives(neighbor)

                # Pareto Local Search with higher intensity
                if np.random.random() < 0.6:  # 60% probability
                    pareto_neighbors = self.pareto_local_search(neighbor,
                                                               self.adaptive_params['local_search_intensity'])

                    for pn in pareto_neighbors:
                        pn_objectives = self.evaluate_objectives(pn)
                        self.update_archive(pn, pn_objectives)

                        if self.dominates(pn_objectives, neighbor_objectives):
                            neighbor = pn
                            neighbor_objectives = pn_objectives

                # Update with SA acceptance
                improved = False
                if self.dominates(neighbor_objectives, current_objectives):
                    current_solution = neighbor
                    current_objectives = neighbor_objectives
                    self.update_archive(neighbor, neighbor_objectives)
                    improved = True
                    k = 0
                elif not self.dominates(current_objectives, neighbor_objectives):
                    # SA acceptance for non-dominated
                    delta = np.sum(np.abs(neighbor_objectives - current_objectives))
                    if np.random.random() < np.exp(-delta / self.temperature):
                        current_solution = neighbor
                        current_objectives = neighbor_objectives
                        self.update_archive(neighbor, neighbor_objectives)
                        improved = True
                        k = 0
                    else:
                        k += 1
                else:
                    k += 1

                # Update learning rates
                self.update_learning_rate(k % self.k_max, improved)

                # Update tabu list
                self.tabu_list.append(tuple(neighbor))

                # Track best LU
                current_lu = -current_objectives[0]
                if current_lu > global_best_lu:
                    global_best_lu = current_lu
                    print(f"New global best: LU={global_best_lu:.0f}, "
                         f"SS={-current_objectives[1]:.4f}, RSS={current_objectives[2]:.1f}")

            # Cool down temperature
            self.temperature = max(self.min_temperature,
                                 self.temperature * self.cooling_rate)

            # Update adaptive parameters
            if iteration % 10 == 0:
                self.update_adaptive_parameters(iteration)

            # Track metrics and convergence
            if iteration % 5 == 0:
                current_hv = self.calculate_current_hypervolume()

                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improvement = 0
                    improvement_str = " (improved)"
                else:
                    no_improvement += 1
                    improvement_str = ""

                print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}{improvement_str}, Temperature={self.temperature:.3f}")

                if no_improvement >= self.min_no_improvement:
                    print(f"Converged after {iteration} iterations (no improvement for {no_improvement} checks)")
                    break

            iteration += 1

        print(f"\nMOVNS v18 completed:")
        print(f"  Final archive: {len(self.archive)} solutions")
        print(f"  Best hypervolume: {best_hv:.4f}")
        print(f"  Global best LU: {global_best_lu:.0f}")

        # Return archive
        return self.format_output()

    def select_neighborhood_adaptively(self) -> int:
        """Select neighborhood based on success rates (4 neighborhoods)"""
        success_rates = []
        for i in range(self.k_max):
            if self.neighborhood_calls[i] > 0:
                rate = self.neighborhood_success[i] / self.neighborhood_calls[i]
                success_rates.append(rate * self.learning_rates[i])
            else:
                success_rates.append(self.learning_rates[i])

        # Softmax selection
        success_rates = np.array(success_rates)
        probabilities = np.exp(success_rates) / np.sum(np.exp(success_rates))

        return np.random.choice(self.k_max, p=probabilities)

    def update_learning_rate(self, neighborhood_idx: int, improved: bool):
        """Update learning rates based on performance"""
        if neighborhood_idx >= self.k_max:
            return

        self.neighborhood_calls[neighborhood_idx] += 1

        if improved:
            self.neighborhood_success[neighborhood_idx] += 1
            self.learning_rates[neighborhood_idx] = min(1.0,
                self.learning_rates[neighborhood_idx] * 1.1)  # Faster increase
        else:
            self.learning_rates[neighborhood_idx] = max(0.1,
                self.learning_rates[neighborhood_idx] * 0.95)

    def update_adaptive_parameters(self, iteration: int):
        """Update adaptive parameters during search"""
        progress = iteration / self.max_iterations

        # Increase local search intensity over time
        self.adaptive_params['local_search_intensity'] = min(30,
            int(15 + progress * 15))

        # Decrease exploration over time
        self.adaptive_params['exploration_rate'] = max(0.1,
            0.3 - progress * 0.2)

        # Adjust perturbation strength
        if len(self.archive) < self.archive_limit // 2:
            self.adaptive_params['perturbation_strength'] = min(0.3,
                self.adaptive_params['perturbation_strength'] * 1.1)
        else:
            self.adaptive_params['perturbation_strength'] = max(0.05,
                self.adaptive_params['perturbation_strength'] * 0.95)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOVNS v18 for package: {package_name}")
    movns = MOVNS_V18(package_name, archive_size=50, max_iterations=50)
    solutions = movns.run()

    print(f"\nFinal results:")
    print(f"Archive size: {len(solutions)}")
    if solutions:
        print(f"Sample solution: {solutions[0]['packages'][:5]}")