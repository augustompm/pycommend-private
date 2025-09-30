"""
MOVNS v20 - Aggressively optimized for speed
- Reduced Pareto Local Search neighbors (10 instead of 30)
- Lower probability of PLS (30% instead of 60%)
- Faster neighborhood operations
- Based on v19 with more aggressive trade-offs
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


class MOVNS_V20(MOVNS_V2):
    """
    Aggressively speed-optimized MOVNS
    Reduces computational complexity while maintaining core algorithm
    """

    def __init__(self, main_package, archive_size=50, max_iterations=50,
                 track_metrics=True):
        # Initialize cache before super
        self.objective_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Reduced k_max to 4 neighborhoods
        super().__init__(main_package, archive_size, max_iterations,
                        k_max=4, track_metrics=track_metrics, min_no_improvement=10)

        # Simplified parameters for speed
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.min_temperature = 0.01

        # Smaller tabu list
        self.tabu_list = deque(maxlen=20)

        self.learning_rates = np.ones(4) * 0.5
        self.neighborhood_success = np.zeros(4)
        self.neighborhood_calls = np.zeros(4)

        # Aggressive optimization parameters
        self.pls_neighbors = 10  # Reduced from 30
        self.pls_probability = 0.3  # Reduced from 0.6
        self.local_search_intensity = 5  # Reduced from 15

        print(f"MOVNS v20 initialized - aggressively optimized for speed")
        print(f"PLS: {self.pls_neighbors} neighbors, {self.pls_probability*100}% probability")

    def chromosome_to_key(self, chromosome):
        """Convert chromosome to hashable key for caching"""
        return tuple(np.where(chromosome == 1)[0])

    def evaluate_objectives(self, chromosome):
        """Cached evaluation of objectives"""
        key = self.chromosome_to_key(chromosome)

        if key in self.objective_cache:
            self.cache_hits += 1
            return self.objective_cache[key].copy()

        self.cache_misses += 1
        indices = np.array(key)

        # Fast calculations
        lu_score = self.calculate_linked_usage_fast(indices)
        ss_score = self.calculate_semantic_similarity_fast(indices)
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        # Cache result
        if len(self.objective_cache) < 3000:  # Smaller cache
            self.objective_cache[key] = objectives.copy()

        return objectives

    def calculate_linked_usage_fast(self, indices):
        """Vectorized linked usage"""
        if len(indices) == 0:
            return 0

        if hasattr(self.rel_matrix, 'toarray'):
            submatrix = self.rel_matrix[indices][:, indices].toarray()
        else:
            submatrix = self.rel_matrix[np.ix_(indices, indices)]

        return submatrix.sum() - np.diagonal(submatrix).sum()

    def calculate_semantic_similarity_fast(self, indices):
        """Vectorized semantic similarity"""
        if len(indices) <= 1:
            return 0

        embeddings_subset = self.embeddings[indices]
        centroid = embeddings_subset.mean(axis=0)

        # Simplified calculation
        dots = embeddings_subset @ centroid
        norms = np.linalg.norm(embeddings_subset, axis=1) * np.linalg.norm(centroid)
        similarities = dots / (norms + 1e-10)

        return similarities.mean()

    def get_neighborhood(self, solution: np.ndarray, k: int) -> np.ndarray:
        """Simplified neighborhood structures"""
        if k == 0:
            return self.n1_single_flip(solution)
        elif k == 1:
            return self.n2_double_flip(solution)
        elif k == 2:
            return self.n3_targeted_swap(solution)
        else:
            return self.n4_size_adjustment(solution)

    def n1_single_flip(self, solution: np.ndarray) -> np.ndarray:
        """Single bit flip"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if np.random.random() < 0.5 and len(indices) < self.max_size:
            # Add
            candidates = self.cooccur_candidates[:20]
            valid = [c for c in candidates if solution[c] == 0]
            if valid:
                neighbor[np.random.choice(valid)] = 1
        elif len(indices) > self.min_size:
            # Remove
            neighbor[np.random.choice(indices)] = 0

        return neighbor

    def n2_double_flip(self, solution: np.ndarray) -> np.ndarray:
        """Double bit flip"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        # Flip 2 bits
        for _ in range(2):
            if np.random.random() < 0.5 and len(np.where(neighbor == 1)[0]) < self.max_size:
                candidates = self.semantic_candidates[:20]
                valid = [c for c in candidates if neighbor[c] == 0]
                if valid:
                    neighbor[np.random.choice(valid)] = 1
            elif len(np.where(neighbor == 1)[0]) > self.min_size:
                current_indices = np.where(neighbor == 1)[0]
                if len(current_indices) > self.min_size:
                    neighbor[np.random.choice(current_indices)] = 0

        return neighbor

    def n3_targeted_swap(self, solution: np.ndarray) -> np.ndarray:
        """Targeted swap based on quality"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if len(indices) >= 2:
            # Remove one, add one
            to_remove = np.random.choice(indices)
            neighbor[to_remove] = 0

            candidates = self.cooccur_candidates[:30]
            valid = [c for c in candidates if neighbor[c] == 0]
            if valid:
                neighbor[np.random.choice(valid)] = 1

        return neighbor

    def n4_size_adjustment(self, solution: np.ndarray) -> np.ndarray:
        """Adjust toward ideal size"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]
        current_size = len(indices)

        if current_size < self.ideal_size:
            # Add packages
            to_add = min(2, self.ideal_size - current_size)
            candidates = self.semantic_candidates[:30]
            valid = [c for c in candidates if neighbor[c] == 0]
            if valid:
                for idx in np.random.choice(valid, min(to_add, len(valid)), replace=False):
                    neighbor[idx] = 1

        elif current_size > self.ideal_size:
            # Remove packages
            to_remove = min(2, current_size - self.ideal_size)
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable:
                for idx in np.random.choice(removable, min(to_remove, len(removable)), replace=False):
                    neighbor[idx] = 0

        return neighbor

    def pareto_local_search_fast(self, solution: np.ndarray) -> List[np.ndarray]:
        """Simplified Pareto Local Search"""
        pareto_set = []
        current_obj = self.evaluate_objectives(solution)

        # Generate limited neighbors
        for k in range(self.k_max):
            neighbor = self.get_neighborhood(solution, k)
            neighbor_obj = self.evaluate_objectives(neighbor)

            # Simple dominance check
            if self.dominates(neighbor_obj, current_obj):
                pareto_set = [neighbor]
                current_obj = neighbor_obj
            elif not self.dominates(current_obj, neighbor_obj):
                is_dominated = False
                for sol in pareto_set:
                    sol_obj = self.evaluate_objectives(sol)
                    if self.dominates(sol_obj, neighbor_obj):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_set.append(neighbor)

        return pareto_set if pareto_set else [solution]

    def run(self) -> List[Dict]:
        """Optimized main loop"""
        print(f"\nStarting MOVNS v20 (aggressively optimized)...")
        print(f"Settings: {self.max_iterations} iterations, {self.archive_limit} archive")

        start_time = time.time()
        iteration = 0
        no_improvement = 0
        best_hv = 0

        while iteration < self.max_iterations:
            # Select from archive
            if len(self.archive) > 0:
                current_idx = np.random.randint(len(self.archive))
                current_solution = self.archive[current_idx]['chromosome'].copy()
                current_objectives = self.archive[current_idx]['objectives'].copy()
            else:
                current_solution = self.smart_initialization(exploration_rate=0.3)[0]
                current_objectives = self.evaluate_objectives(current_solution)
                self.update_archive(current_solution, current_objectives)

            # Simplified VNS loop
            k = 0
            while k < self.k_max:
                neighbor = self.get_neighborhood(current_solution, k)
                neighbor_objectives = self.evaluate_objectives(neighbor)

                # Reduced PLS probability
                if np.random.random() < self.pls_probability:
                    pls_solutions = self.pareto_local_search_fast(neighbor)
                    for pls_sol in pls_solutions[:self.local_search_intensity]:
                        pls_obj = self.evaluate_objectives(pls_sol)
                        self.update_archive(pls_sol, pls_obj)
                        if self.dominates(pls_obj, neighbor_objectives):
                            neighbor = pls_sol
                            neighbor_objectives = pls_obj

                # Update solution
                if self.dominates(neighbor_objectives, current_objectives):
                    current_solution = neighbor
                    current_objectives = neighbor_objectives
                    self.update_archive(neighbor, neighbor_objectives)
                    k = 0
                elif not self.dominates(current_objectives, neighbor_objectives):
                    # Simulated annealing
                    delta = np.sum(np.abs(neighbor_objectives - current_objectives))
                    if np.random.random() < np.exp(-delta / self.temperature):
                        current_solution = neighbor
                        current_objectives = neighbor_objectives
                        self.update_archive(neighbor, neighbor_objectives)
                        k = 0
                    else:
                        k += 1
                else:
                    k += 1

                # Update tabu
                self.tabu_list.append(tuple(neighbor))

            # Cool temperature
            self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)

            # Track progress
            if iteration % 5 == 0:
                if self.track_metrics and len(self.archive) > 0:
                    objectives = np.array([sol['objectives'] for sol in self.archive])
                    normalized = np.array([self.normalize_objectives(obj) for obj in objectives])
                    current_hv = self.metrics_calculator.hypervolume(normalized) if hasattr(self, 'metrics_calculator') else 0
                else:
                    current_hv = 0

                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improvement = 0
                else:
                    no_improvement += 1

                elapsed = time.time() - start_time
                cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0

                print(f"Iter {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}, Cache={cache_rate:.1f}%, "
                     f"Time={elapsed:.1f}s")

                if no_improvement >= self.min_no_improvement:
                    print(f"Converged after {iteration} iterations")
                    break

            iteration += 1

        total_time = time.time() - start_time

        print(f"\nMOVNS v20 completed:")
        print(f"  Archive: {len(self.archive)} solutions")
        print(f"  Best HV: {best_hv:.4f}")
        print(f"  Time: {total_time:.1f}s")
        print(f"  Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")

        return self.format_output()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOVNS v20 for: {package_name}")
    movns = MOVNS_V20(package_name, archive_size=50, max_iterations=20)
    solutions = movns.run()

    print(f"\nFinal: {len(solutions)} solutions")
    if solutions:
        print(f"Sample: {solutions[0]['packages'][:5]}")