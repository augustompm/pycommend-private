"""
MOVNS v19 - Speed Optimized with Caching and Vectorization
- Based on v18 with performance optimizations
- Cache for objective evaluations
- Vectorized matrix operations
- Optimized for Ryzen 6 (12 cores)
"""

import numpy as np
import pickle
import random
import time
from typing import List, Dict, Tuple, Set
from collections import deque
from functools import lru_cache
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics
from optimizer.movns_v18 import MOVNS_V18


class MOVNS_V19(MOVNS_V18):
    """
    Speed-optimized MOVNS with caching and vectorization
    Maintains same algorithmic behavior as v18 but much faster
    """

    def __init__(self, main_package, archive_size=50, max_iterations=50,
                 track_metrics=True):
        # Initialize cache before calling super (which calls evaluate_objectives)
        self.objective_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        super().__init__(main_package, archive_size, max_iterations, track_metrics)

        print(f"MOVNS v19 initialized with caching and vectorization")
        print(f"Cache size limit: 5000 evaluations")

    def chromosome_to_key(self, chromosome):
        """Convert chromosome to hashable key for caching"""
        return tuple(np.where(chromosome == 1)[0])

    def evaluate_objectives(self, chromosome):
        """Cached evaluation of objectives"""
        # Check cache first
        key = self.chromosome_to_key(chromosome)

        if key in self.objective_cache:
            self.cache_hits += 1
            return self.objective_cache[key].copy()

        self.cache_misses += 1

        # Compute objectives if not cached
        indices = np.array(key)

        # Fast vectorized calculations
        lu_score = self.calculate_linked_usage_fast(indices)
        ss_score = self.calculate_semantic_similarity_fast(indices)
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        # Store in cache (limit size to prevent memory issues)
        if len(self.objective_cache) < 5000:
            self.objective_cache[key] = objectives.copy()

        return objectives

    def calculate_linked_usage_fast(self, indices):
        """Vectorized linked usage calculation"""
        if len(indices) == 0:
            return 0

        # Extract submatrix and sum (vectorized)
        if hasattr(self.rel_matrix, 'toarray'):
            # Sparse matrix case
            submatrix = self.rel_matrix[indices][:, indices].toarray()
        else:
            # Dense matrix - use advanced indexing
            submatrix = self.rel_matrix[np.ix_(indices, indices)]

        # Sum all connections (excluding diagonal)
        score = submatrix.sum() - np.diagonal(submatrix).sum()
        return score

    def calculate_semantic_similarity_fast(self, indices):
        """Optimized semantic similarity using vectorization"""
        if len(indices) <= 1:
            return 0

        # Vectorized centroid calculation
        embeddings_subset = self.embeddings[indices]
        centroid = embeddings_subset.mean(axis=0)

        # Vectorized cosine similarity
        norms = np.linalg.norm(embeddings_subset, axis=1)
        centroid_norm = np.linalg.norm(centroid)

        # Dot product with centroid (vectorized)
        dots = embeddings_subset @ centroid

        # Cosine similarities
        similarities = dots / (norms * centroid_norm + 1e-10)

        return similarities.mean()

    def pareto_local_search(self, solution: np.ndarray, max_neighbors: int = 30) -> List[np.ndarray]:
        """Optimized Pareto Local Search with batch evaluation"""
        pareto_set = []
        queue = deque([solution])
        evaluated = set()

        # Pre-generate neighbors for batch evaluation
        neighbors_to_eval = []

        while queue and len(evaluated) < max_neighbors:
            current = queue.popleft()
            current_tuple = tuple(current)

            if current_tuple in evaluated:
                continue
            evaluated.add(current_tuple)

            current_obj = self.evaluate_objectives(current)

            # Generate all neighbors at once
            for k in range(self.k_max):
                for _ in range(2):
                    neighbor = self.get_neighborhood(current, k)
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple not in evaluated and neighbor_tuple not in self.tabu_list:
                        neighbors_to_eval.append((neighbor, current_obj))

        # Batch evaluate neighbors (benefits from cache)
        for neighbor, current_obj in neighbors_to_eval:
            neighbor_obj = self.evaluate_objectives(neighbor)

            is_dominated = False
            for sol, obj in pareto_set:
                if self.dominates(obj, neighbor_obj):
                    is_dominated = True
                    break

            if not is_dominated:
                # Remove dominated solutions from pareto set
                pareto_set = [(s, o) for s, o in pareto_set
                             if not self.dominates(neighbor_obj, o)]
                pareto_set.append((neighbor, neighbor_obj))

                # Add to queue if promising
                if self.dominates(neighbor_obj, current_obj):
                    queue.append(neighbor)

        return [sol for sol, _ in pareto_set]

    def run(self) -> List[Dict]:
        """Main MOVNS loop with performance monitoring"""
        print(f"\nStarting MOVNS v19 (speed optimized)...")
        print(f"Settings: {self.max_iterations} iterations, {self.archive_limit} archive size")

        start_time = time.time()
        iteration = 0
        no_improvement = 0
        best_hv = 0
        global_best_lu = float('-inf')

        while iteration < self.max_iterations:
            # Archive-based selection
            if len(self.archive) > 0:
                if np.random.random() < self.adaptive_params['exploration_rate']:
                    current_idx = np.random.randint(len(self.archive))
                else:
                    # Quality-biased selection
                    if len(self.archive) > 10:
                        # Pre-compute all objectives (benefits from cache)
                        archive_objectives = []
                        for sol in self.archive:
                            obj = self.evaluate_objectives(sol['chromosome'])
                            archive_objectives.append(obj)

                        # Vectorized contribution calculation
                        archive_objectives = np.array(archive_objectives)
                        contributions = -archive_objectives[:, 0] * 0.5 - archive_objectives[:, 1] * 0.3 + (15 - archive_objectives[:, 2]) * 0.2

                        probabilities = contributions - contributions.min() + 1
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

            # VNS main loop
            k = 0
            while k < self.k_max:
                if np.random.random() < 0.7:
                    k = self.select_neighborhood_adaptively()

                neighbor = self.get_neighborhood(current_solution, k)
                neighbor_objectives = self.evaluate_objectives(neighbor)

                # Pareto Local Search (optimized)
                if np.random.random() < 0.6:
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

                self.update_learning_rate(k % self.k_max, improved)
                self.tabu_list.append(tuple(neighbor))

                # Track best LU
                current_lu = -current_objectives[0]
                if current_lu > global_best_lu:
                    global_best_lu = current_lu

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

                elapsed = time.time() - start_time
                cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0

                print(f"Iter {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}{improvement_str}, "
                     f"Cache={cache_rate:.1f}%, Time={elapsed:.1f}s")

                if no_improvement >= self.min_no_improvement:
                    print(f"Converged after {iteration} iterations")
                    break

            iteration += 1

        total_time = time.time() - start_time

        print(f"\nMOVNS v19 completed:")
        print(f"  Final archive: {len(self.archive)} solutions")
        print(f"  Best hypervolume: {best_hv:.4f}")
        print(f"  Global best LU: {global_best_lu:.0f}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Cache stats: {self.cache_hits} hits, {self.cache_misses} misses")
        print(f"  Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")

        return self.format_output()

    def clear_cache(self):
        """Clear the objective cache to free memory"""
        self.objective_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOVNS v19 (speed optimized) for package: {package_name}")
    movns = MOVNS_V19(package_name, archive_size=50, max_iterations=50)
    solutions = movns.run()

    print(f"\nFinal results:")
    print(f"Archive size: {len(solutions)}")
    if solutions:
        print(f"Sample solution: {solutions[0]['packages'][:5]}")