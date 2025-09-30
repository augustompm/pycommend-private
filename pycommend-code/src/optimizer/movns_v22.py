"""
MOVNS v22 - Balanced optimization
Real calculations but with smart optimizations
Target: <30s for 50 iterations while beating MOEA/D
"""

import numpy as np
import pickle
import random
import time
from typing import List, Dict, Tuple
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics
from optimizer.movns_v2 import MOVNS_V2


class MOVNS_V22(MOVNS_V2):
    """
    Balanced MOVNS - Fast but maintains quality
    Real calculations with smart caching and reduced complexity
    """

    def __init__(self, main_package, archive_size=50, max_iterations=50,
                 track_metrics=True):
        # Initialize cache before super
        self.objective_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Call parent with optimized settings
        super().__init__(main_package, archive_size, max_iterations,
                        k_max=3, track_metrics=track_metrics, min_no_improvement=10)

        # Optimized parameters
        self.temperature = 1.0
        self.cooling_rate = 0.995

        # Small tabu list
        self.tabu_list = deque(maxlen=30)

        # PLS settings - calibrated to beat MOEA/D
        self.pls_probability = 0.5  # 50% chance (increased from 40%)
        self.pls_max_neighbors = 8  # 8 neighbors (increased from 5)
        self.pls_iterations = 3  # Quick local search

        print(f"MOVNS v22 initialized - Balanced optimization")
        print(f"Archive: {archive_size}, Iterations: {max_iterations}")

    def chromosome_to_key(self, chromosome):
        """Fast key generation"""
        return tuple(np.where(chromosome == 1)[0])

    def evaluate_objectives(self, chromosome):
        """Cached real evaluation"""
        key = self.chromosome_to_key(chromosome)

        # Check cache
        if key in self.objective_cache:
            self.cache_hits += 1
            return self.objective_cache[key].copy()

        self.cache_misses += 1
        indices = np.array(key)

        # Real calculations but optimized
        lu_score = self.calculate_linked_usage_fast(indices)
        ss_score = self.calculate_semantic_similarity_fast(indices)
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        # Cache if space available
        if len(self.objective_cache) < 2000:
            self.objective_cache[key] = objectives.copy()

        return objectives

    def calculate_linked_usage_fast(self, indices):
        """Optimized linked usage - real calculation"""
        if len(indices) == 0:
            return 0

        # Use slicing for sparse matrix
        if hasattr(self.rel_matrix, 'toarray'):
            # For sparse matrix, only extract needed submatrix
            row_data = self.rel_matrix[indices]
            if hasattr(row_data, 'tocsr'):
                row_data = row_data.tocsr()
            submatrix = row_data[:, indices]
            if hasattr(submatrix, 'toarray'):
                submatrix = submatrix.toarray()
        else:
            # Dense matrix
            submatrix = self.rel_matrix[np.ix_(indices, indices)]

        # Fast sum
        total = np.sum(submatrix)
        diagonal = np.sum(np.diag(submatrix))
        return total - diagonal

    def calculate_semantic_similarity_fast(self, indices):
        """Optimized semantic similarity - real calculation"""
        if len(indices) <= 1:
            return 0

        # Get embeddings
        embeddings_subset = self.embeddings[indices]

        # Fast centroid
        centroid = np.mean(embeddings_subset, axis=0)

        # Vectorized cosine similarity
        dots = embeddings_subset @ centroid
        norms_emb = np.linalg.norm(embeddings_subset, axis=1)
        norm_cent = np.linalg.norm(centroid)

        similarities = dots / (norms_emb * norm_cent + 1e-10)
        return np.mean(similarities)

    def get_neighborhood(self, solution: np.ndarray, k: int) -> np.ndarray:
        """Simple but effective neighborhoods"""
        neighbor = solution.copy()
        indices = np.where(solution == 1)[0]

        if k == 0:
            # Single flip - add or remove
            if np.random.random() < 0.7 and len(indices) < self.max_size:
                # Add from top candidates (increased probability)
                candidates = self.cooccur_candidates[:70]
                valid = [c for c in candidates if solution[c] == 0]
                if valid:
                    neighbor[random.choice(valid)] = 1
            elif len(indices) > self.min_size:
                # Remove random
                neighbor[random.choice(indices)] = 0

        elif k == 1:
            # Double flip
            for _ in range(2):
                current_indices = np.where(neighbor == 1)[0]
                if np.random.random() < 0.5 and len(current_indices) < self.max_size:
                    candidates = self.semantic_candidates[:50]
                    valid = [c for c in candidates if neighbor[c] == 0]
                    if valid:
                        neighbor[random.choice(valid)] = 1
                elif len(current_indices) > self.min_size:
                    neighbor[random.choice(current_indices)] = 0

        else:  # k == 2
            # Smart swap
            if len(indices) >= 3:
                # Remove weakest, add strongest
                to_remove = random.choice(indices)
                neighbor[to_remove] = 0

                # Add from combined pool
                all_candidates = np.unique(np.concatenate([
                    self.cooccur_candidates[:30],
                    self.semantic_candidates[:30]
                ]))
                valid = [c for c in all_candidates if neighbor[c] == 0]
                if valid:
                    neighbor[random.choice(valid)] = 1

        return neighbor

    def simple_local_search(self, solution: np.ndarray, max_neighbors: int = 5):
        """Simple but effective local search"""
        best = solution.copy()
        best_obj = self.evaluate_objectives(best)

        improvements = 0
        for _ in range(max_neighbors):
            # Try a random neighborhood
            k = random.randint(0, self.k_max - 1)
            neighbor = self.get_neighborhood(best, k)

            # Skip if in tabu
            if tuple(neighbor) in self.tabu_list:
                continue

            neighbor_obj = self.evaluate_objectives(neighbor)

            # Accept if better
            if self.dominates(neighbor_obj, best_obj):
                best = neighbor
                best_obj = neighbor_obj
                improvements += 1

        return best, improvements > 0

    def run(self) -> List[Dict]:
        """Optimized main loop"""
        print(f"\nStarting MOVNS v22...")
        start_time = time.time()

        iteration = 0
        no_improvement = 0
        best_hv = 0

        while iteration < self.max_iterations:
            # Select from archive
            if len(self.archive) > 0:
                # Random selection (fast)
                current_idx = random.randint(0, len(self.archive) - 1)
                current = self.archive[current_idx]['chromosome'].copy()
                current_obj = self.archive[current_idx]['objectives'].copy()
            else:
                # Initialize
                current = self.smart_initialization(exploration_rate=0.3)[0]
                current_obj = self.evaluate_objectives(current)
                self.update_archive(current, current_obj)

            # VNS main loop
            k = 0
            local_no_improve = 0

            while k < self.k_max and local_no_improve < 3:
                # Shaking
                neighbor = self.get_neighborhood(current, k)
                neighbor_obj = self.evaluate_objectives(neighbor)

                # Local search with probability
                if random.random() < self.pls_probability:
                    improved_neighbor, improved = self.simple_local_search(neighbor, self.pls_max_neighbors)
                    if improved:
                        neighbor = improved_neighbor
                        neighbor_obj = self.evaluate_objectives(neighbor)

                # Update archive
                self.update_archive(neighbor, neighbor_obj)

                # Accept or reject
                if self.dominates(neighbor_obj, current_obj):
                    current = neighbor
                    current_obj = neighbor_obj
                    k = 0
                    local_no_improve = 0
                elif not self.dominates(current_obj, neighbor_obj):
                    # Simulated annealing acceptance
                    delta = np.sum(np.abs(neighbor_obj - current_obj))
                    if random.random() < np.exp(-delta / self.temperature):
                        current = neighbor
                        current_obj = neighbor_obj
                        k = 0
                    else:
                        k += 1
                        local_no_improve += 1
                else:
                    k += 1
                    local_no_improve += 1

                # Update tabu
                self.tabu_list.append(tuple(neighbor))

            # Cool temperature
            self.temperature *= self.cooling_rate

            # Progress tracking
            if iteration % 5 == 0:
                # Calculate HV if tracking
                if self.track_metrics and len(self.archive) > 0:
                    objectives = np.array([sol['objectives'] for sol in self.archive])
                    normalized = np.array([self.normalize_objectives(obj) for obj in objectives])
                    current_hv = self.metrics_calculator.hypervolume(normalized)

                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1
                else:
                    current_hv = 0

                elapsed = time.time() - start_time
                cache_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0

                print(f"Iter {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}, Cache={cache_rate:.1f}%, "
                     f"Time={elapsed:.1f}s")

                # Early stopping
                if no_improvement >= self.min_no_improvement:
                    print(f"Converged after {iteration} iterations")
                    break

            iteration += 1

        total_time = time.time() - start_time

        print(f"\nMOVNS v22 completed:")
        print(f"  Archive: {len(self.archive)} solutions")
        print(f"  Best HV: {best_hv:.4f}")
        print(f"  Time: {total_time:.1f}s")
        print(f"  Cache hit rate: {self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}%")

        # Return formatted output
        results = []
        for sol_dict in self.archive:
            indices = np.where(sol_dict['chromosome'] == 1)[0]
            packages = [self.package_names[i] for i in indices]
            results.append({
                'chromosome': sol_dict['chromosome'],
                'objectives': sol_dict['objectives'],
                'packages': packages
            })
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    print(f"Testing MOVNS v22 for: {package_name}")

    # Quick test with 20 iterations
    movns = MOVNS_V22(package_name, archive_size=50, max_iterations=20)
    solutions = movns.run()

    print(f"\nFinal: {len(solutions)} solutions")
    if solutions:
        print(f"Sample: {solutions[0]['packages'][:5]}")