"""
MOVNS v15 - Fixed Archive Management
Focus: Maintain 80-100 solutions like MOEA/D for fair comparison
Strategy: Controlled archive growth with diversity preservation
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import warnings
from src.optimizer.movns_v2 import MOVNS_V2

warnings.filterwarnings('ignore')


class MOVNS_V15(MOVNS_V2):
    """
    v15: Fixed archive management to maintain 80-100 solutions
    - Smart initialization to start with 50+ solutions
    - Gradual archive growth
    - Diversity-preserving archive update
    - Balanced intensification/diversification
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):
        super().__init__(main_package, archive_size, max_iterations, track_metrics)

        self.min_archive_size = 80
        self.target_archive_size = 100
        self.initial_archive_size = 50

        self.intensification_rate = 0.6
        self.diversification_rate = 0.4

        print(f"MOVNS v15 initialized - Fixed Archive Management")
        print(f"Archive: {archive_size}, Iterations: {max_iterations}")
        print(f"Target: Maintain {self.min_archive_size}-{self.target_archive_size} solutions")

    def initialize_archive_v15(self):
        """Initialize archive with 50+ diverse solutions"""
        print(f"Initializing archive with {self.initial_archive_size}+ solutions...")

        self.archive = []

        # Strategy 1: Co-occurrence based (20 solutions)
        if hasattr(self, 'cooccur_candidates'):
            for i in range(min(20, len(self.cooccur_candidates))):
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 8)
                candidates = self.cooccur_candidates[:50]
                if len(candidates) > n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    objectives = self.evaluate_objectives(solution)
                    self.update_archive(solution, objectives)

        # Strategy 2: Semantic similarity based (20 solutions)
        if hasattr(self, 'semantic_candidates'):
            for i in range(min(20, len(self.semantic_candidates))):
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 8)
                candidates = self.semantic_candidates[:50]
                if len(candidates) > n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    objectives = self.evaluate_objectives(solution)
                    self.update_archive(solution, objectives)

        # Strategy 3: Cluster-based (20 solutions)
        if hasattr(self, 'cluster_candidates'):
            for cluster_id in set(self.cluster_labels):
                solution = np.zeros(len(self.package_names), dtype=int)
                cluster_members = np.where(self.cluster_labels == cluster_id)[0]
                if len(cluster_members) > 0:
                    n_select = min(5, len(cluster_members))
                    indices = np.random.choice(cluster_members, n_select, replace=False)
                    solution[indices] = 1
                    solution[self.main_package_idx] = 1
                    objectives = self.evaluate_objectives(solution)
                    self.update_archive(solution, objectives)
                    if len(self.archive) >= 60:
                        break

        # Strategy 4: Random diverse (fill to 50+)
        while len(self.archive) < self.initial_archive_size:
            solution = np.zeros(len(self.package_names), dtype=int)
            n_select = np.random.randint(2, 10)
            indices = np.random.choice(len(solution), n_select, replace=False)
            solution[indices] = 1
            solution[self.main_package_idx] = 1
            objectives = self.evaluate_objectives(solution)
            self.update_archive(solution, objectives)

        print(f"Archive initialized with {len(self.archive)} solutions")

    def controlled_archive_update(self, solution: np.ndarray, objectives: np.ndarray):
        """Update archive with controlled growth"""
        new_solution = {
            'chromosome': solution.copy(),
            'objectives': objectives.copy()
        }

        # Check dominance
        is_dominated = False
        to_remove = []

        for i, sol in enumerate(self.archive):
            if self.dominates(sol['objectives'], objectives):
                is_dominated = True
                break
            elif self.dominates(objectives, sol['objectives']):
                to_remove.append(i)

        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(to_remove):
                self.archive.pop(i)

            # Add new solution
            self.archive.append(new_solution)

            # Only trim if significantly over limit
            if len(self.archive) > self.target_archive_size + 20:
                # Use crowding distance to select diverse subset
                self.trim_archive_with_crowding()

    def trim_archive_with_crowding(self):
        """Trim archive to target size using crowding distance"""
        if len(self.archive) <= self.target_archive_size:
            return

        # Calculate crowding distance
        for sol in self.archive:
            sol['crowding_distance'] = 0

        n_obj = 3
        for m in range(n_obj):
            sorted_archive = sorted(self.archive, key=lambda x: x['objectives'][m])

            sorted_archive[0]['crowding_distance'] = float('inf')
            sorted_archive[-1]['crowding_distance'] = float('inf')

            obj_range = sorted_archive[-1]['objectives'][m] - sorted_archive[0]['objectives'][m]
            if obj_range > 0:
                for i in range(1, len(sorted_archive) - 1):
                    distance = (sorted_archive[i+1]['objectives'][m] -
                              sorted_archive[i-1]['objectives'][m]) / obj_range
                    sorted_archive[i]['crowding_distance'] += distance

        # Keep most diverse solutions
        self.archive.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)
        self.archive = self.archive[:self.target_archive_size]

    def generate_diverse_solution(self) -> np.ndarray:
        """Generate diverse solution to maintain archive size"""
        solution = np.zeros(len(self.package_names), dtype=int)

        # Use different strategies
        strategy = np.random.choice(['cooccur', 'semantic', 'cluster', 'random'])

        if strategy == 'cooccur' and hasattr(self, 'cooccur_candidates'):
            n_select = np.random.randint(4, 8)
            candidates = self.cooccur_candidates[:100]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1

        elif strategy == 'semantic' and hasattr(self, 'semantic_candidates'):
            n_select = np.random.randint(4, 8)
            candidates = self.semantic_candidates[:100]
            if len(candidates) >= n_select:
                indices = np.random.choice(candidates, n_select, replace=False)
                solution[indices] = 1

        elif strategy == 'cluster':
            cluster_id = np.random.choice(list(set(self.cluster_labels)))
            cluster_members = np.where(self.cluster_labels == cluster_id)[0]
            if len(cluster_members) > 0:
                n_select = min(6, len(cluster_members))
                indices = np.random.choice(cluster_members, n_select, replace=False)
                solution[indices] = 1

        else:  # random
            n_select = np.random.randint(3, 10)
            indices = np.random.choice(len(solution), n_select, replace=False)
            solution[indices] = 1

        solution[self.main_package_idx] = 1
        return solution

    def vns_local_search_v15(self, solution: np.ndarray, k: int) -> np.ndarray:
        """VNS local search with balanced intensification/diversification"""

        if np.random.random() < self.intensification_rate:
            # Intensification neighborhoods
            if k == 0:
                neighbor = self.n1_single_flip(solution)
            elif k == 1:
                neighbor = self.n2_multi_flip(solution)
            else:
                neighbor = self.n3_segment_exchange(solution)
        else:
            # Diversification neighborhoods
            neighbor = self.generate_diverse_neighbor(solution)

        return neighbor

    def generate_diverse_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate diverse neighbor for exploration"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(active) > 3 and len(inactive) > 0:
            # Large change for diversity
            n_changes = min(3, len(active) // 2)

            # Remove random active
            remove_idx = np.random.choice(active, n_changes, replace=False)
            for idx in remove_idx:
                neighbor[idx] = 0

            # Add diverse inactive
            if len(inactive) >= n_changes:
                add_idx = np.random.choice(inactive, n_changes, replace=False)
                for idx in add_idx:
                    neighbor[idx] = 1

        return neighbor

    def run(self) -> List[Dict]:
        """Run MOVNS v15 with fixed archive management"""
        print(f"\nStarting MOVNS v15 for {self.main_package}...")
        print("="*60)

        # Initialize with 50+ solutions
        self.initialize_archive_v15()

        best_hv = 0
        no_improvement = 0

        for iteration in range(self.max_iterations):
            # Ensure minimum archive size
            while len(self.archive) < self.min_archive_size:
                new_solution = self.generate_diverse_solution()
                objectives = self.evaluate_objectives(new_solution)
                self.controlled_archive_update(new_solution, objectives)

            # Select parent from archive
            parent_idx = np.random.randint(len(self.archive))
            parent = self.archive[parent_idx]['chromosome'].copy()

            # VNS main loop
            k = 0
            k_max = 4

            while k < k_max:
                # Generate neighbor
                neighbor = self.vns_local_search_v15(parent, k)

                if not np.array_equal(neighbor, parent):
                    neighbor_obj = self.evaluate_objectives(neighbor)
                    parent_obj = self.evaluate_objectives(parent)

                    # Update archive
                    self.controlled_archive_update(neighbor, neighbor_obj)

                    # Check improvement
                    if self.dominates(neighbor_obj, parent_obj):
                        parent = neighbor
                        k = 0  # Reset to first neighborhood
                    else:
                        k += 1
                else:
                    k += 1

            # Additional diversification every few iterations
            if iteration % 3 == 0:
                for _ in range(2):
                    diverse_solution = self.generate_diverse_solution()
                    objectives = self.evaluate_objectives(diverse_solution)
                    self.controlled_archive_update(diverse_solution, objectives)

            # Track metrics
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics and 'hypervolume' in metrics:
                    current_hv = metrics['hypervolume']

                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    if iteration % 10 == 0:
                        print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                              f"HV={current_hv:.4f}, Best={best_hv:.4f}")

            # Early stopping
            if no_improvement >= 15:
                print(f"Converged at iteration {iteration}")
                break

        # Final trim to target size
        if len(self.archive) > self.target_archive_size:
            self.trim_archive_with_crowding()

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_v15.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_V15(
        package,
        archive_size=100,
        max_iterations=30,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    if solutions:
        best_lu = min(solutions, key=lambda x: x['objectives'][0])
        best_ss = min(solutions, key=lambda x: x['objectives'][1])
        best_rss = min(solutions, key=lambda x: x['objectives'][2])

        print(f"Best LU: {-best_lu['objectives'][0]:.0f}")
        print(f"Best SS: {-best_ss['objectives'][1]:.4f}")
        print(f"Best RSS: {best_rss['objectives'][2]:.1f}")


if __name__ == "__main__":
    main()