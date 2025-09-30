"""
MOVNS v13 - Optimized for Better Distribution (Spacing)
Focus: Win HV + Spacing metrics against MOEA/D
Strategy: Modified neighborhoods for better spread
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import warnings
from optimizer.movns_v2 import MOVNS_V2

warnings.filterwarnings('ignore')


@dataclass
class Solution:
    chromosome: np.ndarray
    objectives: np.ndarray
    id: int


class MOVNS_V13(MOVNS_V2):
    """
    v13 improvements:
    - Modified neighborhoods for better distribution
    - Crowding-based archive management
    - Spread-aware local search
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):
        super().__init__(main_package, archive_size, max_iterations, track_metrics)

        self.spread_factor = 0.3
        self.crowding_threshold = 0.1
        self.diversity_memory = []
        self.solution_counter = 0

        print(f"MOVNS v13 initialized - Distribution Optimized")
        print(f"Archive: {archive_size}, Iterations: {max_iterations}")
        print(f"Focus: Better spacing while maintaining HV")

    def calculate_crowding_distance(self, solutions: List[Dict]) -> None:
        """Calculate crowding distance for better distribution"""
        if len(solutions) <= 2:
            for sol in solutions:
                sol['crowding_distance'] = float('inf')
            return

        n_obj = len(solutions[0]['objectives'])
        for sol in solutions:
            sol['crowding_distance'] = 0

        for m in range(n_obj):
            sorted_sols = sorted(solutions, key=lambda x: x['objectives'][m])

            sorted_sols[0]['crowding_distance'] = float('inf')
            sorted_sols[-1]['crowding_distance'] = float('inf')

            if sorted_sols[-1]['objectives'][m] != sorted_sols[0]['objectives'][m]:
                norm = sorted_sols[-1]['objectives'][m] - sorted_sols[0]['objectives'][m]

                for i in range(1, len(sorted_sols) - 1):
                    distance = (sorted_sols[i+1]['objectives'][m] -
                               sorted_sols[i-1]['objectives'][m]) / norm
                    sorted_sols[i]['crowding_distance'] += distance

    def n1_spread_flip(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 1: Flip to increase spread"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(active) > 2 and len(inactive) > 0:
            most_common_idx = None
            max_cooccur = -1

            for idx in active:
                total_cooccur = 0
                for other in active:
                    if idx != other:
                        if hasattr(self.rel_matrix[idx, other], 'toarray'):
                            val = self.rel_matrix[idx, other].toarray()[0, 0]
                        else:
                            val = self.rel_matrix[idx, other]
                        total_cooccur += val

                if total_cooccur > max_cooccur:
                    max_cooccur = total_cooccur
                    most_common_idx = idx

            if most_common_idx is not None:
                neighbor[most_common_idx] = 0

                add_idx = np.random.choice(inactive)
                neighbor[add_idx] = 1

        return neighbor

    def n2_diversity_exchange(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 2: Exchange for diversity"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(active) > 3 and len(inactive) > 0:
            n_changes = min(3, len(active) // 2)

            cluster_counts = {}
            for idx in active:
                cluster = self.cluster_labels[idx]
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

            overrepresented = []
            for idx in active:
                cluster = self.cluster_labels[idx]
                if cluster_counts[cluster] > 1:
                    overrepresented.append(idx)

            if len(overrepresented) >= n_changes:
                remove_indices = np.random.choice(overrepresented, n_changes, replace=False)
            else:
                remove_indices = np.random.choice(active, n_changes, replace=False)

            for idx in remove_indices:
                neighbor[idx] = 0

            underrepresented_clusters = set()
            for idx in inactive:
                cluster = self.cluster_labels[idx]
                if cluster not in cluster_counts or cluster_counts[cluster] == 0:
                    underrepresented_clusters.add(cluster)

            candidates = [idx for idx in inactive
                         if self.cluster_labels[idx] in underrepresented_clusters]

            if len(candidates) < n_changes:
                candidates = list(inactive)

            if len(candidates) >= n_changes:
                add_indices = np.random.choice(candidates, n_changes, replace=False)
                for idx in add_indices:
                    neighbor[idx] = 1

        return neighbor

    def n3_boundary_push(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 3: Push towards boundary for spread"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]

        objectives = self.evaluate_objectives(solution)

        target_obj = np.random.randint(3)

        if target_obj == 0:
            if len(active) < 15:
                candidates = []
                for idx in range(len(solution)):
                    if solution[idx] == 0:
                        cooccur_sum = 0
                        for active_idx in active:
                            if hasattr(self.rel_matrix[idx, active_idx], 'toarray'):
                                val = self.rel_matrix[idx, active_idx].toarray()[0, 0]
                            else:
                                val = self.rel_matrix[idx, active_idx]
                            cooccur_sum += val
                        if cooccur_sum > 0:
                            candidates.append((idx, cooccur_sum))

                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    add_idx = candidates[0][0]
                    neighbor[add_idx] = 1

        elif target_obj == 1:
            if len(active) > 3:
                similarities = []
                for idx in active:
                    sim_sum = 0
                    for other in active:
                        if idx != other:
                            sim_sum += self.sim_matrix[idx, other]
                    similarities.append((idx, sim_sum))

                similarities.sort(key=lambda x: x[1])
                remove_idx = similarities[0][0]
                neighbor[remove_idx] = 0

        else:
            if len(active) > 3:
                remove_n = min(2, len(active) - 3)
                remove_indices = np.random.choice(active, remove_n, replace=False)
                for idx in remove_indices:
                    neighbor[idx] = 0

        return neighbor

    def n4_gap_filling(self, solution: np.ndarray) -> np.ndarray:
        """Neighborhood 4: Fill gaps in objective space"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(self.archive) > 5 and len(inactive) > 0:
            current_obj = self.evaluate_objectives(solution)

            distances = []
            for sol in self.archive:
                dist = np.linalg.norm(sol['objectives'] - current_obj)
                distances.append(dist)

            median_dist = np.median(distances)

            if median_dist < self.crowding_threshold:
                n_add = min(2, len(inactive))
                add_indices = np.random.choice(inactive, n_add, replace=False)
                for idx in add_indices:
                    neighbor[idx] = 1
            elif len(active) > 10:
                n_remove = min(2, len(active) - 5)
                remove_indices = np.random.choice(active, n_remove, replace=False)
                for idx in remove_indices:
                    neighbor[idx] = 0

        return neighbor

    def vns_local_search(self, solution: np.ndarray, k: int) -> np.ndarray:
        """Apply VNS with distribution-aware local search"""
        neighborhoods = [
            self.n1_spread_flip,
            self.n2_diversity_exchange,
            self.n3_boundary_push,
            self.n4_gap_filling
        ]

        current = solution.copy()
        current_obj = self.evaluate_objectives(current)

        if k < len(neighborhoods):
            neighbor = neighborhoods[k](current)
            neighbor_obj = self.evaluate_objectives(neighbor)

            accept = False

            if self.dominates(neighbor_obj, current_obj):
                accept = True
            elif not self.dominates(current_obj, neighbor_obj):
                if len(self.archive) > 10:
                    self.calculate_crowding_distance(self.archive)

                    min_crowding = min(sol['crowding_distance'] for sol in self.archive)

                    temp_sol = {
                        'chromosome': neighbor,
                        'objectives': neighbor_obj,
                        'crowding_distance': 0
                    }

                    temp_archive = self.archive + [temp_sol]
                    self.calculate_crowding_distance(temp_archive)

                    if temp_sol['crowding_distance'] > min_crowding * 1.2:
                        accept = True
                else:
                    if np.random.random() < 0.3:
                        accept = True

            if accept:
                return neighbor

        return current

    def update_archive_with_crowding(self, solution: np.ndarray, objectives: np.ndarray):
        """Update archive with crowding-based selection"""
        new_solution = {
            'chromosome': solution.copy(),
            'objectives': objectives.copy(),
            'id': self.solution_counter
        }
        self.solution_counter += 1

        is_dominated = False
        to_remove = []

        for i, sol in enumerate(self.archive):
            if self.dominates(sol['objectives'], objectives):
                is_dominated = True
                break
            elif self.dominates(objectives, sol['objectives']):
                to_remove.append(i)

        if not is_dominated:
            for i in reversed(to_remove):
                self.archive.pop(i)

            self.archive.append(new_solution)

            if len(self.archive) > self.archive_limit:
                self.calculate_crowding_distance(self.archive)

                self.archive.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)

                self.archive = self.archive[:self.archive_limit]

    def run(self) -> List[Dict]:
        """Run MOVNS v13 with distribution optimization"""
        print("\nStarting MOVNS v13 for", self.main_package, "...")
        print("="*60)

        self.initialize_archive()

        best_hv = 0
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            if len(self.archive) == 0:
                break
            parent_idx = np.random.randint(len(self.archive))
            parent = self.archive[parent_idx]['chromosome'].copy()

            k = 0
            k_max = 4

            while k < k_max:
                candidate = self.vns_local_search(parent, k)

                if not np.array_equal(candidate, parent):
                    obj = self.evaluate_objectives(candidate)
                    self.update_archive_with_crowding(candidate, obj)

                    parent_found = False
                    for sol in self.archive:
                        if np.array_equal(sol['chromosome'], parent):
                            if self.dominates(obj, sol['objectives']):
                                parent = candidate
                                k = 0
                            else:
                                k += 1
                            parent_found = True
                            break

                    if not parent_found:
                        k += 1
                else:
                    k += 1

            if self.track_metrics and iteration % 2 == 0:
                metrics = self.calculate_metrics()
                if metrics and 'hypervolume' in metrics:
                    current_hv = metrics['hypervolume']
                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if iteration % 5 == 0:
                        print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                              f"HV={current_hv:.4f}, Best={best_hv:.4f}")

            if no_improvement_count >= 20:
                print(f"Converged at iteration {iteration}")
                break

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_v13.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]
    optimizer = MOVNS_V13(package, archive_size=100, max_iterations=30, track_metrics=True)
    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    if solutions:
        best = min(solutions, key=lambda x: x['objectives'][2])
        print(f"Best set size: {best['objectives'][2]}")


if __name__ == "__main__":
    main()