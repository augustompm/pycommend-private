"""
MOVNS v14 - Balanced HV and Spacing
Strategy: Maintain high HV (VNS strength) while moderately improving spacing
Focus: Quality first, distribution second
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import warnings
from optimizer.movns_advanced import MOVNS_Advanced

warnings.filterwarnings('ignore')


class MOVNS_V14(MOVNS_Advanced):
    """
    v14: Sensible balance between HV and Spacing
    - Keep aggressive local search for HV
    - Add mild diversity mechanisms
    - Maintain large archive (100)
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):
        super().__init__(main_package, archive_size, max_iterations, track_metrics)

        self.diversity_rate = 0.2  # 20% chance for diversity moves
        self.intensification_rate = 0.8  # 80% focus on quality

        print(f"MOVNS v14 initialized - Balanced HV/Spacing")
        print(f"Archive: {archive_size}, Iterations: {max_iterations}")
        print(f"Strategy: 80% intensification, 20% diversification")

    def adaptive_local_search(self, solution: np.ndarray) -> np.ndarray:
        """Balanced local search with quality focus"""

        if np.random.random() < self.intensification_rate:
            # Quality-focused search (80% of time)
            return self.quality_focused_search(solution)
        else:
            # Diversity-focused search (20% of time)
            return self.diversity_focused_search(solution)

    def quality_focused_search(self, solution: np.ndarray) -> np.ndarray:
        """Aggressive search for high-quality solutions (HV)"""
        best_solution = solution.copy()
        best_obj = self.evaluate_objectives(best_solution)

        # Try aggressive improvements
        for _ in range(5):
            neighbor = self.aggressive_neighbor(solution)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if self.dominates(neighbor_obj, best_obj):
                best_solution = neighbor.copy()
                best_obj = neighbor_obj

        return best_solution

    def aggressive_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate neighbor focused on objective improvement"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(inactive) > 0 and len(active) < 15:
            # Add high-value packages
            scores = []
            for idx in inactive:
                # Combined score for quality
                cooccur_score = 0
                if hasattr(self.rel_matrix[self.main_package_idx, idx], 'toarray'):
                    cooccur_score = self.rel_matrix[self.main_package_idx, idx].toarray()[0, 0]
                else:
                    cooccur_score = self.rel_matrix[self.main_package_idx, idx]

                sim_score = self.sim_matrix[self.main_package_idx, idx]
                combined = cooccur_score + 0.5 * sim_score
                scores.append((idx, combined))

            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                # Add top candidates
                n_add = min(2, len(scores))
                for i in range(n_add):
                    neighbor[scores[i][0]] = 1

        elif len(active) > 8:
            # Remove weak packages
            scores = []
            for idx in active:
                cooccur_score = 0
                if hasattr(self.rel_matrix[self.main_package_idx, idx], 'toarray'):
                    cooccur_score = self.rel_matrix[self.main_package_idx, idx].toarray()[0, 0]
                else:
                    cooccur_score = self.rel_matrix[self.main_package_idx, idx]

                scores.append((idx, cooccur_score))

            scores.sort(key=lambda x: x[1])
            # Remove weakest
            if scores:
                neighbor[scores[0][0]] = 0

        return neighbor

    def diversity_focused_search(self, solution: np.ndarray) -> np.ndarray:
        """Mild diversity enhancement without sacrificing quality"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if len(active) > 3 and len(inactive) > 0:
            # Smart swap for diversity
            # Find packages from different clusters
            active_clusters = set(self.cluster_labels[idx] for idx in active)

            diverse_candidates = []
            for idx in inactive:
                if self.cluster_labels[idx] not in active_clusters:
                    diverse_candidates.append(idx)

            if diverse_candidates and len(active) > 5:
                # Swap one element
                add_idx = np.random.choice(diverse_candidates)
                remove_idx = np.random.choice(active)

                neighbor[add_idx] = 1
                neighbor[remove_idx] = 0

        return neighbor

    def run(self) -> List[Dict]:
        """Run MOVNS v14 with balanced approach"""
        print("\nStarting MOVNS v14 for", self.main_package, "...")
        print("="*60)

        # Initialize with good archive
        self.initialize_archive()

        # Ensure we start with enough solutions
        while len(self.archive) < 50:
            # Generate diverse initial solutions
            solution = np.zeros(len(self.package_names), dtype=int)
            n_select = np.random.randint(3, 8)

            # Mix strategies for diversity
            if np.random.random() < 0.5 and hasattr(self, 'cooccur_candidates'):
                # Co-occurrence based
                indices = np.random.choice(self.cooccur_candidates[:100],
                                         min(n_select, len(self.cooccur_candidates)),
                                         replace=False)
            else:
                # Random selection
                indices = np.random.choice(len(solution), n_select, replace=False)

            solution[indices] = 1
            solution[self.main_package_idx] = 1

            objectives = self.evaluate_objectives(solution)
            self.update_archive(solution, objectives)

        best_hv = 0
        no_improvement = 0

        for iteration in range(self.max_iterations):
            # Select parent from archive
            if len(self.archive) == 0:
                break

            parent_idx = np.random.randint(len(self.archive))
            parent = self.archive[parent_idx]['chromosome'].copy()

            # Apply adaptive local search
            improved = self.adaptive_local_search(parent)

            if not np.array_equal(improved, parent):
                obj = self.evaluate_objectives(improved)
                self.update_archive(improved, obj)

            # Additional search from random archive member
            if iteration % 3 == 0 and len(self.archive) > 10:
                random_idx = np.random.randint(len(self.archive))
                random_parent = self.archive[random_idx]['chromosome'].copy()

                # Apply different search strategy
                if np.random.random() < 0.3:
                    candidate = self.diversity_focused_search(random_parent)
                else:
                    candidate = self.quality_focused_search(random_parent)

                if not np.array_equal(candidate, random_parent):
                    obj = self.evaluate_objectives(candidate)
                    self.update_archive(candidate, obj)

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

            # Early stopping if converged
            if no_improvement >= 15:
                print(f"Converged at iteration {iteration}")
                break

        # Final refinement pass
        print("\nFinal refinement pass...")
        for _ in range(5):
            if len(self.archive) > 0:
                idx = np.random.randint(len(self.archive))
                solution = self.archive[idx]['chromosome'].copy()
                refined = self.quality_focused_search(solution)
                if not np.array_equal(refined, solution):
                    obj = self.evaluate_objectives(refined)
                    self.update_archive(refined, obj)

        print(f"\nFinal archive: {len(self.archive)} solutions")

        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                print(f"Final HV: {final_metrics.get('hypervolume', 0):.4f}")

        return self.archive


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_v14.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_V14(
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