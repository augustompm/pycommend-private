"""
MOVNS v15 Fast - Efficient Archive Management
Focus: Maintain 80-100 solutions efficiently for fair comparison
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from optimizer.movns_v2 import MOVNS_V2


class MOVNS_V15_Fast(MOVNS_V2):
    """
    v15 Fast: Efficient archive management
    - Fast initialization with basic strategies
    - Maintain 80-100 solutions consistently
    - Simple but effective neighborhoods
    """

    def __init__(self, main_package: str, archive_size: int = 100,
                 max_iterations: int = 100, track_metrics: bool = False):
        self.min_archive = 80
        self.max_archive = 100

        super().__init__(main_package, archive_size, max_iterations,
                         track_metrics=track_metrics)

        print(f"MOVNS v15 Fast initialized")
        print(f"Target archive: {self.min_archive}-{self.max_archive} solutions")
        print(f"Track metrics: {self.track_metrics}")

    def initialize_archive(self):
        """Fast initialization to 80+ solutions"""
        self.archive = []

        # Quick generation of 80 diverse solutions
        for i in range(self.min_archive):
            solution = np.zeros(len(self.package_names), dtype=int)

            # Vary strategy
            if i < 30 and hasattr(self, 'cooccur_candidates'):
                # Co-occurrence based
                n_select = np.random.randint(3, 7)
                candidates = self.cooccur_candidates[:50]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
            elif i < 60 and hasattr(self, 'semantic_candidates'):
                # Semantic based
                n_select = np.random.randint(3, 7)
                candidates = self.semantic_candidates[:50]
                if len(candidates) >= n_select:
                    indices = np.random.choice(candidates, n_select, replace=False)
                    solution[indices] = 1
            else:
                # Random
                n_select = np.random.randint(2, 8)
                indices = np.random.choice(len(solution), n_select, replace=False)
                solution[indices] = 1

            solution[self.main_package_idx] = 1
            objectives = self.evaluate_objectives(solution)

            # Simple add without dominance check for speed
            self.archive.append({
                'chromosome': solution.copy(),
                'objectives': objectives.copy()
            })

        # Remove dominated solutions
        self.filter_dominated()

        print(f"Archive initialized with {len(self.archive)} solutions")

    def filter_dominated(self):
        """Remove dominated solutions from archive"""
        non_dominated = []

        for i, sol_i in enumerate(self.archive):
            is_dominated = False
            for j, sol_j in enumerate(self.archive):
                if i != j and self.dominates(sol_j['objectives'], sol_i['objectives']):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(sol_i)

        self.archive = non_dominated

    def simple_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate simple neighbor"""
        neighbor = solution.copy()
        active = np.where(solution == 1)[0]
        inactive = np.where(solution == 0)[0]

        if np.random.random() < 0.5 and len(inactive) > 0:
            # Add
            idx = np.random.choice(inactive)
            neighbor[idx] = 1
        elif len(active) > 2:
            # Remove
            idx = np.random.choice(active)
            if idx != self.main_package_idx:
                neighbor[idx] = 0

        return neighbor

    def run(self) -> List[Dict]:
        """Run MOVNS v15 Fast"""
        print(f"\nStarting MOVNS v15 Fast for {self.main_package}...")

        # Initialize metrics tracking
        if self.track_metrics:
            self.metrics_history = {'hypervolume': [], 'spacing': [], 'diversity': []}
            print(f"Metrics tracking enabled")

        # Fast initialization
        self.initialize_archive()

        for iteration in range(self.max_iterations):
            # Maintain minimum archive size
            while len(self.archive) < self.min_archive:
                solution = np.zeros(len(self.package_names), dtype=int)
                n_select = np.random.randint(3, 7)
                indices = np.random.choice(len(solution), n_select, replace=False)
                solution[indices] = 1
                solution[self.main_package_idx] = 1
                objectives = self.evaluate_objectives(solution)
                self.archive.append({
                    'chromosome': solution.copy(),
                    'objectives': objectives.copy()
                })

            # Select and improve
            parent_idx = np.random.randint(len(self.archive))
            parent = self.archive[parent_idx]['chromosome'].copy()

            # Simple VNS
            for _ in range(3):
                neighbor = self.simple_neighbor(parent)
                neighbor_obj = self.evaluate_objectives(neighbor)

                # Add if non-dominated
                is_dominated = False
                for sol in self.archive:
                    if self.dominates(sol['objectives'], neighbor_obj):
                        is_dominated = True
                        break

                if not is_dominated:
                    self.archive.append({
                        'chromosome': neighbor.copy(),
                        'objectives': neighbor_obj.copy()
                    })

                    # Trim if too large
                    if len(self.archive) > self.max_archive + 20:
                        self.filter_dominated()
                        if len(self.archive) > self.max_archive:
                            # Random trim
                            self.archive = self.archive[:self.max_archive]

            # Track metrics periodically
            metrics = None
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    self.metrics_history['hypervolume'].append(metrics.get('hypervolume', 0))
                    self.metrics_history['spacing'].append(metrics.get('spacing', 0))
                    self.metrics_history['diversity'].append(metrics.get('diversity', 0))

            if iteration % 10 == 0:
                if self.track_metrics and metrics:
                    print(f"Iteration {iteration}: Archive={len(self.archive)}, HV={metrics.get('hypervolume', 0):.4f}")
                else:
                    print(f"Iteration {iteration}: Archive={len(self.archive)}")

        # Final filter only if much larger
        if len(self.archive) > self.max_archive * 1.5:
            self.filter_dominated()

        # Ensure size limits
        if len(self.archive) > self.max_archive:
            # Keep diverse subset
            indices = np.random.choice(len(self.archive), self.max_archive, replace=False)
            self.archive = [self.archive[i] for i in indices]

        # Final metrics calculation
        if self.track_metrics:
            final_metrics = self.calculate_metrics()
            if final_metrics:
                self.metrics_history['hypervolume'].append(final_metrics.get('hypervolume', 0))
                self.metrics_history['spacing'].append(final_metrics.get('spacing', 0))
                self.metrics_history['diversity'].append(final_metrics.get('diversity', 0))
                print(f"Final archive: {len(self.archive)} solutions, HV={final_metrics.get('hypervolume', 0):.4f}")
            else:
                print(f"Final archive: {len(self.archive)} solutions")
        else:
            print(f"Final archive: {len(self.archive)} solutions")

        return self.archive

    def get_metrics_history(self):
        """Return metrics history"""
        if hasattr(self, 'metrics_history'):
            return self.metrics_history
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python movns_v15_fast.py <package_name>")
        sys.exit(1)

    package = sys.argv[1]

    optimizer = MOVNS_V15_Fast(
        package,
        archive_size=100,
        max_iterations=20,
        track_metrics=True
    )

    solutions = optimizer.run()

    print(f"\nFound {len(solutions)} solutions")
    if 80 <= len(solutions) <= 100:
        print("Archive size target achieved!")


if __name__ == "__main__":
    main()