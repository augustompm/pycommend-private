"""
MOVNS v2 - Multi-Objective Variable Neighborhood Search with Full Improvements
Based on Dahite et al. (2022) Mathematics MDPI
Improvements: Normalization, Better Convergence, Smart Archive Management
"""

import numpy as np
import pickle
import random
from sklearn.cluster import KMeans
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOVNS_V2:
    """
    MOVNS v2 - Following Dahite et al. (2022) with improvements

    Key features:
    1. MOBI/P local search strategy
    2. Objective normalization for fair dominance
    3. Smart archive management with crowding distance
    4. Improved convergence criterion
    5. Dynamic bounds tracking
    """

    def __init__(self, main_package, archive_size=100, max_iterations=50,
                 k_max=4, track_metrics=False, min_no_improvement=10):
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.k_max = k_max
        self.track_metrics = track_metrics
        self.min_no_improvement = min_no_improvement

        self.n_objectives = 3
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        self.archive = []
        self.explored_solutions = set()
        self.counter_archive_improvement = 0

        self.neighborhoods = []
        self.initialize_neighborhoods()

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()

        self.initialize_archive()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'igd_plus': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS v2 initialized for '{main_package}'")
        print(f"Archive size: {archive_size}, Max iterations: {max_iterations}")
        print(f"Min no-improvement for stopping: {min_no_improvement}")

    def load_all_data(self):
        """Load required data matrices"""
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            relationships_data = pickle.load(f)
        self.rel_matrix = relationships_data['matrix']
        self.package_names = relationships_data['package_names']
        self.n_packages = len(self.package_names)

        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            similarity_data = pickle.load(f)
        self.sim_matrix = similarity_data['similarity_matrix']

        with open('data/package_embeddings_10k.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        self.embeddings = embeddings_data['embeddings']

        if self.main_package not in self.package_names:
            raise ValueError(f"Package '{self.main_package}' not found")
        self.main_package_idx = self.package_names.index(self.main_package)

    def initialize_semantic_components(self):
        """Initialize clustering for semantic coherence"""
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = [idx for idx in cluster_members
                                  if idx != self.main_package_idx][:100]

    def compute_candidate_pools(self):
        """Pre-compute candidate pools for efficiency"""
        self.threshold = 1.0

        cooccur_scores = [(idx, self.rel_matrix[self.main_package_idx, idx])
                         for idx in range(self.n_packages)
                         if idx != self.main_package_idx]
        cooccur_scores.sort(key=lambda x: x[1], reverse=True)
        self.cooccur_candidates = [idx for idx, _ in cooccur_scores[:200]]

        semantic_scores = [(idx, self.sim_matrix[self.main_package_idx, idx])
                          for idx in range(self.n_packages)
                          if idx != self.main_package_idx]
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        self.semantic_candidates = [idx for idx, _ in semantic_scores[:200]]

    def normalize_objectives(self, objectives):
        """Normalize objectives to [0,1] for fair comparison"""
        norm_obj = np.zeros_like(objectives)

        for i in range(len(objectives)):
            if self.obj_max[i] - self.obj_min[i] != 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / (self.obj_max[i] - self.obj_min[i])
            else:
                norm_obj[i] = 0.5

        norm_obj = np.clip(norm_obj, 0, 1)
        return norm_obj

    def update_bounds(self, objectives):
        """Update objective bounds dynamically"""
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

    def evaluate_objectives(self, chromosome):
        """Evaluate the 3 objectives"""
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[self.main_package_idx, idx]

        strong_links = len([idx for idx in indices
                           if self.rel_matrix[self.main_package_idx, idx] > self.threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[self.main_package_idx, idx]
                                 for idx in indices]

            if len(indices) > 1:
                internal_coherence = 0
                pair_count = 0
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        internal_coherence += self.sim_matrix[idx1, idx2]
                        pair_count += 1
                if pair_count > 0:
                    internal_coherence = internal_coherence / pair_count * 0.8
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        rss_score = len(indices)
        size_penalty = abs(len(indices) - self.ideal_size) * 0.05
        rss_score = rss_score * (1 + size_penalty)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        self.update_bounds(objectives)

        return objectives

    def dominates(self, obj1, obj2):
        """Check dominance using normalized objectives"""
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def is_non_dominated(self, objectives, archive):
        """Check if objectives are non-dominated in archive"""
        for item in archive:
            if isinstance(item, dict):
                obj = item['objectives']
            else:
                obj = item[1] if isinstance(item, tuple) else item
            if self.dominates(obj, objectives):
                return False
        return True

    def update_archive(self, solution, objectives):
        """Update archive following Dahite et al. (2022)"""
        self.update_bounds(objectives)

        dominated = []
        for i, sol_dict in enumerate(self.archive):
            if self.dominates(objectives, sol_dict['objectives']):
                dominated.append(i)
            elif self.dominates(sol_dict['objectives'], objectives):
                return False

        for i in reversed(dominated):
            del self.archive[i]

        self.archive.append({
            'chromosome': solution.copy(),
            'objectives': objectives.copy()
        })

        if len(dominated) > 0 or len(self.archive) == 1:
            self.counter_archive_improvement += 1

        return True

    def truncate_archive_with_crowding(self):
        """Truncate archive using crowding distance"""
        if len(self.archive) <= self.archive_limit:
            return

        objectives = np.array([sol['objectives'] for sol in self.archive])
        n_solutions = len(self.archive)
        n_objectives = objectives.shape[1]

        norm_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        crowding_distances = np.zeros(n_solutions)

        for m in range(n_objectives):
            sorted_indices = np.argsort(norm_objectives[:, m])

            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            for i in range(1, n_solutions - 1):
                if crowding_distances[sorted_indices[i]] != float('inf'):
                    distance = norm_objectives[sorted_indices[i + 1], m] - \
                              norm_objectives[sorted_indices[i - 1], m]
                    crowding_distances[sorted_indices[i]] += distance

        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def initialize_neighborhoods(self):
        """Initialize VNS neighborhoods following Dahite et al."""

        def n1_single_flip(solution):
            """N1: Small change - flip one bit"""
            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]
            removable = [idx for idx in indices if idx != self.main_package_idx]

            if random.random() < 0.5 and removable:
                idx = random.choice(removable)
                new_solution[idx] = 0
            else:
                if hasattr(self, 'cooccur_candidates'):
                    candidates = self.cooccur_candidates[:30]
                else:
                    candidates = list(range(self.n_packages))
                valid = [c for c in candidates if new_solution[c] == 0]
                if valid:
                    new_solution[random.choice(valid)] = 1

            return new_solution

        def n2_double_flip(solution):
            """N2: Medium change - flip 2 bits"""
            new_solution = solution.copy()
            for _ in range(2):
                new_solution = n1_single_flip(new_solution)
            return new_solution

        def n3_triple_flip(solution):
            """N3: Large change - flip 3 bits"""
            new_solution = solution.copy()
            for _ in range(3):
                new_solution = n1_single_flip(new_solution)
            return new_solution

        def n4_swap_segment(solution):
            """N4: Structural change - swap segment"""
            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]
            removable = [idx for idx in indices if idx != self.main_package_idx]

            if len(removable) >= 2:
                n_remove = min(2, len(removable))
                to_remove = random.sample(removable, n_remove)
                for idx in to_remove:
                    new_solution[idx] = 0

                if hasattr(self, 'semantic_candidates'):
                    candidates = self.semantic_candidates[:50]
                else:
                    candidates = list(range(self.n_packages))
                valid = [c for c in candidates if new_solution[c] == 0]
                if valid:
                    n_add = min(n_remove, len(valid))
                    to_add = random.sample(valid, n_add)
                    for idx in to_add:
                        new_solution[idx] = 1

            return new_solution

        self.neighborhoods = [n1_single_flip, n2_double_flip,
                            n3_triple_flip, n4_swap_segment]

    def shake(self, solution, neighborhood, intensity=1):
        """Shaking phase - diversification"""
        shaken = solution.copy()

        for _ in range(intensity):
            shaken = neighborhood(shaken)
            shaken = self.repair_solution(shaken)

        return shaken

    def mobi_p_local_search(self, solution, max_neighbors=10):
        """
        MOBI/P local search from Dahite et al. (2022)
        Multi-Objective Best Improvement with Probability
        """
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)

        local_archive = []

        for neighborhood in self.neighborhoods:
            for _ in range(max_neighbors // len(self.neighborhoods)):
                neighbor = neighborhood(solution)
                neighbor = self.repair_solution(neighbor)
                neighbor_obj = self.evaluate_objectives(neighbor)

                if self.dominates(neighbor_obj, best_objectives):
                    best_solution = neighbor
                    best_objectives = neighbor_obj
                    local_archive = [(neighbor, neighbor_obj)]
                elif not self.dominates(best_objectives, neighbor_obj):
                    if self.is_non_dominated(neighbor_obj, local_archive):
                        dominated = []
                        for i, (sol, obj) in enumerate(local_archive):
                            if self.dominates(neighbor_obj, obj):
                                dominated.append(i)

                        for i in reversed(dominated):
                            del local_archive[i]

                        local_archive.append((neighbor, neighbor_obj))

        return local_archive if local_archive else [(best_solution, best_objectives)]

    def repair_solution(self, chromosome):
        """Ensure solution validity"""
        indices = np.where(chromosome == 1)[0]

        if self.main_package_idx not in indices:
            chromosome[self.main_package_idx] = 1

        indices = np.where(chromosome == 1)[0]

        if len(indices) > self.max_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            n_remove = len(indices) - self.max_size
            if removable and n_remove > 0:
                to_remove = random.sample(removable, min(n_remove, len(removable)))
                for idx in to_remove:
                    chromosome[idx] = 0

        elif len(indices) < self.min_size:
            n_add = self.min_size - len(indices)
            candidates = list(range(self.n_packages))
            candidates = [c for c in candidates if chromosome[c] == 0]
            if candidates:
                to_add = random.sample(candidates, min(n_add, len(candidates)))
                for idx in to_add:
                    chromosome[idx] = 1

        return chromosome

    def best_insertion_heuristic(self):
        """Initialize solution using best insertion heuristic"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(self.min_size, min(7, self.max_size))

        strategy = random.choice(['cooccurrence', 'semantic', 'hybrid'])

        if strategy == 'cooccurrence':
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            if n_select > 0:
                selected = np.random.choice(self.cooccur_candidates[:50],
                                          n_select, replace=False)
                chromosome[selected] = 1

        elif strategy == 'semantic':
            n_select = min(target_size - 1, len(self.semantic_candidates))
            if n_select > 0:
                selected = np.random.choice(self.semantic_candidates[:50],
                                          n_select, replace=False)
                chromosome[selected] = 1

        else:
            n_cooccur = (target_size - 1) // 2
            n_semantic = target_size - 1 - n_cooccur

            if n_cooccur > 0 and self.cooccur_candidates:
                selected = np.random.choice(self.cooccur_candidates[:30],
                                          min(n_cooccur, len(self.cooccur_candidates)),
                                          replace=False)
                chromosome[selected] = 1

            if n_semantic > 0 and self.semantic_candidates:
                available = [c for c in self.semantic_candidates[:30]
                           if chromosome[c] == 0]
                if available:
                    selected = np.random.choice(available,
                                              min(n_semantic, len(available)),
                                              replace=False)
                    chromosome[selected] = 1

        return chromosome

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        print("Initializing archive...")

        for _ in range(20):
            chromosome = self.best_insertion_heuristic()
            objectives = self.evaluate_objectives(chromosome)
            self.update_archive(chromosome, objectives)

        for _ in range(50):
            chromosome = self.best_insertion_heuristic()
            objectives = self.evaluate_objectives(chromosome)
            self.update_bounds(objectives)

        self.truncate_archive_with_crowding()

        print(f"Archive initialized with {len(self.archive)} solutions")
        print(f"Objective bounds: LU=[{self.obj_min[0]:.1f}, {self.obj_max[0]:.1f}], "
              f"SS=[{self.obj_min[1]:.3f}, {self.obj_max[1]:.3f}], "
              f"RSS=[{self.obj_min[2]:.1f}, {self.obj_max[2]:.1f}]")

    def select_unexplored_solution(self):
        """Select solution from archive not previously explored"""
        unexplored = []
        for sol_dict in self.archive:
            sol_tuple = tuple(sol_dict['chromosome'])
            if sol_tuple not in self.explored_solutions:
                unexplored.append(sol_dict)

        if unexplored:
            selected = random.choice(unexplored)
            self.explored_solutions.add(tuple(selected['chromosome']))
            return selected['chromosome']
        else:
            self.explored_solutions.clear()
            selected = random.choice(self.archive)
            self.explored_solutions.add(tuple(selected['chromosome']))
            return selected['chromosome']

    def calculate_metrics(self):
        """Calculate quality metrics"""
        if not self.track_metrics or len(self.archive) < 3:
            return None

        objectives = np.array([sol['objectives'] for sol in self.archive])

        # Normalize objectives for hypervolume calculation
        normalized_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(normalized_objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(normalized_objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(normalized_objectives)

        return metrics

    def run(self):
        """
        Main MOVNS loop following Dahite et al. (2022)
        Algorithm 5: MOGVNS/P
        """
        print(f"\nStarting MOVNS v2 for {self.main_package}...")
        print("="*60)

        no_improvement_count = 0
        best_hv = 0
        MIN_ITERATIONS = 15
        IMPROVEMENT_THRESHOLD = 0.001

        previous_archive_count = self.counter_archive_improvement

        for iteration in range(self.max_iterations):

            if len(self.archive) > 0:
                s = self.select_unexplored_solution()
            else:
                s = self.best_insertion_heuristic()

            k = 0
            local_no_improvement = 0

            while k < self.k_max and local_no_improvement < 2:

                s_prime = self.shake(s, self.neighborhoods[k], intensity=min(k+1, 3))

                local_archive = self.mobi_p_local_search(s_prime, max_neighbors=10)

                archive_improved = False
                for (new_sol, new_obj) in local_archive:
                    if self.update_archive(new_sol, new_obj):
                        archive_improved = True

                if archive_improved:
                    k = 0
                    local_no_improvement = 0
                    s = new_sol
                else:
                    k += 1
                    local_no_improvement += 1

            self.truncate_archive_with_crowding()

            if self.counter_archive_improvement > previous_archive_count:
                no_improvement_count = 0
                previous_archive_count = self.counter_archive_improvement
            else:
                no_improvement_count += 1

            if self.track_metrics:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

                    current_hv = metrics.get('hypervolume', 0)

                    if iteration >= MIN_ITERATIONS:
                        if best_hv == 0:
                            best_hv = current_hv
                        else:
                            relative_improvement = (current_hv - best_hv) / (best_hv + 1e-10)

                            if relative_improvement > IMPROVEMENT_THRESHOLD:
                                best_hv = current_hv

            if iteration % 5 == 0:
                best = min(self.archive, key=lambda x: x['objectives'][0])
                print(f"Iteration {iteration}: Archive size={len(self.archive)}, "
                      f"Improvements={self.counter_archive_improvement}")
                print(f"  Best: LU={-best['objectives'][0]:.2f}, "
                      f"SS={-best['objectives'][1]:.4f}, RSS={best['objectives'][2]:.1f}")

                if self.track_metrics and metrics:
                    print(f"  HV={current_hv:.4f}, No-improvement={no_improvement_count}")

            if iteration >= MIN_ITERATIONS and no_improvement_count >= self.min_no_improvement:
                print(f"\nStopping at iteration {iteration} after {self.min_no_improvement} "
                      f"iterations without archive improvement")
                break

        solutions = []
        for sol_dict in self.archive:
            indices = np.where(sol_dict['chromosome'] == 1)[0]
            recommendations = [self.package_names[idx] for idx in indices
                             if idx != self.main_package_idx]
            solutions.append({
                'recommendations': recommendations,
                'objectives': sol_dict['objectives'],
                'linked_usage': -sol_dict['objectives'][0],
                'semantic_similarity': -sol_dict['objectives'][1],
                'set_size': sol_dict['objectives'][2]
            })

        print(f"\nFinal archive: {len(self.archive)} solutions")
        print(f"Total archive improvements: {self.counter_archive_improvement}")

        return solutions

    def get_metrics_history(self):
        """Return metrics history"""
        return self.metrics_history if self.track_metrics else None


def main(package_name='fastapi'):
    """Test MOVNS v2"""
    print(f"MOVNS v2 - Testing with '{package_name}'")
    print("="*60)

    movns = MOVNS_V2(package_name, archive_size=100, max_iterations=50,
                     track_metrics=True, min_no_improvement=10)

    solutions = movns.run()

    print(f"\nFound {len(solutions)} non-dominated solutions")

    if solutions:
        best_lu = max(solutions, key=lambda x: x['linked_usage'])
        best_ss = max(solutions, key=lambda x: x['semantic_similarity'])
        best_size = min(solutions, key=lambda x: x['set_size'])

        print(f"\nBest by Linked Usage: {best_lu['recommendations'][:5]}")
        print(f"  LU={best_lu['linked_usage']:.2f}, SS={best_lu['semantic_similarity']:.4f}")

        print(f"\nBest by Semantic: {best_ss['recommendations'][:5]}")
        print(f"  LU={best_ss['linked_usage']:.2f}, SS={best_ss['semantic_similarity']:.4f}")

        print(f"\nSmallest: {best_size['recommendations']}")
        print(f"  Size={best_size['set_size']}")

    return solutions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    main(package_name)