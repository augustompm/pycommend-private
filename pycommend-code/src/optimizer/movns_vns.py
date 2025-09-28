"""
MOVNS for PyCommend VNS - Multi-Objective Variable Neighborhood Search
Based on MOVND/PI from Dahite et al. (2022) with MOBI/P strategy
"""

import numpy as np
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOVNS_VNS:
    """
    MOVNS for library recommendation with 3 objectives:
    1. LU (Linked Usage): Maximize co-occurrence in real projects
    2. SS (Semantic Similarity): Maximize weighted topical coherence
    3. RSS (Recommended Set Size): Minimize set size

    Uses VNS with MOBI/P local search strategy instead of genetic operators
    """

    def __init__(self, main_package, archive_size=100, max_iterations=30, track_metrics=False):
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5
        self.track_metrics = track_metrics

        self.k_max = 4
        self.archive = []
        self.neighborhoods = None

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.reference_set = None
            self.metrics_history = {
                'hypervolume': [],
                'igd_plus': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS initialized for '{main_package}'")
        print(f"Using 3 objectives: LU, SS, RSS")
        print(f"VNS with {self.k_max} neighborhoods and MOBI/P local search")

    def load_all_data(self):
        """Load all required data matrices"""
        print("Loading data matrices...")

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
            raise ValueError(f"Package '{self.main_package}' not found in dataset")
        self.main_package_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic clustering for better initialization"""
        print("Initializing semantic components...")

        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.sum(self.cluster_labels == self.target_cluster)
        print(f"Target package in cluster {self.target_cluster} with {cluster_members} members")

    def compute_candidate_pools(self):
        """Precompute candidate pools for smart initialization"""
        main_idx = self.main_package_idx

        cooccur_scores = self.rel_matrix[main_idx].toarray().flatten()
        self.cooccur_candidates = np.argsort(cooccur_scores)[::-1]
        self.cooccur_candidates = self.cooccur_candidates[cooccur_scores[self.cooccur_candidates] > 0][:200]

        target_embedding = self.embeddings[main_idx]
        similarities = cosine_similarity([target_embedding], self.embeddings)[0]
        self.semantic_candidates = np.argsort(similarities)[::-1][1:201]

        self.cluster_candidates = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = self.cluster_candidates[self.cluster_candidates != main_idx]

        print(f"Candidate pools ready: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

    def evaluate_objectives(self, chromosome):
        """
        Evaluate 3 objectives:
        LU: Linked Usage (maximize co-occurrence)
        SS: Semantic Similarity (maximize topical coherence)
        RSS: Recommended Set Size (minimize)
        """
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        threshold = np.percentile(self.rel_matrix[main_idx].data, 75) if self.rel_matrix[main_idx].data.size > 0 else 1.0
        strong_links = len([idx for idx in indices if self.rel_matrix[main_idx, idx] > threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            if len(indices) > 1:
                selected_embeddings = self.embeddings[indices]
                centroid = np.mean(selected_embeddings, axis=0)
                coherence_scores = cosine_similarity(selected_embeddings, [centroid]).flatten()
                internal_coherence = np.mean(coherence_scores)
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())
            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        rss_score = len(indices)
        if len(indices) < self.ideal_size:
            rss_score += (self.ideal_size - len(indices)) * 0.5
        elif len(indices) > self.ideal_size * 1.5:
            rss_score += (len(indices) - self.ideal_size * 1.5) * 0.3

        return np.array([-lu_score, -ss_score, rss_score])

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solutions using domain knowledge"""
        chromosome = np.zeros(self.n_packages, dtype=np.int8)

        if strategy == 'small':
            size = random.randint(2, 4)
        elif strategy == 'medium':
            size = random.randint(5, 7)
        elif strategy == 'large':
            size = random.randint(8, 12)
        else:
            size = random.randint(3, 10)

        if strategy == 'cooccur' and len(self.cooccur_candidates) > 0:
            weights = self.rel_matrix[self.main_package_idx, self.cooccur_candidates].toarray().flatten()
            if weights.sum() > 0:
                weights = weights / weights.sum()
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False, p=weights)
            else:
                selected = np.random.choice(self.cooccur_candidates,
                                          min(size, len(self.cooccur_candidates)),
                                          replace=False)

        elif strategy == 'semantic' and len(self.semantic_candidates) > 0:
            selected = self.semantic_candidates[:min(size, len(self.semantic_candidates))]

        elif strategy == 'cluster' and len(self.cluster_candidates) > 0:
            selected = np.random.choice(self.cluster_candidates,
                                      min(size, len(self.cluster_candidates)),
                                      replace=False)

        else:
            candidates = []

            if len(self.cooccur_candidates) > 0:
                candidates.extend(self.cooccur_candidates[:size//2])
            if len(self.semantic_candidates) > 0:
                candidates.extend(self.semantic_candidates[:size//3])
            if len(self.cluster_candidates) > 0:
                sample_size = min(size//4, len(self.cluster_candidates))
                candidates.extend(np.random.choice(self.cluster_candidates, sample_size, replace=False))

            if len(candidates) > 0:
                candidates = list(set(candidates))
                selected = np.random.choice(candidates, min(size, len(candidates)), replace=False)
            else:
                valid_indices = list(range(self.n_packages))
                valid_indices.remove(self.main_package_idx)
                selected = np.random.choice(valid_indices, size, replace=False)

        chromosome[selected] = 1
        return chromosome

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        print("Initializing archive...")

        strategies = ['small', 'medium', 'large', 'cooccur', 'semantic', 'hybrid']
        solutions_per_strategy = self.archive_limit // len(strategies)

        for strategy in strategies:
            for _ in range(solutions_per_strategy):
                chromosome = self.smart_initialization(strategy)
                objectives = self.evaluate_objectives(chromosome)
                self.update_archive(chromosome, objectives)

        while len(self.archive) < self.archive_limit // 2:
            chromosome = self.smart_initialization('hybrid')
            objectives = self.evaluate_objectives(chromosome)
            self.update_archive(chromosome, objectives)

        print(f"Archive initialized with {len(self.archive)} solutions")

    def define_neighborhoods(self):
        """Define 4 VNS neighborhoods based on problem structure"""

        def n1_single_flip(solution):
            """Small change: flip 1 random bit"""
            s_new = solution.copy()
            idx = np.random.randint(self.n_packages)
            if idx != self.main_package_idx:
                s_new[idx] = 1 - s_new[idx]
            return s_new

        def n2_multi_flip(solution):
            """Medium change: flip 2-3 bits"""
            s_new = solution.copy()
            n_flips = np.random.randint(2, 4)
            indices = [i for i in range(self.n_packages) if i != self.main_package_idx]
            flip_indices = np.random.choice(indices, min(n_flips, len(indices)), replace=False)
            s_new[flip_indices] = 1 - s_new[flip_indices]
            return s_new

        def n3_segment_exchange(solution):
            """Large change: exchange segment using co-occurrence"""
            s_new = solution.copy()
            active = np.where(solution == 1)[0]

            if len(active) > 2 and len(self.cooccur_candidates) > 0:
                n_remove = min(len(active) // 3, 3)
                to_remove = np.random.choice(active, n_remove, replace=False)
                s_new[to_remove] = 0

                available = [c for c in self.cooccur_candidates[:50] if s_new[c] == 0]
                if available:
                    n_add = min(n_remove, len(available))
                    to_add = np.random.choice(available, n_add, replace=False)
                    s_new[to_add] = 1

            return s_new

        def n4_smart_adjustment(solution):
            """Smart change: adjust to ideal size using domain knowledge"""
            s_new = solution.copy()
            current_size = np.sum(solution)

            if current_size < self.ideal_size and len(self.cooccur_candidates) > 0:
                available = [c for c in self.cooccur_candidates if s_new[c] == 0]
                for c in available[:self.ideal_size - current_size]:
                    s_new[c] = 1
                    if np.sum(s_new) >= self.ideal_size:
                        break

            elif current_size > self.ideal_size * 1.5:
                active = np.where(solution == 1)[0]
                scores = [self.rel_matrix[self.main_package_idx, i] for i in active]
                weakest = active[np.argsort(scores)[:int(current_size - self.ideal_size)]]
                s_new[weakest] = 0

            return s_new

        return [n1_single_flip, n2_multi_flip, n3_segment_exchange, n4_smart_adjustment]

    def shake(self, solution, neighborhood, intensity):
        """Perturbation with adaptive intensity"""
        s = solution
        for _ in range(intensity):
            s = neighborhood(s)
            s = self.repair_solution(s)
        return s

    def repair_solution(self, solution):
        """Ensure solution satisfies constraints"""
        current_size = np.sum(solution)

        if current_size < self.min_size:
            candidates = np.where(solution == 0)[0]
            candidates = [c for c in candidates if c != self.main_package_idx]
            if candidates:
                n_add = self.min_size - current_size
                add_indices = np.random.choice(candidates, min(n_add, len(candidates)), replace=False)
                solution[add_indices] = 1

        elif current_size > self.max_size:
            active = np.where(solution == 1)[0]
            n_remove = current_size - self.max_size
            remove_indices = np.random.choice(active, n_remove, replace=False)
            solution[remove_indices] = 0

        return solution

    def mobi_p_local_search(self, solution, neighborhood):
        """
        Multi-Objective Best Improvement with Pareto
        Core innovation from Dahite et al. (2022)
        """
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)
        pareto_set = []

        for _ in range(20):
            neighbor = neighborhood(solution)
            neighbor = self.repair_solution(neighbor)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if self.dominates(neighbor_obj, best_objectives):
                best_solution = neighbor
                best_objectives = neighbor_obj
                pareto_set = [(neighbor, neighbor_obj)]
            elif not self.dominates(best_objectives, neighbor_obj):
                pareto_set.append((neighbor, neighbor_obj))

        return self.filter_non_dominated(pareto_set)

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (for minimization)"""
        return all(obj1 <= obj2) and any(obj1 < obj2)

    def filter_non_dominated(self, solutions):
        """Return only non-dominated solutions from set"""
        if not solutions:
            return []

        non_dominated = []
        for i, (sol_i, obj_i) in enumerate(solutions):
            is_dominated = False
            for j, (sol_j, obj_j) in enumerate(solutions):
                if i != j and self.dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append((sol_i, obj_i))

        return non_dominated

    def update_archive(self, solution, objectives):
        """Update Pareto archive with new solution"""
        self.archive = [s for s in self.archive
                       if not self.dominates(objectives, s['objectives'])]

        for s in self.archive:
            if self.dominates(s['objectives'], objectives):
                return False

        self.archive.append({
            'chromosome': solution,
            'objectives': objectives
        })
        return True

    def truncate_archive(self):
        """Maintain archive size using diversity preservation"""
        if len(self.archive) <= self.archive_limit:
            return

        objectives = np.array([s['objectives'] for s in self.archive])

        distances = []
        for i in range(len(self.archive)):
            min_dist = float('inf')
            for j in range(len(self.archive)):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        sorted_indices = np.argsort(distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def calculate_metrics(self):
        """Calculate quality metrics for current archive"""
        if not self.archive or not self.track_metrics:
            return None

        objectives = np.array([s['objectives'] for s in self.archive])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(objectives)

        if self.reference_set is not None:
            metrics['igd_plus'] = self.metrics_calculator.igd_plus(objectives, self.reference_set)
        else:
            metrics['igd_plus'] = None

        metrics['spacing'] = self.metrics_calculator.spacing(objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(objectives)

        return metrics

    def generate_reference_set(self):
        """Generate reference set for IGD+ calculation"""
        temp_solutions = []
        for _ in range(50):
            chromosome = self.smart_initialization('hybrid')
            objectives = self.evaluate_objectives(chromosome)
            temp_solutions.append(objectives)

        temp_solutions = np.array(temp_solutions)

        lu_best = np.min(temp_solutions[:, 0]) * 1.2
        lu_worst = np.max(temp_solutions[:, 0]) * 0.8
        ss_best = np.min(temp_solutions[:, 1]) * 1.2
        ss_worst = np.max(temp_solutions[:, 1]) * 0.8
        rss_best = np.min(temp_solutions[:, 2]) * 0.8
        rss_worst = np.max(temp_solutions[:, 2]) * 1.2

        n_per_dim = 5
        lu_range = np.linspace(lu_best, lu_worst, n_per_dim)
        ss_range = np.linspace(ss_best, ss_worst, n_per_dim)
        rss_range = np.linspace(rss_best, rss_worst, n_per_dim)

        ref_points = []
        for lu in lu_range:
            for ss in ss_range:
                for rss in rss_range:
                    ref_points.append([lu, ss, rss])

        self.reference_set = np.array(ref_points)

    def run(self):
        """Main MOVNS algorithm with VNS loop"""
        print(f"\nStarting MOVNS for {self.main_package}...")
        print("="*60)

        self.initialize_archive()
        self.neighborhoods = self.define_neighborhoods()

        if self.track_metrics:
            self.generate_reference_set()

        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            improved = False
            archive_copy = self.archive.copy()

            for sol_dict in archive_copy:
                solution = sol_dict['chromosome']
                k = 0

                while k < self.k_max:
                    s_prime = self.shake(solution, self.neighborhoods[k], intensity=k+1)

                    improved_solutions = self.mobi_p_local_search(s_prime, self.neighborhoods[k])

                    archive_updated = False
                    for (new_sol, new_obj) in improved_solutions:
                        if self.update_archive(new_sol, new_obj):
                            archive_updated = True
                            improved = True

                    if archive_updated:
                        k = 0
                    else:
                        k += 1

            self.truncate_archive()

            if self.track_metrics:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

            if iteration % 5 == 0:
                best = min(self.archive, key=lambda x: x['objectives'][0])
                print(f"Iteration {iteration}: Archive size={len(self.archive)}")
                print(f"  Best: LU={-best['objectives'][0]:.2f}, "
                      f"SS={-best['objectives'][1]:.4f}, RSS={best['objectives'][2]:.1f}")

                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={metrics.get('hypervolume', 0):.4f}, "
                          f"Spacing={metrics.get('spacing', 0):.4f}")

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= 5:
                    print(f"Early stopping at iteration {iteration} (no improvement)")
                    break
            else:
                no_improvement_count = 0

        return self.get_final_solutions()

    def get_final_solutions(self):
        """Extract final Pareto-optimal solutions"""
        final_solutions = []

        for sol_dict in self.archive:
            chromosome = sol_dict['chromosome']
            objectives = sol_dict['objectives']
            indices = np.where(chromosome == 1)[0]
            packages = [self.package_names[i] for i in indices]

            final_solutions.append({
                'packages': packages,
                'objectives': {
                    'linked_usage': -objectives[0],
                    'semantic_similarity': -objectives[1],
                    'set_size': objectives[2]
                }
            })

        return final_solutions

    def get_metrics_history(self):
        """Return metrics history if tracking enabled"""
        if self.track_metrics:
            return self.metrics_history
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MOVNS for PyCommend')
    parser.add_argument('--package', type=str, default='numpy',
                       help='Main package for recommendations')
    parser.add_argument('--archive-size', type=int, default=100,
                       help='Archive size')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Maximum VNS iterations')
    parser.add_argument('--track-metrics', action='store_true',
                       help='Track quality metrics')

    args = parser.parse_args()

    movns = MOVNS_VNS(
        main_package=args.package,
        archive_size=args.archive_size,
        max_iterations=args.iterations,
        track_metrics=args.track_metrics
    )

    solutions = movns.run()

    print(f"\n{'='*60}")
    print(f"Final Pareto-optimal solutions: {len(solutions)}")
    print(f"{'='*60}")

    for i, sol in enumerate(solutions[:5], 1):
        print(f"\nSolution {i}:")
        print(f"  Packages: {', '.join(sol['packages'])}")
        print(f"  LU: {sol['objectives']['linked_usage']:.2f}")
        print(f"  SS: {sol['objectives']['semantic_similarity']:.4f}")
        print(f"  Size: {sol['objectives']['set_size']:.0f}")