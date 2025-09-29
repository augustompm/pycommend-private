"""
MOVNS Improved - Multi-Objective Variable Neighborhood Search with Normalization
Based on Dahite et al. (2022) with improvements from MOEA/D analysis
Key improvements: Objective normalization, better early stopping, smart archive management
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


class MOVNS_Improved:
    """
    Improved MOVNS with normalization and better convergence
    Key improvements from MOEA/D:
    1. Objective normalization for fair dominance checking
    2. Dynamic bounds tracking
    3. Better early stopping criterion
    4. Smart archive management with crowding distance
    """

    def __init__(self, main_package, archive_size=100, max_iterations=50,
                 k_max=4, track_metrics=False, min_no_improvement=10):
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.k_max = k_max
        self.track_metrics = track_metrics
        self.min_no_improvement = min_no_improvement  # Minimum iterations without improvement before stopping

        self.n_objectives = 3
        self.min_size = 2
        self.max_size = 15
        self.ideal_size = 5

        # Initialize objective bounds (critical for normalization)
        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        # Archive for non-dominated solutions
        self.archive = []

        # VNS neighborhoods
        self.neighborhoods = []
        self.initialize_neighborhoods()

        # Load data
        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()

        # Initialize archive with diverse solutions
        self.initialize_archive()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'igd_plus': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS Improved initialized for '{main_package}'")
        print(f"Using {self.n_objectives} objectives with normalization")
        print(f"Early stopping after {min_no_improvement} iterations without improvement")

    def normalize_objectives(self, objectives):
        """
        Normalize objectives to [0,1] range
        Critical for fair dominance checking with different scales
        """
        norm_obj = np.zeros_like(objectives)

        for i in range(len(objectives)):
            if self.obj_max[i] - self.obj_min[i] != 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / (self.obj_max[i] - self.obj_min[i])
            else:
                norm_obj[i] = 0.5

        norm_obj = np.clip(norm_obj, 0, 1)
        return norm_obj

    def update_bounds(self, objectives):
        """
        Update objective bounds dynamically
        Ensures normalization adapts to actual data range
        """
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

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
        """Initialize semantic clustering for coherence calculation"""
        n_clusters = min(200, self.n_packages // 50)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.target_cluster = self.cluster_labels[self.main_package_idx]

        cluster_members = np.where(self.cluster_labels == self.target_cluster)[0]
        self.cluster_candidates = [idx for idx in cluster_members
                                  if idx != self.main_package_idx][:100]

        print(f"Target package in cluster {self.target_cluster} with {len(cluster_members)} members")

    def compute_candidate_pools(self):
        """Pre-compute candidate pools for efficient search"""
        self.threshold = 1.0

        # Cooccurrence-based candidates
        cooccur_scores = [(idx, self.rel_matrix[self.main_package_idx, idx])
                         for idx in range(self.n_packages)
                         if idx != self.main_package_idx]
        cooccur_scores.sort(key=lambda x: x[1], reverse=True)
        self.cooccur_candidates = [idx for idx, _ in cooccur_scores[:200]]

        # Semantic similarity candidates
        semantic_scores = [(idx, self.sim_matrix[self.main_package_idx, idx])
                          for idx in range(self.n_packages)
                          if idx != self.main_package_idx]
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        self.semantic_candidates = [idx for idx, _ in semantic_scores[:200]]

        print(f"Candidate pools ready: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

    def evaluate_objectives(self, chromosome):
        """Evaluate 3 objectives with proper scaling"""
        main_idx = self.main_package_idx
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([float('inf')] * 3)

        # LU: Linked Usage
        linked_usage = 0
        for idx in indices:
            linked_usage += self.rel_matrix[main_idx, idx]

        strong_links = len([idx for idx in indices
                           if self.rel_matrix[main_idx, idx] > self.threshold])
        lu_score = linked_usage * (1 + 0.1 * strong_links)

        # SS: Semantic Similarity
        if len(indices) > 0:
            direct_similarities = [self.sim_matrix[main_idx, idx] for idx in indices]

            if len(indices) > 1:
                internal_coherence = 0
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        internal_coherence += self.sim_matrix[idx1, idx2]
                internal_coherence /= (len(indices) * (len(indices) - 1) / 2)
                internal_coherence *= 0.8
            else:
                internal_coherence = 0.5

            weights = 1.0 / (1.0 + np.arange(len(direct_similarities)))
            weighted_sim = np.average(direct_similarities, weights=weights/weights.sum())

            ss_score = 0.7 * weighted_sim + 0.3 * internal_coherence
        else:
            ss_score = 0

        # RSS: Recommended Set Size
        rss_score = len(indices)
        size_penalty = abs(len(indices) - self.ideal_size) * 0.05
        rss_score = rss_score * (1 + size_penalty)

        objectives = np.array([-lu_score, -ss_score, rss_score])

        # Update bounds for normalization
        self.update_bounds(objectives)

        return objectives

    def dominates(self, obj1, obj2):
        """
        Check if obj1 dominates obj2 using NORMALIZED objectives
        This ensures fair comparison across different scales
        """
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def update_archive(self, solution, objectives):
        """
        Update archive with smart dominance checking using normalized objectives
        """
        # Update bounds first
        self.update_bounds(objectives)

        # Check dominance using normalized objectives
        dominated = []
        for i, sol_dict in enumerate(self.archive):
            if self.dominates(objectives, sol_dict['objectives']):
                dominated.append(i)
            elif self.dominates(sol_dict['objectives'], objectives):
                return False

        # Remove dominated solutions
        for i in reversed(dominated):
            del self.archive[i]

        # Add new solution
        self.archive.append({
            'chromosome': solution.copy(),
            'objectives': objectives.copy()
        })

        return True

    def truncate_archive(self):
        """
        Truncate archive using crowding distance (not random)
        Preserves diversity better than random removal
        """
        if len(self.archive) <= self.archive_limit:
            return

        # Calculate crowding distance
        objectives = np.array([sol['objectives'] for sol in self.archive])
        n_solutions = len(self.archive)
        n_objectives = objectives.shape[1]

        # Normalize for distance calculation
        norm_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        crowding_distances = np.zeros(n_solutions)

        for m in range(n_objectives):
            sorted_indices = np.argsort(norm_objectives[:, m])

            # Boundary solutions get infinite distance
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            # Calculate distance for internal solutions
            for i in range(1, n_solutions - 1):
                if crowding_distances[sorted_indices[i]] != float('inf'):
                    distance = norm_objectives[sorted_indices[i + 1], m] - \
                              norm_objectives[sorted_indices[i - 1], m]
                    crowding_distances[sorted_indices[i]] += distance

        # Sort by crowding distance and keep most diverse
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def initialize_neighborhoods(self):
        """Initialize VNS neighborhood structures"""
        def n1_single_flip(solution):
            """Small perturbation: flip one bit"""
            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]
            removable = [idx for idx in indices if idx != self.main_package_idx]

            if random.random() < 0.5 and removable:
                idx = random.choice(removable)
                new_solution[idx] = 0
            else:
                candidates = self.cooccur_candidates[:30] if hasattr(self, 'cooccur_candidates') else []
                valid = [c for c in candidates if new_solution[c] == 0]
                if valid:
                    new_solution[random.choice(valid)] = 1

            return new_solution

        def n2_multi_flip(solution):
            """Medium perturbation: flip 2-3 bits"""
            new_solution = solution.copy()
            n_flips = random.randint(2, 3)

            for _ in range(n_flips):
                new_solution = n1_single_flip(new_solution)

            return new_solution

        def n3_segment_exchange(solution):
            """Large structural change"""
            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]
            removable = [idx for idx in indices if idx != self.main_package_idx]

            if len(removable) >= 2:
                n_remove = min(random.randint(1, 3), len(removable))
                to_remove = random.sample(removable, n_remove)
                for idx in to_remove:
                    new_solution[idx] = 0

                candidates = self.semantic_candidates[:50] if hasattr(self, 'semantic_candidates') else []
                valid = [c for c in candidates if new_solution[c] == 0]
                if valid:
                    n_add = min(n_remove, len(valid))
                    to_add = random.sample(valid, n_add)
                    for idx in to_add:
                        new_solution[idx] = 1

            return new_solution

        def n4_smart_adjustment(solution):
            """Domain-specific intelligent adjustment"""
            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]
            current_size = len(indices)

            if current_size > self.ideal_size + 2:
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    weak_links = [(idx, self.rel_matrix[self.main_package_idx, idx])
                                 for idx in removable]
                    weak_links.sort(key=lambda x: x[1])
                    n_remove = min(2, len(weak_links))
                    for idx, _ in weak_links[:n_remove]:
                        new_solution[idx] = 0

            elif current_size < self.ideal_size - 1:
                candidates = self.cooccur_candidates[:50] if hasattr(self, 'cooccur_candidates') else []
                valid = [c for c in candidates if new_solution[c] == 0]
                if valid:
                    n_add = min(2, len(valid))
                    to_add = random.sample(valid, n_add)
                    for idx in to_add:
                        new_solution[idx] = 1

            return new_solution

        self.neighborhoods = [n1_single_flip, n2_multi_flip, n3_segment_exchange, n4_smart_adjustment]

    def shake(self, solution, neighborhood, intensity=1):
        """Shaking phase of VNS"""
        shaken = solution.copy()

        for _ in range(intensity):
            shaken = neighborhood(shaken)
            shaken = self.repair_solution(shaken)

        return shaken

    def mobi_p_local_search(self, solution, neighborhood, samples=3):
        """MOBI/P local search strategy"""
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)

        candidates = []

        for _ in range(samples):
            neighbor = neighborhood(solution)
            neighbor = self.repair_solution(neighbor)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if self.dominates(neighbor_obj, best_objectives):
                best_solution = neighbor
                best_objectives = neighbor_obj
                candidates = [(neighbor, neighbor_obj)]
            elif not self.dominates(best_objectives, neighbor_obj):
                candidates.append((neighbor, neighbor_obj))

        non_dominated = []
        for i, (sol1, obj1) in enumerate(candidates):
            is_dominated = False
            for j, (sol2, obj2) in enumerate(candidates):
                if i != j and self.dominates(obj2, obj1):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append((sol1, obj1))

        return non_dominated if non_dominated else [(best_solution, best_objectives)]

    def repair_solution(self, chromosome):
        """Repair solution to ensure validity"""
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

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solution with domain knowledge"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(self.min_size, min(7, self.max_size))

        if strategy == 'cooccurrence':
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            if n_select > 0:
                selected = np.random.choice(self.cooccur_candidates[:50], n_select, replace=False)
                chromosome[selected] = 1

        elif strategy == 'semantic':
            n_select = min(target_size - 1, len(self.semantic_candidates))
            if n_select > 0:
                selected = np.random.choice(self.semantic_candidates[:50], n_select, replace=False)
                chromosome[selected] = 1

        elif strategy == 'cluster':
            n_select = min(target_size - 1, len(self.cluster_candidates))
            if n_select > 0:
                selected = np.random.choice(self.cluster_candidates, n_select, replace=False)
                chromosome[selected] = 1

        else:  # hybrid
            n_cooccur = (target_size - 1) // 3
            n_semantic = (target_size - 1) // 3
            n_cluster = target_size - 1 - n_cooccur - n_semantic

            if n_cooccur > 0 and self.cooccur_candidates:
                selected = np.random.choice(self.cooccur_candidates[:30],
                                          min(n_cooccur, len(self.cooccur_candidates)),
                                          replace=False)
                chromosome[selected] = 1

            if n_semantic > 0 and self.semantic_candidates:
                available = [c for c in self.semantic_candidates[:30] if chromosome[c] == 0]
                if available:
                    selected = np.random.choice(available,
                                              min(n_semantic, len(available)),
                                              replace=False)
                    chromosome[selected] = 1

            if n_cluster > 0 and self.cluster_candidates:
                available = [c for c in self.cluster_candidates if chromosome[c] == 0]
                if available:
                    selected = np.random.choice(available,
                                              min(n_cluster, len(available)),
                                              replace=False)
                    chromosome[selected] = 1

        return chromosome

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        print("Initializing archive with diverse population...")
        strategies = ['cooccurrence', 'semantic', 'cluster', 'hybrid']

        for strategy in strategies:
            for _ in range(5):
                chromosome = self.smart_initialization(strategy)
                objectives = self.evaluate_objectives(chromosome)
                self.update_archive(chromosome, objectives)

        self.truncate_archive()

        # Generate 100 random solutions to better understand objective ranges
        for _ in range(100):
            chromosome = self.smart_initialization('hybrid')
            objectives = self.evaluate_objectives(chromosome)
            # Just update bounds, don't add to archive
            self.update_bounds(objectives)

        print(f"Archive initialized with {len(self.archive)} solutions")
        print(f"Objective bounds: LU=[{self.obj_min[0]:.1f}, {self.obj_max[0]:.1f}], "
              f"SS=[{self.obj_min[1]:.3f}, {self.obj_max[1]:.3f}], "
              f"RSS=[{self.obj_min[2]:.1f}, {self.obj_max[2]:.1f}]")

    def calculate_metrics(self):
        """Calculate quality metrics for current archive"""
        if not self.track_metrics or len(self.archive) < 3:
            return None

        objectives = np.array([sol['objectives'] for sol in self.archive])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(objectives)

        return metrics

    def run(self):
        """
        Main MOVNS loop with improved convergence criteria
        """
        print(f"\nStarting MOVNS Improved for {self.main_package}...")
        print("="*60)

        # Convergence tracking
        no_improvement_count = 0
        best_hv = 0
        MIN_ITERATIONS = 15  # Don't stop before this
        IMPROVEMENT_THRESHOLD = 0.001  # 0.1% improvement threshold

        for iteration in range(self.max_iterations):
            improved_this_iteration = False

            # Adaptive samples based on progress
            adaptive_samples = min(2 + iteration // 10, 5)

            # Select solution from archive
            if len(self.archive) > 0:
                # Select based on crowding distance (explore less crowded areas)
                sol_dict = random.choice(self.archive)
                solution = sol_dict['chromosome']
            else:
                solution = self.smart_initialization('hybrid')

            k = 0
            vns_no_improvement = 0

            # VNS loop for single solution
            while k < self.k_max and vns_no_improvement < 3:
                # Shaking phase
                s_prime = self.shake(solution, self.neighborhoods[k], intensity=k+1)

                # Local search with MOBI/P
                improved_solutions = self.mobi_p_local_search(s_prime,
                                                             self.neighborhoods[k],
                                                             samples=adaptive_samples)

                # Update archive
                archive_updated = False
                for (new_sol, new_obj) in improved_solutions:
                    if self.update_archive(new_sol, new_obj):
                        archive_updated = True
                        improved_this_iteration = True
                        solution = new_sol

                # Neighborhood change
                if archive_updated:
                    k = 0
                    vns_no_improvement = 0
                else:
                    k += 1
                    vns_no_improvement += 1

            # Archive management
            self.truncate_archive()

            # Population injection every 10 iterations (from MOEA/D)
            if iteration % 10 == 0 and iteration > 0 and len(self.archive) > 10:
                # Inject archive diversity back into search
                diverse_idx = random.randint(0, min(5, len(self.archive) - 1))
                solution = self.archive[diverse_idx]['chromosome']

            # Track metrics and convergence
            if self.track_metrics:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

                    current_hv = metrics.get('hypervolume', 0)

                    # Check improvement (only after minimum iterations)
                    if iteration >= MIN_ITERATIONS:
                        if best_hv == 0:
                            best_hv = current_hv
                        else:
                            relative_improvement = (current_hv - best_hv) / (best_hv + 1e-10)

                            if relative_improvement > IMPROVEMENT_THRESHOLD:
                                best_hv = current_hv
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1

            # Progress reporting
            if iteration % 5 == 0:
                best = min(self.archive, key=lambda x: x['objectives'][0])
                print(f"Iteration {iteration}: Archive size={len(self.archive)}")
                print(f"  Best: LU={-best['objectives'][0]:.2f}, "
                      f"SS={-best['objectives'][1]:.4f}, RSS={best['objectives'][2]:.1f}")

                if self.track_metrics and metrics:
                    print(f"  Metrics: HV={current_hv:.4f}, "
                          f"Spacing={metrics.get('spacing', 0):.4f}")
                    print(f"  No improvement count: {no_improvement_count}/{self.min_no_improvement}")

            # Early stopping check (only after minimum iterations)
            if iteration >= MIN_ITERATIONS and no_improvement_count >= self.min_no_improvement:
                print(f"\nStopping at iteration {iteration} after {self.min_no_improvement} "
                      f"iterations without significant improvement (>{IMPROVEMENT_THRESHOLD*100}%)")
                print(f"Final hypervolume: {best_hv:.4f}")
                break

        # Convert archive to solution format
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

        return solutions

    def get_metrics_history(self):
        """Return metrics history for analysis"""
        return self.metrics_history if self.track_metrics else None


def main(package_name='fastapi'):
    """Test function"""
    print(f"MOVNS Improved - Library Recommendation for '{package_name}'")
    print("="*60)

    movns = MOVNS_Improved(package_name, archive_size=100, max_iterations=50,
                           track_metrics=True, min_no_improvement=10)

    solutions = movns.run()

    print(f"\nFound {len(solutions)} non-dominated solutions")

    if solutions:
        best_lu = max(solutions, key=lambda x: x['linked_usage'])
        best_ss = max(solutions, key=lambda x: x['semantic_similarity'])
        best_size = min(solutions, key=lambda x: x['set_size'])

        print(f"\nBest by Linked Usage: {best_lu['recommendations'][:5]}")
        print(f"  LU={best_lu['linked_usage']:.2f}, SS={best_lu['semantic_similarity']:.4f}, Size={best_lu['set_size']}")

        print(f"\nBest by Semantic Similarity: {best_ss['recommendations'][:5]}")
        print(f"  LU={best_ss['linked_usage']:.2f}, SS={best_ss['semantic_similarity']:.4f}, Size={best_ss['set_size']}")

        print(f"\nSmallest Set: {best_size['recommendations']}")
        print(f"  LU={best_size['linked_usage']:.2f}, SS={best_size['semantic_similarity']:.4f}, Size={best_size['set_size']}")

    return solutions


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        package_name = sys.argv[1]
    else:
        package_name = 'fastapi'

    main(package_name)