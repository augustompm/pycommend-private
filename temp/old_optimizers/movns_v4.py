"""
MOVNS v4 - Calibrated Variable Neighborhood Search
Based on extensive hyperparameter exploration and literature insights
Target: Surpass MOEA/D performance (HV > 0.26)
"""

import numpy as np
import random
import pickle
import time
from collections import deque

class MOVNS_V4:
    def __init__(self, main_package,
                 archive_size=150,           # Increased from 50
                 secondary_archive_size=50,   # New secondary archive
                 max_iterations=75,           # More iterations
                 mobi_p_neighbors=50,         # More neighbors tested
                 shaking_base_intensity=2,    # Base shaking
                 shaking_max_intensity=10,    # Max shaking
                 n_neighborhoods=6,           # More neighborhoods
                 adaptive=True,               # Adaptive parameters
                 memory_size=100,             # Memory for learning
                 track_metrics=False,
                 min_no_improvement=15):      # More patient
        """
        MOVNS v4 with calibrated hyperparameters
        """
        self.main_package = main_package
        self.archive_limit = archive_size
        self.secondary_limit = secondary_archive_size
        self.max_iterations = max_iterations
        self.mobi_p_neighbors = mobi_p_neighbors
        self.shaking_base = shaking_base_intensity
        self.shaking_max = shaking_max_intensity
        self.n_neighborhoods = n_neighborhoods
        self.adaptive = adaptive
        self.memory_size = memory_size
        self.track_metrics = track_metrics
        self.min_no_improvement = min_no_improvement

        # Archives
        self.archive = []                    # Primary Pareto archive
        self.secondary_archive = []          # Secondary promising solutions
        self.explored_solutions = set()

        # Counters and memory
        self.counter_archive_improvement = 0
        self.iteration_improvements = []
        self.neighborhood_success = [0] * n_neighborhoods
        self.memory = deque(maxlen=memory_size)

        # Objective bounds
        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        # Problem parameters
        self.min_size = 2
        self.max_size = 10    # Further reduced to avoid LU explosion
        self.ideal_size = 5

        # Load data and initialize
        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.generate_weight_vectors()
        self.initialize_archives()
        self.initialize_specialized_neighborhoods()

        if self.track_metrics:
            from evaluation.quality_metrics import QualityMetrics
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS v4 initialized for '{main_package}'")
        print(f"Primary archive: {archive_size}, Secondary: {secondary_archive_size}")
        print(f"MOBI/P neighbors: {mobi_p_neighbors}, Adaptive: {adaptive}")

    def load_all_data(self):
        """Load all required data matrices"""
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
        """Initialize semantic analysis components"""
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = 200
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000,
                                random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        main_cluster = self.cluster_labels[self.main_package_idx]
        self.cluster_members = np.where(self.cluster_labels == main_cluster)[0]
        print(f"Target package in cluster {main_cluster} with {len(self.cluster_members)} members")

    def compute_candidate_pools(self):
        """Compute candidate pools for efficient search"""
        # Co-occurrence candidates
        co_occurrences = self.rel_matrix[self.main_package_idx].toarray().flatten()
        top_cooccur_indices = np.argsort(co_occurrences)[::-1][1:301]  # Increased to 300
        self.cooccur_candidates = [idx for idx in top_cooccur_indices
                                   if co_occurrences[idx] > 0]

        # Semantic candidates
        similarities = self.sim_matrix[self.main_package_idx]
        top_similar_indices = np.argsort(similarities)[::-1][1:301]  # Increased to 300
        self.semantic_candidates = [idx for idx in top_similar_indices
                                    if similarities[idx] > 0.3]

        # Cluster candidates
        self.cluster_candidates = [idx for idx in self.cluster_members
                                   if idx != self.main_package_idx]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, "
              f"cluster={len(self.cluster_candidates)}")

    def generate_weight_vectors(self):
        """Generate weight vectors for decomposition-guided search"""
        # Uniform weight vectors like MOEA/D
        n_vectors = 30
        self.weight_vectors = []

        for i in range(n_vectors):
            weight = np.random.dirichlet(np.ones(3))
            self.weight_vectors.append(weight)

        # Add extreme weights for each objective
        self.weight_vectors.append([0.8, 0.1, 0.1])  # Focus LU
        self.weight_vectors.append([0.1, 0.8, 0.1])  # Focus SS
        self.weight_vectors.append([0.1, 0.1, 0.8])  # Focus RSS

    def normalize_objectives(self, objectives):
        """Normalize objectives to [0,1] range"""
        norm_obj = np.zeros_like(objectives)
        for i in range(len(objectives)):
            if self.obj_max[i] - self.obj_min[i] != 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / (self.obj_max[i] - self.obj_min[i])
        return np.clip(norm_obj, 0, 1)

    def update_bounds(self, objectives):
        """Update objective bounds dynamically"""
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

    def evaluate_objectives(self, chromosome):
        """Evaluate three objectives for a solution"""
        indices = np.where(chromosome == 1)[0]

        # Strict size constraints
        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([0, 0, float('inf')])

        # Calculate co-occurrence (LU)
        co_occurrences = self.rel_matrix[indices][:, indices].toarray()
        np.fill_diagonal(co_occurrences, 0)
        lu_score = np.sum(co_occurrences) / 2

        # Calculate semantic similarity (SS)
        if len(indices) > 1:
            pairwise_sim = self.sim_matrix[indices][:, indices]
            np.fill_diagonal(pairwise_sim, 0)
            n_pairs = len(indices) * (len(indices) - 1)
            avg_similarity = np.sum(pairwise_sim) / n_pairs if n_pairs > 0 else 0
            ss_score = avg_similarity
        else:
            ss_score = 0

        # Recommended set size (RSS)
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])
        self.update_bounds(objectives)

        return objectives

    def dominates(self, obj1, obj2):
        """Check dominance with normalized objectives"""
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def initialize_archives(self):
        """Initialize both primary and secondary archives"""
        print("Initializing archives with diverse strategies...")

        strategies = ['cooccur', 'semantic', 'cluster', 'mixed', 'random']
        strategy_weights = [0.3, 0.3, 0.2, 0.15, 0.05]

        attempts = 0
        max_attempts = 1000

        # Fill primary archive
        while len(self.archive) < self.archive_limit and attempts < max_attempts:
            strategy = np.random.choice(strategies, p=strategy_weights)
            solution = self.smart_initialization(strategy)
            objectives = self.evaluate_objectives(solution)

            if objectives[2] != float('inf'):
                if self.add_to_archive(solution, objectives, self.archive):
                    self.counter_archive_improvement += 1

            attempts += 1

        # Initialize secondary with variations
        for _ in range(min(self.secondary_limit, len(self.archive))):
            base_sol, base_obj = random.choice(self.archive)
            variation = self.create_variation(base_sol)
            var_obj = self.evaluate_objectives(variation)

            if var_obj[2] != float('inf'):
                self.secondary_archive.append((variation, var_obj))

        print(f"Archives initialized: Primary={len(self.archive)}, "
              f"Secondary={len(self.secondary_archive)}")

    def smart_initialization(self, strategy='mixed'):
        """Initialize solution using domain knowledge"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        # Adaptive target size
        if strategy in ['cooccur', 'semantic']:
            target_size = random.randint(3, 7)
        else:
            target_size = random.randint(self.min_size, min(10, self.max_size))

        if strategy == 'cooccur' and self.cooccur_candidates:
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            candidates = self.cooccur_candidates[:100]
            selected = np.random.choice(candidates, min(n_select, len(candidates)),
                                      replace=False)
            chromosome[selected] = 1

        elif strategy == 'semantic' and self.semantic_candidates:
            n_select = min(target_size - 1, len(self.semantic_candidates))
            candidates = self.semantic_candidates[:100]
            selected = np.random.choice(candidates, min(n_select, len(candidates)),
                                      replace=False)
            chromosome[selected] = 1

        elif strategy == 'cluster' and self.cluster_candidates:
            n_select = min(target_size - 1, len(self.cluster_candidates))
            selected = np.random.choice(self.cluster_candidates, n_select, replace=False)
            chromosome[selected] = 1

        elif strategy == 'mixed':
            # Mix of strategies
            n_cooccur = target_size // 3
            n_semantic = target_size // 3
            n_cluster = target_size - n_cooccur - n_semantic - 1

            if self.cooccur_candidates and n_cooccur > 0:
                selected = np.random.choice(self.cooccur_candidates[:50],
                                          min(n_cooccur, len(self.cooccur_candidates[:50])),
                                          replace=False)
                chromosome[selected] = 1

            if self.semantic_candidates and n_semantic > 0:
                selected = np.random.choice(self.semantic_candidates[:50],
                                          min(n_semantic, len(self.semantic_candidates[:50])),
                                          replace=False)
                chromosome[selected] = 1

            if self.cluster_candidates and n_cluster > 0:
                selected = np.random.choice(self.cluster_candidates,
                                          min(n_cluster, len(self.cluster_candidates)),
                                          replace=False)
                chromosome[selected] = 1
        else:
            # Random
            n_select = min(target_size - 1, self.n_packages - 1)
            candidates = [i for i in range(self.n_packages)
                         if i != self.main_package_idx]
            selected = np.random.choice(candidates, n_select, replace=False)
            chromosome[selected] = 1

        return chromosome

    def create_variation(self, solution):
        """Create variation of existing solution"""
        variation = solution.copy()
        indices = np.where(solution == 1)[0]

        # Random modification
        if random.random() < 0.5 and len(indices) > self.min_size:
            # Remove random package
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable:
                variation[random.choice(removable)] = 0
        else:
            # Add random package
            candidates = list(range(self.n_packages))
            valid = [c for c in candidates if variation[c] == 0]
            if valid:
                variation[random.choice(valid)] = 1

        return self.repair_solution(variation)

    def initialize_specialized_neighborhoods(self):
        """Initialize 6 specialized VNS neighborhoods"""

        def n1_greedy_lu(solution):
            """N1: Greedy addition for Linked Usage"""
            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            # Find package with highest co-occurrence to current set
            best_score = -1
            best_idx = -1

            for candidate in self.cooccur_candidates[:100]:
                if candidate not in indices and len(indices) < self.max_size - 1:
                    # Calculate total co-occurrence with current packages
                    score = sum(self.rel_matrix[candidate, idx] for idx in indices)
                    if score > best_score:
                        best_score = score
                        best_idx = candidate

            if best_idx >= 0:
                new_solution[best_idx] = 1

            return new_solution

        def n2_greedy_ss(solution):
            """N2: Greedy addition for Semantic Similarity"""
            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            # Find most semantically similar package
            best_score = -1
            best_idx = -1

            for candidate in self.semantic_candidates[:100]:
                if candidate not in indices and len(indices) < self.max_size - 1:
                    # Calculate average similarity with current packages
                    score = np.mean([self.sim_matrix[candidate, idx] for idx in indices])
                    if score > best_score:
                        best_score = score
                        best_idx = candidate

            if best_idx >= 0:
                new_solution[best_idx] = 1

            return new_solution

        def n3_size_optimization(solution):
            """N3: Optimize size by removing weak contributors"""
            indices = np.where(solution == 1)[0]
            new_solution = solution.copy()

            if len(indices) > self.ideal_size:
                # Calculate contribution of each package
                contributions = []
                for idx in indices:
                    if idx != self.main_package_idx:
                        temp = solution.copy()
                        temp[idx] = 0
                        current_obj = self.evaluate_objectives(solution)
                        reduced_obj = self.evaluate_objectives(temp)

                        # Contribution = loss when removed (normalized)
                        norm_current = self.normalize_objectives(current_obj)
                        norm_reduced = self.normalize_objectives(reduced_obj)
                        contribution = np.sum(norm_current - norm_reduced)
                        contributions.append((idx, contribution))

                if contributions:
                    # Remove weakest contributor
                    contributions.sort(key=lambda x: x[1])
                    new_solution[contributions[0][0]] = 0

            return new_solution

        def n4_quality_swap(solution):
            """N4: Swap for quality improvement"""
            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            if len(indices) > self.min_size:
                # Remove random package
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    to_remove = random.choice(removable)
                    new_solution[to_remove] = 0

                    # Add better package based on current objective weakness
                    current_obj = self.evaluate_objectives(solution)
                    norm_obj = self.normalize_objectives(current_obj)

                    # Focus on weakest objective
                    if norm_obj[0] < norm_obj[1]:  # LU weaker
                        candidates = self.cooccur_candidates[:50]
                    else:  # SS weaker
                        candidates = self.semantic_candidates[:50]

                    valid = [c for c in candidates if c not in indices]
                    if valid:
                        new_solution[random.choice(valid)] = 1

            return new_solution

        def n5_path_relinking(solution):
            """N5: Path relinking towards archive solution"""
            if len(self.archive) < 10:
                return n1_greedy_lu(solution)  # Fallback

            # Select target from top archive solutions
            target_sol, target_obj = random.choice(self.archive[:20])

            current_indices = set(np.where(solution == 1)[0])
            target_indices = set(np.where(target_sol == 1)[0])

            # Move one step towards target
            only_in_target = target_indices - current_indices
            only_in_current = current_indices - target_indices

            new_solution = solution.copy()

            # Add from target with probability 0.7
            if only_in_target and random.random() < 0.7:
                new_solution[random.choice(list(only_in_target))] = 1

            # Remove from current with probability 0.3
            if only_in_current and random.random() < 0.3:
                removable = [idx for idx in only_in_current
                           if idx != self.main_package_idx]
                if removable:
                    new_solution[random.choice(removable)] = 0

            return new_solution

        def n6_decomposition_guided(solution):
            """N6: Move guided by weight vector (like MOEA/D)"""
            # Select random weight vector
            weight = random.choice(self.weight_vectors)

            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]

            # Make move based on weight emphasis
            if weight[0] > 0.5:  # Emphasize LU
                # Add high co-occurrence package
                candidates = [c for c in self.cooccur_candidates[:100]
                            if new_solution[c] == 0]
                if candidates and len(indices) < self.max_size:
                    new_solution[random.choice(candidates)] = 1

            elif weight[1] > 0.3:  # Emphasize SS
                # Add semantically similar package
                candidates = [c for c in self.semantic_candidates[:100]
                            if new_solution[c] == 0]
                if candidates and len(indices) < self.max_size:
                    new_solution[random.choice(candidates)] = 1

            else:  # Emphasize RSS (smaller size)
                if len(indices) > self.ideal_size:
                    removable = [idx for idx in indices
                               if idx != self.main_package_idx]
                    if removable:
                        new_solution[random.choice(removable)] = 0

            return new_solution

        self.neighborhoods = [
            n1_greedy_lu,
            n2_greedy_ss,
            n3_size_optimization,
            n4_quality_swap,
            n5_path_relinking,
            n6_decomposition_guided
        ]

    def repair_solution(self, solution):
        """Repair solution to ensure feasibility"""
        indices = np.where(solution == 1)[0]

        # Ensure main package is included
        if self.main_package_idx not in indices:
            solution[self.main_package_idx] = 1
            indices = np.where(solution == 1)[0]

        # Enforce size constraints
        if len(indices) > self.max_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            n_remove = len(indices) - self.max_size
            to_remove = random.sample(removable, min(n_remove, len(removable)))
            for idx in to_remove:
                solution[idx] = 0

        return solution

    def adaptive_shaking(self, solution, neighborhood, base_intensity, no_improvement):
        """Adaptive shaking with intensity based on stagnation"""
        if self.adaptive:
            # Increase intensity with stagnation
            if no_improvement > 5:
                intensity = min(self.shaking_max,
                              base_intensity * (1.5 ** (no_improvement / 5)))
            else:
                intensity = base_intensity
        else:
            intensity = base_intensity

        intensity = int(intensity)
        shaken = solution.copy()

        # Apply neighborhood multiple times
        for _ in range(intensity):
            shaken = neighborhood(shaken)
            shaken = self.repair_solution(shaken)

        return shaken

    def enhanced_mobi_p(self, solution):
        """Enhanced MOBI/P with more neighbors tested"""
        non_dominated = []
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)

        # Test many neighbors
        tested = 0
        max_to_test = self.mobi_p_neighbors

        # Test all neighborhoods multiple times
        for _ in range(max_to_test // self.n_neighborhoods + 1):
            for neighborhood in self.neighborhoods:
                if tested >= max_to_test:
                    break

                neighbor = neighborhood(solution)
                neighbor = self.repair_solution(neighbor)
                neighbor_obj = self.evaluate_objectives(neighbor)
                tested += 1

                if neighbor_obj[2] != float('inf'):
                    if self.dominates(neighbor_obj, best_objectives):
                        best_solution = neighbor
                        best_objectives = neighbor_obj
                        non_dominated = [(neighbor, neighbor_obj)]
                    elif not self.dominates(best_objectives, neighbor_obj):
                        # Check against current non-dominated
                        is_dominated = False
                        to_remove = []

                        for i, (sol, obj) in enumerate(non_dominated):
                            if self.dominates(obj, neighbor_obj):
                                is_dominated = True
                                break
                            elif self.dominates(neighbor_obj, obj):
                                to_remove.append(i)

                        if not is_dominated:
                            for i in reversed(to_remove):
                                non_dominated.pop(i)
                            non_dominated.append((neighbor, neighbor_obj))

        return non_dominated[:self.mobi_p_neighbors]

    def add_to_archive(self, solution, objectives, archive):
        """Add solution to archive if non-dominated"""
        is_dominated = False
        to_remove = []

        for i, (arch_sol, arch_obj) in enumerate(archive):
            if self.dominates(arch_obj, objectives):
                is_dominated = True
                break
            elif self.dominates(objectives, arch_obj):
                to_remove.append(i)

        if not is_dominated:
            for i in reversed(to_remove):
                archive.pop(i)
            archive.append((solution, objectives))
            return True

        return False

    def update_archives(self, solutions):
        """Update both primary and secondary archives"""
        improvement = False

        for solution, objectives in solutions:
            # Try to add to primary archive
            if self.add_to_archive(solution, objectives, self.archive):
                improvement = True
                self.counter_archive_improvement += 1
            else:
                # If not added to primary, consider for secondary
                if len(self.secondary_archive) < self.secondary_limit:
                    self.secondary_archive.append((solution, objectives))
                elif random.random() < 0.3:  # 30% chance to replace
                    idx = random.randint(0, len(self.secondary_archive) - 1)
                    self.secondary_archive[idx] = (solution, objectives)

        # Truncate primary archive if needed
        if len(self.archive) > self.archive_limit:
            self.truncate_archive_with_crowding()

        return improvement

    def truncate_archive_with_crowding(self):
        """Truncate archive maintaining diversity"""
        objectives_array = np.array([obj for _, obj in self.archive])
        n_solutions = len(self.archive)

        # Calculate crowding distance
        crowding_distances = np.zeros(n_solutions)
        n_objectives = objectives_array.shape[1]

        for m in range(n_objectives):
            sorted_indices = np.argsort(objectives_array[:, m])
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            obj_range = objectives_array[sorted_indices[-1], m] - objectives_array[sorted_indices[0], m]

            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (objectives_array[sorted_indices[i + 1], m] -
                              objectives_array[sorted_indices[i - 1], m]) / obj_range
                    crowding_distances[sorted_indices[i]] += distance

        # Keep most diverse solutions
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def inject_from_secondary(self):
        """Inject solutions from secondary archive (like MOEA/D)"""
        if len(self.secondary_archive) > 0 and len(self.archive) > 0:
            # Select random solutions from secondary
            n_inject = min(3, len(self.secondary_archive))
            to_inject = random.sample(self.secondary_archive, n_inject)

            for sol, obj in to_inject:
                # Try to replace worst solution in archive
                if len(self.archive) >= self.archive_limit:
                    # Find solution with lowest crowding distance
                    objectives_array = np.array([o for _, o in self.archive])
                    worst_idx = random.randint(self.archive_limit // 2, len(self.archive) - 1)
                    self.archive[worst_idx] = (sol, obj)

    def select_solution_biased(self):
        """Select solution with bias towards unexplored"""
        unexplored = []

        for i, (sol, obj) in enumerate(self.archive):
            sol_tuple = tuple(sol)
            if sol_tuple not in self.explored_solutions:
                unexplored.append((i, sol, obj))

        if unexplored:
            # Select from unexplored
            idx, solution, _ = random.choice(unexplored)
            self.explored_solutions.add(tuple(solution))
            return solution

        # If all explored, select based on neighborhood success
        if self.archive and self.memory:
            # Bias towards solutions that led to improvements
            weights = [1.0] * len(self.archive)
            for i in range(min(20, len(self.archive))):
                weights[i] = 2.0  # Double weight for top solutions

            weights = np.array(weights) / np.sum(weights)
            idx = np.random.choice(len(self.archive), p=weights)
            return self.archive[idx][0]

        # Default random selection
        if self.archive:
            return random.choice(self.archive)[0]

        return self.smart_initialization('mixed')

    def update_memory(self, neighborhood_idx, improvement):
        """Update memory with successful moves"""
        self.memory.append((neighborhood_idx, improvement))
        self.neighborhood_success[neighborhood_idx] += improvement

    def select_neighborhood_adaptive(self):
        """Select neighborhood based on past success"""
        if not self.adaptive or sum(self.neighborhood_success) == 0:
            return 0  # Start with first neighborhood

        # Calculate probabilities based on success
        total_success = sum(self.neighborhood_success)
        probabilities = [s / total_success for s in self.neighborhood_success]

        # Add exploration factor
        min_prob = 0.05
        probabilities = [max(p, min_prob) for p in probabilities]
        probabilities = np.array(probabilities) / np.sum(probabilities)

        return np.random.choice(self.n_neighborhoods, p=probabilities)

    def calculate_metrics(self):
        """Calculate quality metrics"""
        if not self.track_metrics or len(self.archive) < 3:
            return None

        objectives = np.array([obj for _, obj in self.archive])

        # Normalize for hypervolume
        normalized_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(normalized_objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(normalized_objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(normalized_objectives)

        return metrics

    def run(self):
        """Main MOVNS v4 loop with all enhancements"""
        print(f"\nStarting MOVNS v4 for {self.main_package}...")
        print("="*60)

        MIN_ITERATIONS = 20
        IMPROVEMENT_THRESHOLD = 0.001
        no_improvement = 0
        best_hv = 0

        for iteration in range(self.max_iterations):
            # Select solution with bias
            current_solution = self.select_solution_biased()

            # Adaptive neighborhood selection
            if self.adaptive:
                k = self.select_neighborhood_adaptive()
            else:
                k = 0

            improvements_this_iter = 0
            neighborhood_tries = 0

            while k < self.n_neighborhoods and neighborhood_tries < self.n_neighborhoods * 2:
                neighborhood_tries += 1

                # Adaptive shaking
                x_prime = self.adaptive_shaking(current_solution,
                                               self.neighborhoods[k],
                                               self.shaking_base,
                                               no_improvement)

                # Enhanced MOBI/P
                non_dominated = self.enhanced_mobi_p(x_prime)

                # Update archives
                if non_dominated:
                    improved = self.update_archives(non_dominated)

                    if improved:
                        improvements_this_iter += len(non_dominated)
                        self.update_memory(k, len(non_dominated))
                        current_solution = non_dominated[0][0]

                        if self.adaptive:
                            k = self.select_neighborhood_adaptive()
                        else:
                            k = 0  # Reset to first neighborhood
                    else:
                        k += 1
                else:
                    k += 1

            self.iteration_improvements.append(improvements_this_iter)

            # Archive injection (like MOEA/D)
            if iteration % 5 == 0 and iteration > 0:
                self.inject_from_secondary()

            # Print progress
            if iteration % 10 == 0:
                best_obj = min(self.archive, key=lambda x: x[1][0])[1]
                print(f"Iteration {iteration}: Archive size={len(self.archive)}, "
                      f"Secondary={len(self.secondary_archive)}")
                print(f"  Best: LU={-best_obj[0]:.1f}, SS={-best_obj[1]:.4f}, "
                      f"RSS={best_obj[2]:.1f}")
                print(f"  Improvements: {self.counter_archive_improvement}, "
                      f"This iter: {improvements_this_iter}")

                if self.track_metrics:
                    metrics = self.calculate_metrics()
                    if metrics:
                        for key in self.metrics_history:
                            if key in metrics:
                                self.metrics_history[key].append(metrics[key])

                        current_hv = metrics.get('hypervolume', 0)
                        print(f"  HV={current_hv:.4f}, No-improvement={no_improvement}")

                        if iteration >= MIN_ITERATIONS:
                            if current_hv > best_hv * (1 + IMPROVEMENT_THRESHOLD):
                                best_hv = current_hv
                                no_improvement = 0
                            else:
                                no_improvement += 1

                            if no_improvement >= self.min_no_improvement:
                                print(f"\nStopping: No improvement for "
                                     f"{self.min_no_improvement} iterations")
                                break

            # Adaptive parameter adjustment
            if self.adaptive and iteration % 10 == 0:
                recent_improvements = self.iteration_improvements[-10:]
                if sum(recent_improvements) < 5:
                    # Increase exploration
                    self.mobi_p_neighbors = min(70, int(self.mobi_p_neighbors * 1.2))
                    self.shaking_base = min(self.shaking_max, self.shaking_base + 1)
                else:
                    # Decrease exploration
                    self.mobi_p_neighbors = max(30, int(self.mobi_p_neighbors * 0.9))
                    self.shaking_base = max(1, self.shaking_base - 0.5)

        print(f"\nFinal archive: {len(self.archive)} solutions")
        print(f"Secondary archive: {len(self.secondary_archive)} solutions")
        print(f"Total archive improvements: {self.counter_archive_improvement}")

        # Return final Pareto front
        final_solutions = []
        for solution, objectives in self.archive:
            indices = np.where(solution == 1)[0]
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
        """Return metrics history for analysis"""
        return self.metrics_history if self.track_metrics else None