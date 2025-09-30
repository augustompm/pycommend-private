"""
MOVNS v3 - Enhanced Variable Neighborhood Search
Incorporates lessons from MOEA/D analysis to surpass its performance
"""

import numpy as np
import random
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity

class MOVNS_V3:
    def __init__(self, main_package, archive_size=100, max_iterations=50,
                 k_max=4, track_metrics=False, min_no_improvement=10):
        """
        MOVNS v3 with enhanced neighborhoods
        Archive size increased to 100 to match MOEA/D
        """
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.k_max = k_max
        self.track_metrics = track_metrics
        self.min_no_improvement = min_no_improvement

        self.archive = []
        self.explored_solutions = set()
        self.counter_archive_improvement = 0
        self.iteration_improvements = []

        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        self.min_size = 2
        self.max_size = 25
        self.ideal_size = 5

        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.initialize_archive()
        self.initialize_enhanced_neighborhoods()

        if self.track_metrics:
            from evaluation.quality_metrics import QualityMetrics
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS v3 initialized for '{main_package}'")
        print(f"Archive size: {archive_size}, Max iterations: {max_iterations}")
        print(f"Enhanced neighborhoods with objective guidance")

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

        # Calculate threshold from non-zero values
        if hasattr(self.rel_matrix, 'toarray'):
            # Sparse matrix
            matrix_dense = self.rel_matrix.toarray()
        else:
            # Already dense
            matrix_dense = np.array(self.rel_matrix)

        non_zero_values = matrix_dense[matrix_dense > 0]
        if len(non_zero_values) > 0:
            self.co_occurrence_threshold = np.percentile(non_zero_values, 75)
        else:
            self.co_occurrence_threshold = 0

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
        co_occurrences = self.rel_matrix[self.main_package_idx].toarray().flatten()
        top_cooccur_indices = np.argsort(co_occurrences)[::-1][1:201]
        self.cooccur_candidates = [idx for idx in top_cooccur_indices
                                   if co_occurrences[idx] > 0]

        similarities = self.sim_matrix[self.main_package_idx]
        top_similar_indices = np.argsort(similarities)[::-1][1:201]
        self.semantic_candidates = [idx for idx in top_similar_indices
                                    if similarities[idx] > 0.3]

        self.cluster_candidates = [idx for idx in self.cluster_members
                                   if idx != self.main_package_idx]

        print(f"Candidate pools: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, "
              f"cluster={len(self.cluster_candidates)}")

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
        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([0, 0, float('inf')])

        co_occurrences = self.rel_matrix[indices][:, indices].toarray()
        np.fill_diagonal(co_occurrences, 0)
        lu_score = np.sum(co_occurrences) / 2

        if len(indices) > 1:
            pairwise_sim = self.sim_matrix[indices][:, indices]
            np.fill_diagonal(pairwise_sim, 0)
            n_pairs = len(indices) * (len(indices) - 1)
            avg_similarity = np.sum(pairwise_sim) / n_pairs if n_pairs > 0 else 0
            ss_score = avg_similarity
        else:
            ss_score = 0

        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])
        self.update_bounds(objectives)
        return objectives

    def dominates(self, obj1, obj2):
        """Check dominance with normalized objectives"""
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        print("Initializing archive with diverse strategies...")

        strategies = ['cooccur', 'semantic', 'cluster', 'random']
        strategy_weights = [0.35, 0.35, 0.2, 0.1]

        attempts = 0
        max_attempts = 500

        while len(self.archive) < self.archive_limit and attempts < max_attempts:
            strategy = np.random.choice(strategies, p=strategy_weights)
            solution = self.smart_initialization(strategy)
            objectives = self.evaluate_objectives(solution)

            if objectives[2] != float('inf'):
                is_dominated = False
                to_remove = []

                for i, (sol, obj) in enumerate(self.archive):
                    if self.dominates(obj, objectives):
                        is_dominated = True
                        break
                    elif self.dominates(objectives, obj):
                        to_remove.append(i)

                if not is_dominated:
                    for i in reversed(to_remove):
                        self.archive.pop(i)
                    self.archive.append((solution, objectives))

            attempts += 1

        print(f"Archive initialized with {len(self.archive)} solutions")

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solution using domain knowledge"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(self.min_size, min(10, self.max_size))

        if strategy == 'cooccur' and len(self.cooccur_candidates) > 0:
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            selected = np.random.choice(self.cooccur_candidates[:50],
                                      n_select, replace=False)
            chromosome[selected] = 1

        elif strategy == 'semantic' and len(self.semantic_candidates) > 0:
            n_select = min(target_size - 1, len(self.semantic_candidates))
            selected = np.random.choice(self.semantic_candidates[:50],
                                      n_select, replace=False)
            chromosome[selected] = 1

        elif strategy == 'cluster' and len(self.cluster_candidates) > 0:
            n_select = min(target_size - 1, len(self.cluster_candidates))
            selected = np.random.choice(self.cluster_candidates,
                                      n_select, replace=False)
            chromosome[selected] = 1

        else:
            n_select = min(target_size - 1, self.n_packages - 1)
            candidates = [i for i in range(self.n_packages)
                         if i != self.main_package_idx]
            selected = np.random.choice(candidates, n_select, replace=False)
            chromosome[selected] = 1

        return chromosome

    def initialize_enhanced_neighborhoods(self):
        """Initialize enhanced VNS neighborhoods"""

        def n1_objective_guided(solution):
            """N1: Single change guided by weakest objective"""
            objectives = self.evaluate_objectives(solution)
            norm_obj = self.normalize_objectives(objectives)
            indices = np.where(solution == 1)[0]

            # Identify weakest objective (excluding RSS)
            lu_norm = norm_obj[0]
            ss_norm = norm_obj[1]

            new_solution = solution.copy()

            if lu_norm < ss_norm:
                # LU is weaker - add high co-occurrence package
                candidates = [c for c in self.cooccur_candidates[:20]
                            if solution[c] == 0]
                if candidates:
                    new_solution[random.choice(candidates)] = 1
            else:
                # SS is weaker - add semantically similar package
                candidates = [c for c in self.semantic_candidates[:20]
                            if solution[c] == 0]
                if candidates:
                    new_solution[random.choice(candidates)] = 1

            # Balance size if needed
            if np.sum(new_solution) > self.ideal_size + 3:
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    # Remove package with lowest contribution
                    contributions = []
                    for idx in removable:
                        temp = new_solution.copy()
                        temp[idx] = 0
                        temp_obj = self.evaluate_objectives(temp)
                        contribution = np.sum(objectives - temp_obj)
                        contributions.append((idx, contribution))

                    contributions.sort(key=lambda x: x[1])
                    new_solution[contributions[0][0]] = 0

            return new_solution

        def n2_weight_directed(solution):
            """N2: Move in specific weight direction (like MOEA/D)"""
            # Generate random weight vector
            weight = np.random.dirichlet(np.ones(3))

            new_solution = solution.copy()
            indices = np.where(solution == 1)[0]

            # Emphasize objective based on weight
            if weight[0] > 0.5:  # Focus on LU
                candidates = [c for c in self.cooccur_candidates[:30]
                            if solution[c] == 0]
                if candidates and random.random() < 0.7:
                    new_solution[random.choice(candidates)] = 1
            elif weight[1] > 0.3:  # Focus on SS
                candidates = [c for c in self.semantic_candidates[:30]
                            if solution[c] == 0]
                if candidates and random.random() < 0.7:
                    new_solution[random.choice(candidates)] = 1

            # Size control based on weight[2] (RSS)
            current_size = np.sum(new_solution)
            if weight[2] > 0.3 and current_size > self.ideal_size:
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    new_solution[random.choice(removable)] = 0

            return new_solution

        def n3_archive_crossover(solution):
            """N3: Learn from successful archive solutions"""
            if len(self.archive) < 5:
                return n1_objective_guided(solution)

            # Select promising solution from archive
            archive_sol, archive_obj = random.choice(self.archive[:20])  # Top 20

            new_solution = solution.copy()
            current_indices = set(np.where(solution == 1)[0])
            archive_indices = set(np.where(archive_sol == 1)[0])

            # Find differences
            only_in_archive = archive_indices - current_indices
            only_in_current = current_indices - archive_indices

            # Apply subset of beneficial changes
            if only_in_archive:
                n_add = min(2, len(only_in_archive))
                to_add = random.sample(list(only_in_archive), n_add)
                for idx in to_add:
                    new_solution[idx] = 1

            # Remove some non-beneficial packages
            if only_in_current and len(current_indices) > self.ideal_size:
                removable = [idx for idx in only_in_current
                           if idx != self.main_package_idx]
                if removable:
                    n_remove = min(1, len(removable))
                    to_remove = random.sample(removable, n_remove)
                    for idx in to_remove:
                        new_solution[idx] = 0

            return new_solution

        def n4_decomposition_hybrid(solution):
            """N4: Explore multiple directions simultaneously"""
            # Generate multiple weight vectors
            n_directions = 3
            candidates = []

            for _ in range(n_directions):
                weight = np.random.dirichlet(np.ones(3))
                temp_solution = solution.copy()

                # Make move based on weight emphasis
                if weight[0] > weight[1]:
                    # Emphasize LU
                    cooccur_valid = [c for c in self.cooccur_candidates[:40]
                                   if temp_solution[c] == 0]
                    if cooccur_valid:
                        temp_solution[random.choice(cooccur_valid)] = 1
                else:
                    # Emphasize SS
                    semantic_valid = [c for c in self.semantic_candidates[:40]
                                    if temp_solution[c] == 0]
                    if semantic_valid:
                        temp_solution[random.choice(semantic_valid)] = 1

                # Evaluate and store
                temp_obj = self.evaluate_objectives(temp_solution)
                candidates.append((temp_solution, temp_obj, weight))

            # Select best candidate based on weighted sum
            best_score = float('inf')
            best_solution = solution

            for cand_sol, cand_obj, weight in candidates:
                norm_obj = self.normalize_objectives(cand_obj)
                score = np.sum(weight * norm_obj)
                if score < best_score:
                    best_score = score
                    best_solution = cand_sol

            return best_solution

        self.neighborhoods = [
            n1_objective_guided,
            n2_weight_directed,
            n3_archive_crossover,
            n4_decomposition_hybrid
        ]

    def repair_solution(self, solution):
        """Repair solution to ensure feasibility"""
        indices = np.where(solution == 1)[0]

        if self.main_package_idx not in indices:
            solution[self.main_package_idx] = 1

        if len(indices) > self.max_size:
            removable = [idx for idx in indices if idx != self.main_package_idx]
            n_remove = len(indices) - self.max_size
            to_remove = random.sample(removable, min(n_remove, len(removable)))
            for idx in to_remove:
                solution[idx] = 0

        return solution

    def shake(self, solution, neighborhood, intensity=1):
        """Shaking phase with adaptive intensity"""
        shaken = solution.copy()

        # Adaptive intensity based on recent improvements
        if len(self.iteration_improvements) > 5:
            recent_improvements = self.iteration_improvements[-5:]
            if sum(recent_improvements) == 0:
                intensity = min(3, intensity + 1)  # Increase if stagnant

        for _ in range(intensity):
            shaken = neighborhood(shaken)
            shaken = self.repair_solution(shaken)

        return shaken

    def mobi_p_local_search(self, solution, max_neighbors=15):
        """Enhanced MOBI/P with archive injection"""
        non_dominated = []
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)

        # Test all neighborhoods
        for neighborhood in self.neighborhoods:
            neighbor = neighborhood(solution)
            neighbor = self.repair_solution(neighbor)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if neighbor_obj[2] != float('inf'):
                if self.dominates(neighbor_obj, best_objectives):
                    best_solution = neighbor
                    best_objectives = neighbor_obj
                    non_dominated = [(neighbor, neighbor_obj)]
                elif not self.dominates(best_objectives, neighbor_obj):
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

        # Archive injection (like MOEA/D)
        if len(self.archive) > 20 and random.random() < 0.2:
            # Inject good solution from archive
            archive_sol, archive_obj = random.choice(self.archive[:10])
            if not self.dominates(best_objectives, archive_obj):
                non_dominated.append((archive_sol, archive_obj))

        return non_dominated[:max_neighbors]

    def update_archive(self, solutions):
        """Update archive with new solutions"""
        improvement = False

        for solution, objectives in solutions:
            is_dominated = False
            to_remove = []

            for i, (arch_sol, arch_obj) in enumerate(self.archive):
                if self.dominates(arch_obj, objectives):
                    is_dominated = True
                    break
                elif self.dominates(objectives, arch_obj):
                    to_remove.append(i)

            if not is_dominated:
                for i in reversed(to_remove):
                    self.archive.pop(i)

                self.archive.append((solution, objectives))
                improvement = True
                self.counter_archive_improvement += 1

        # Truncate if exceeds limit using crowding distance
        if len(self.archive) > self.archive_limit:
            self.truncate_archive_with_crowding()

        return improvement

    def truncate_archive_with_crowding(self):
        """Truncate archive maintaining diversity"""
        objectives_array = np.array([obj for _, obj in self.archive])
        n_solutions = len(self.archive)

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

        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def select_unexplored_solution(self):
        """Select solution from archive for exploration"""
        unexplored = []
        for i, (sol, obj) in enumerate(self.archive):
            sol_tuple = tuple(sol)
            if sol_tuple not in self.explored_solutions:
                unexplored.append((i, sol, obj))

        if unexplored:
            idx, solution, _ = random.choice(unexplored)
            self.explored_solutions.add(tuple(solution))
            return solution

        # If all explored, select random
        if self.archive:
            solution, _ = random.choice(self.archive)
            self.explored_solutions.add(tuple(solution))
            return solution

        return self.smart_initialization('hybrid')

    def calculate_metrics(self):
        """Calculate quality metrics"""
        if not self.track_metrics or len(self.archive) < 3:
            return None

        objectives = np.array([sol[1] for sol in self.archive])

        # Normalize for hypervolume calculation
        normalized_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(normalized_objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(normalized_objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(normalized_objectives)

        return metrics

    def run(self):
        """Main MOVNS loop with enhancements"""
        print(f"\nStarting MOVNS v3 for {self.main_package}...")
        print("="*60)

        MIN_ITERATIONS = 15
        IMPROVEMENT_THRESHOLD = 0.001
        no_improvement = 0
        best_hv = 0

        for iteration in range(self.max_iterations):
            current_solution = self.select_unexplored_solution()

            k = 0
            improvements_this_iter = 0

            while k < self.k_max:
                # Shaking with adaptive intensity
                x_prime = self.shake(current_solution, self.neighborhoods[k],
                                   intensity=1 + (k // 2))

                # Local search
                non_dominated = self.mobi_p_local_search(x_prime)

                # Update archive
                if non_dominated:
                    improved = self.update_archive(non_dominated)

                    if improved:
                        improvements_this_iter += 1
                        current_solution = non_dominated[0][0]
                        k = 0  # Reset to first neighborhood
                    else:
                        k += 1  # Move to next neighborhood
                else:
                    k += 1

            self.iteration_improvements.append(improvements_this_iter)

            # Print progress
            if iteration % 5 == 0:
                best_obj = min(self.archive, key=lambda x: x[1][0])[1]
                print(f"Iteration {iteration}: Archive size={len(self.archive)}, "
                      f"Improvements={self.counter_archive_improvement}")
                print(f"  Best: LU={-best_obj[0]:.1f}, SS={-best_obj[1]:.4f}, "
                      f"RSS={best_obj[2]:.1f}")

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

        print(f"\nFinal archive: {len(self.archive)} solutions")
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