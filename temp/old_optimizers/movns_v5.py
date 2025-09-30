"""
MOVNS v5 - Decomposition-Guided Variable Neighborhood Search
Hybrid approach combining MOEA/D decomposition with VNS intensification
Based on MOVNS v2 but with decomposition-based neighborhoods
"""

import numpy as np
import pickle
import random
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics


class MOVNS_V5:
    """
    MOVNS v5 - Decomposition-Guided VNS

    Key innovations:
    1. Each neighborhood represents a decomposition direction
    2. Weight vectors guide local search (like MOEA/D)
    3. Adaptive weight selection based on archive gaps
    4. Simplified from v3/v4, based on successful v2
    """

    def __init__(self, main_package, archive_size=100, max_iterations=50,
                 n_weight_vectors=30, track_metrics=False, min_no_improvement=10):
        self.main_package = main_package
        self.archive_limit = archive_size
        self.max_iterations = max_iterations
        self.n_weight_vectors = n_weight_vectors
        self.track_metrics = track_metrics
        self.min_no_improvement = min_no_improvement

        # Problem parameters
        self.n_objectives = 3
        self.min_size = 2
        self.max_size = 10  # Limit to avoid LU explosion
        self.ideal_size = 5

        # Objective bounds for normalization
        self.obj_min = np.array([-10000.0, -1.0, 2.0])
        self.obj_max = np.array([0.0, 0.0, 15.0])

        # Archive and tracking
        self.archive = []
        self.explored_solutions = set()
        self.counter_archive_improvement = 0

        # Ideal point (best value for each objective)
        self.z_star = np.array([0.0, 0.0, 1.0])  # Will be updated

        # Load data and initialize
        self.load_all_data()
        self.initialize_semantic_components()
        self.compute_candidate_pools()
        self.generate_weight_vectors()
        self.initialize_archive()
        self.initialize_decomposition_neighborhoods()

        if self.track_metrics:
            self.metrics_calculator = QualityMetrics()
            self.metrics_history = {
                'hypervolume': [],
                'spacing': [],
                'diversity': []
            }

        print(f"MOVNS v5 initialized for '{main_package}'")
        print(f"Archive size: {archive_size}, Weight vectors: {n_weight_vectors}")
        print(f"Using decomposition-guided neighborhoods")

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
            raise ValueError(f"Package '{self.main_package}' not found")
        self.main_package_idx = self.package_names.index(self.main_package)

        print(f"Data loaded: {self.n_packages} packages")

    def initialize_semantic_components(self):
        """Initialize semantic analysis components"""
        n_clusters = 200
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        self.cluster_labels = kmeans.fit_predict(self.embeddings)
        self.cluster_centers = kmeans.cluster_centers_

        main_cluster = self.cluster_labels[self.main_package_idx]
        self.cluster_members = np.where(self.cluster_labels == main_cluster)[0]
        print(f"Target package in cluster {main_cluster} with {len(self.cluster_members)} members")

    def compute_candidate_pools(self):
        """Pre-compute candidate pools for efficiency"""
        # Co-occurrence candidates
        co_occurrences = self.rel_matrix[self.main_package_idx].toarray().flatten()
        top_cooccur_indices = np.argsort(co_occurrences)[::-1][1:201]
        self.cooccur_candidates = [idx for idx in top_cooccur_indices
                                  if co_occurrences[idx] > 0]

        # Semantic candidates
        similarities = self.sim_matrix[self.main_package_idx]
        top_similar_indices = np.argsort(similarities)[::-1][1:201]
        self.semantic_candidates = [idx for idx in top_similar_indices
                                   if similarities[idx] > 0.3]

        # Cluster candidates
        self.cluster_candidates = [idx for idx in self.cluster_members
                                  if idx != self.main_package_idx]

        print(f"Candidate pools ready: cooccur={len(self.cooccur_candidates)}, "
              f"semantic={len(self.semantic_candidates)}, cluster={len(self.cluster_candidates)}")

    def generate_weight_vectors(self):
        """Generate uniformly distributed weight vectors"""
        self.weight_vectors = []

        # Uniform distribution using Das and Dennis method (simplified)
        for i in range(self.n_weight_vectors):
            weight = np.random.dirichlet(np.ones(self.n_objectives))
            self.weight_vectors.append(weight)

        # Add extreme weights for each objective
        self.weight_vectors.append(np.array([0.8, 0.1, 0.1]))  # LU focus
        self.weight_vectors.append(np.array([0.1, 0.8, 0.1]))  # SS focus
        self.weight_vectors.append(np.array([0.1, 0.1, 0.8]))  # RSS focus

        # Add balanced weight
        self.weight_vectors.append(np.array([0.33, 0.33, 0.34]))

        self.weight_vectors = np.array(self.weight_vectors)
        print(f"Generated {len(self.weight_vectors)} weight vectors")

    def normalize_objectives(self, objectives):
        """Normalize objectives to [0,1] range"""
        norm_obj = np.zeros_like(objectives)
        for i in range(len(objectives)):
            if self.obj_max[i] - self.obj_min[i] != 0:
                norm_obj[i] = (objectives[i] - self.obj_min[i]) / (self.obj_max[i] - self.obj_min[i])
        return np.clip(norm_obj, 0, 1)

    def update_bounds(self, objectives):
        """Update objective bounds and ideal point"""
        self.obj_min = np.minimum(self.obj_min, objectives)
        self.obj_max = np.maximum(self.obj_max, objectives)

        # Update ideal point (best seen for each objective)
        self.z_star = np.minimum(self.z_star, self.normalize_objectives(objectives))

    def evaluate_objectives(self, chromosome):
        """Evaluate the three objectives"""
        indices = np.where(chromosome == 1)[0]

        if len(indices) < self.min_size or len(indices) > self.max_size:
            return np.array([0, 0, float('inf')])

        # LU: Linked Usage (co-occurrence)
        co_occurrences = self.rel_matrix[indices][:, indices].toarray()
        np.fill_diagonal(co_occurrences, 0)
        lu_score = np.sum(co_occurrences) / 2

        # SS: Semantic Similarity
        if len(indices) > 1:
            pairwise_sim = self.sim_matrix[indices][:, indices]
            np.fill_diagonal(pairwise_sim, 0)
            n_pairs = len(indices) * (len(indices) - 1)
            ss_score = np.sum(pairwise_sim) / n_pairs if n_pairs > 0 else 0
        else:
            ss_score = 0

        # RSS: Recommended Set Size
        rss_score = len(indices)

        objectives = np.array([-lu_score, -ss_score, rss_score])
        self.update_bounds(objectives)

        return objectives

    def decompose(self, objectives, weight, method='tchebycheff'):
        """Decomposition function (like MOEA/D)"""
        norm_obj = self.normalize_objectives(objectives)

        if method == 'weighted_sum':
            return np.sum(weight * norm_obj)

        elif method == 'tchebycheff':
            # Tchebycheff approach
            return np.max(weight * np.abs(norm_obj - self.z_star))

        elif method == 'pbi':
            # Penalty-based boundary intersection
            d1 = np.abs(np.dot(norm_obj - self.z_star, weight)) / np.linalg.norm(weight)
            d2 = np.linalg.norm((norm_obj - self.z_star) - d1 * weight / np.linalg.norm(weight))
            theta = 5.0
            return d1 + theta * d2

        else:
            return np.sum(weight * norm_obj)

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (using normalized objectives)"""
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)
        return all(norm1 <= norm2) and any(norm1 < norm2)

    def initialize_decomposition_neighborhoods(self):
        """Create decomposition-based neighborhoods"""

        def n1_weighted_lu(solution):
            """N1: Maximize LU using weight [0.7, 0.2, 0.1]"""
            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            if len(indices) < self.max_size:
                # Add package with highest co-occurrence
                best_score = -float('inf')
                best_idx = -1

                for candidate in self.cooccur_candidates[:50]:
                    if candidate not in indices:
                        # Calculate co-occurrence gain
                        score = sum(self.rel_matrix[candidate, idx] for idx in indices)
                        if score > best_score:
                            best_score = score
                            best_idx = candidate

                if best_idx >= 0:
                    new_solution[best_idx] = 1

            return new_solution

        def n2_weighted_ss(solution):
            """N2: Maximize SS using weight [0.2, 0.7, 0.1]"""
            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            if len(indices) < self.max_size:
                # Add most semantically similar package
                best_score = -float('inf')
                best_idx = -1

                for candidate in self.semantic_candidates[:50]:
                    if candidate not in indices:
                        # Calculate average similarity
                        score = np.mean([self.sim_matrix[candidate, idx] for idx in indices])
                        if score > best_score:
                            best_score = score
                            best_idx = candidate

                if best_idx >= 0:
                    new_solution[best_idx] = 1

            return new_solution

        def n3_minimize_rss(solution):
            """N3: Minimize RSS while maintaining quality"""
            indices = list(np.where(solution == 1)[0])
            new_solution = solution.copy()

            if len(indices) > self.ideal_size:
                # Remove package with lowest contribution
                removable = [idx for idx in indices if idx != self.main_package_idx]

                if removable:
                    # Calculate contribution of each package
                    contributions = []
                    current_obj = self.evaluate_objectives(solution)

                    for idx in removable:
                        temp = solution.copy()
                        temp[idx] = 0
                        temp_obj = self.evaluate_objectives(temp)

                        # Contribution is the loss when removed
                        norm_current = self.normalize_objectives(current_obj)
                        norm_temp = self.normalize_objectives(temp_obj)
                        contribution = np.sum(norm_current[:2] - norm_temp[:2])  # Focus on LU and SS
                        contributions.append((idx, contribution))

                    # Remove lowest contributor
                    contributions.sort(key=lambda x: x[1])
                    new_solution[contributions[0][0]] = 0

            return new_solution

        def n4_tchebycheff_move(solution):
            """N4: Move guided by Tchebycheff decomposition"""
            # Select random weight
            weight = random.choice(self.weight_vectors)
            current_obj = self.evaluate_objectives(solution)
            current_score = self.decompose(current_obj, weight, 'tchebycheff')

            indices = set(np.where(solution == 1)[0])
            new_solution = solution.copy()

            # Try adding or removing to improve Tchebycheff score
            if random.random() < 0.6 and len(indices) < self.max_size:
                # Try adding
                best_score = current_score
                best_idx = -1

                candidates = (self.cooccur_candidates[:30] if weight[0] > 0.5
                            else self.semantic_candidates[:30])

                for candidate in candidates:
                    if candidate not in indices:
                        temp = solution.copy()
                        temp[candidate] = 1
                        temp_obj = self.evaluate_objectives(temp)
                        temp_score = self.decompose(temp_obj, weight, 'tchebycheff')

                        if temp_score < best_score:
                            best_score = temp_score
                            best_idx = candidate

                if best_idx >= 0:
                    new_solution[best_idx] = 1

            elif len(indices) > self.min_size:
                # Try removing
                removable = [idx for idx in indices if idx != self.main_package_idx]
                if removable:
                    idx_to_remove = random.choice(removable)
                    new_solution[idx_to_remove] = 0

            return new_solution

        def n5_balanced_move(solution):
            """N5: Balance all objectives using weight [0.33, 0.33, 0.34]"""
            current_obj = self.evaluate_objectives(solution)
            norm_obj = self.normalize_objectives(current_obj)

            # Identify weakest objective
            weakest = np.argmin(norm_obj[:2])  # Focus on LU and SS

            if weakest == 0:
                return n1_weighted_lu(solution)
            else:
                return n2_weighted_ss(solution)

        def n6_adaptive_decomposition(solution):
            """N6: Adaptive move based on archive gaps"""
            # Find least explored weight region
            if len(self.archive) > 10:
                # Calculate which weight vector is least represented
                archive_weights = []
                for sol, obj in self.archive:
                    # Find closest weight vector
                    norm_obj = self.normalize_objectives(obj)
                    distances = [self.decompose(obj, w, 'tchebycheff') for w in self.weight_vectors]
                    closest = np.argmin(distances)
                    archive_weights.append(closest)

                # Find least used weight
                weight_counts = np.zeros(len(self.weight_vectors))
                for w in archive_weights:
                    weight_counts[w] += 1

                least_used = np.argmin(weight_counts)
                weight = self.weight_vectors[least_used]
            else:
                weight = random.choice(self.weight_vectors)

            # Move in direction of selected weight
            return self.directional_move(solution, weight)

        # Store neighborhoods
        self.neighborhoods = [
            n1_weighted_lu,
            n2_weighted_ss,
            n3_minimize_rss,
            n4_tchebycheff_move,
            n5_balanced_move,
            n6_adaptive_decomposition
        ]

        self.n_neighborhoods = len(self.neighborhoods)
        print(f"Created {self.n_neighborhoods} decomposition-based neighborhoods")

    def directional_move(self, solution, weight):
        """Generic move in direction of weight vector"""
        indices = set(np.where(solution == 1)[0])
        new_solution = solution.copy()

        # Decide action based on weight
        if weight[2] > 0.5 and len(indices) > self.ideal_size:
            # High RSS weight - remove packages
            removable = [idx for idx in indices if idx != self.main_package_idx]
            if removable:
                new_solution[random.choice(removable)] = 0

        elif weight[0] > weight[1] and len(indices) < self.max_size:
            # LU-focused - add co-occurrence
            candidates = [c for c in self.cooccur_candidates[:30] if c not in indices]
            if candidates:
                new_solution[random.choice(candidates)] = 1

        elif weight[1] > weight[0] and len(indices) < self.max_size:
            # SS-focused - add semantic
            candidates = [c for c in self.semantic_candidates[:30] if c not in indices]
            if candidates:
                new_solution[random.choice(candidates)] = 1

        return new_solution

    def repair_solution(self, solution):
        """Ensure solution feasibility"""
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

    def shake(self, solution, k):
        """Shaking phase - apply neighborhood k times"""
        shaken = solution.copy()

        # Apply neighborhood k with increasing intensity
        for _ in range(k + 1):
            neighborhood = self.neighborhoods[k % self.n_neighborhoods]
            shaken = neighborhood(shaken)
            shaken = self.repair_solution(shaken)

        return shaken

    def decomposition_local_search(self, solution, max_neighbors=20):
        """Local search guided by decomposition"""
        non_dominated = []
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)

        # Test multiple weight-guided moves
        weights_to_test = random.sample(list(self.weight_vectors),
                                      min(10, len(self.weight_vectors)))

        for weight in weights_to_test:
            # Make move in direction of weight
            neighbor = self.directional_move(solution, weight)
            neighbor = self.repair_solution(neighbor)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if neighbor_obj[2] != float('inf'):
                # Check dominance
                if self.dominates(neighbor_obj, best_objectives):
                    best_solution = neighbor
                    best_objectives = neighbor_obj
                    non_dominated = [(neighbor, neighbor_obj)]
                elif not self.dominates(best_objectives, neighbor_obj):
                    # Non-dominated - add to list
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

        # Also test each neighborhood once
        for neighborhood in self.neighborhoods[:3]:  # Test first 3 neighborhoods
            neighbor = neighborhood(solution)
            neighbor = self.repair_solution(neighbor)
            neighbor_obj = self.evaluate_objectives(neighbor)

            if neighbor_obj[2] != float('inf'):
                if self.dominates(neighbor_obj, best_objectives):
                    best_solution = neighbor
                    best_objectives = neighbor_obj
                    non_dominated = [(neighbor, neighbor_obj)]
                elif not self.dominates(best_objectives, neighbor_obj):
                    non_dominated.append((neighbor, neighbor_obj))

        return non_dominated[:max_neighbors]

    def smart_initialization(self, strategy='hybrid'):
        """Initialize solution with domain knowledge"""
        chromosome = np.zeros(self.n_packages)
        chromosome[self.main_package_idx] = 1

        target_size = random.randint(3, 7)

        if strategy == 'cooccur' and self.cooccur_candidates:
            n_select = min(target_size - 1, len(self.cooccur_candidates))
            selected = np.random.choice(self.cooccur_candidates[:50],
                                      min(n_select, 50), replace=False)
            chromosome[selected] = 1

        elif strategy == 'semantic' and self.semantic_candidates:
            n_select = min(target_size - 1, len(self.semantic_candidates))
            selected = np.random.choice(self.semantic_candidates[:50],
                                      min(n_select, 50), replace=False)
            chromosome[selected] = 1

        elif strategy == 'hybrid':
            # Mix both strategies
            n_cooccur = target_size // 2
            n_semantic = target_size - n_cooccur - 1

            if self.cooccur_candidates and n_cooccur > 0:
                selected = np.random.choice(self.cooccur_candidates[:30],
                                          min(n_cooccur, 30), replace=False)
                chromosome[selected] = 1

            if self.semantic_candidates and n_semantic > 0:
                selected = np.random.choice(self.semantic_candidates[:30],
                                          min(n_semantic, 30), replace=False)
                chromosome[selected] = 1

        else:
            # Random
            n_select = target_size - 1
            candidates = [i for i in range(self.n_packages) if i != self.main_package_idx]
            selected = np.random.choice(candidates, n_select, replace=False)
            chromosome[selected] = 1

        return chromosome

    def initialize_archive(self):
        """Initialize archive with diverse solutions"""
        print("Initializing archive...")

        strategies = ['cooccur', 'semantic', 'hybrid', 'random']

        for _ in range(self.archive_limit * 2):
            strategy = random.choice(strategies)
            solution = self.smart_initialization(strategy)
            objectives = self.evaluate_objectives(solution)

            if objectives[2] != float('inf'):
                self.update_archive(solution, objectives)

            if len(self.archive) >= self.archive_limit:
                break

        print(f"Archive initialized with {len(self.archive)} solutions")
        print(f"Objective bounds: LU=[{self.obj_min[0]:.1f}, {self.obj_max[0]:.1f}], "
              f"SS=[{self.obj_min[1]:.3f}, {self.obj_max[1]:.3f}], "
              f"RSS=[{self.obj_min[2]:.1f}, {self.obj_max[2]:.1f}]")

    def update_archive(self, solution, objectives):
        """Update archive with non-dominated solutions"""
        is_dominated = False
        to_remove = []

        for i, (arch_sol, arch_obj) in enumerate(self.archive):
            if self.dominates(arch_obj, objectives):
                is_dominated = True
                break
            elif self.dominates(objectives, arch_obj):
                to_remove.append(i)

        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(to_remove):
                self.archive.pop(i)

            self.archive.append((solution, objectives))
            self.counter_archive_improvement += 1

            # Truncate if exceeds limit
            if len(self.archive) > self.archive_limit:
                self.truncate_archive_with_crowding()

            return True

        return False

    def truncate_archive_with_crowding(self):
        """Truncate archive using crowding distance"""
        objectives_array = np.array([obj for _, obj in self.archive])
        n_solutions = len(self.archive)

        # Calculate crowding distance
        crowding_distances = np.zeros(n_solutions)

        for m in range(self.n_objectives):
            sorted_indices = np.argsort(objectives_array[:, m])
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')

            obj_range = (objectives_array[sorted_indices[-1], m] -
                        objectives_array[sorted_indices[0], m])

            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (objectives_array[sorted_indices[i + 1], m] -
                              objectives_array[sorted_indices[i - 1], m]) / obj_range
                    crowding_distances[sorted_indices[i]] += distance

        # Keep solutions with highest crowding distance
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_limit]]

    def select_unexplored_solution(self):
        """Select solution from archive for exploration"""
        # Prefer unexplored solutions
        for sol, obj in self.archive:
            sol_tuple = tuple(sol)
            if sol_tuple not in self.explored_solutions:
                self.explored_solutions.add(sol_tuple)
                return sol

        # If all explored, select based on crowding or random
        if self.archive:
            selected = random.choice(self.archive)
            self.explored_solutions.add(tuple(selected[0]))
            return selected[0]

        # Fallback to new initialization
        return self.smart_initialization('hybrid')

    def calculate_metrics(self):
        """Calculate quality metrics for tracking"""
        if not self.track_metrics or len(self.archive) < 3:
            return None

        objectives = np.array([sol[1] for sol in self.archive])

        # Normalize objectives for hypervolume calculation
        normalized_objectives = np.array([self.normalize_objectives(obj) for obj in objectives])

        metrics = {}
        metrics['hypervolume'] = self.metrics_calculator.hypervolume(normalized_objectives)
        metrics['spacing'] = self.metrics_calculator.spacing(normalized_objectives)
        metrics['diversity'] = self.metrics_calculator.diversity(normalized_objectives)

        return metrics

    def run(self):
        """Main MOVNS v5 loop with decomposition-guided search"""
        print(f"\nStarting MOVNS v5 for {self.main_package}...")
        print("="*60)

        MIN_ITERATIONS = 15
        IMPROVEMENT_THRESHOLD = 0.001
        no_improvement = 0
        best_hv = 0

        for iteration in range(self.max_iterations):
            # Select solution from archive
            current_solution = self.select_unexplored_solution()

            # Variable Neighborhood Search with decomposition
            k = 0
            improvements_this_iter = 0

            while k < self.n_neighborhoods:
                # Shaking phase
                x_prime = self.shake(current_solution, k)

                # Local search with decomposition
                non_dominated = self.decomposition_local_search(x_prime)

                # Update archive with non-dominated solutions
                improved = False
                for sol, obj in non_dominated:
                    if self.update_archive(sol, obj):
                        improved = True
                        improvements_this_iter += 1

                # Neighborhood change
                if improved:
                    current_solution = non_dominated[0][0]  # Use best from local search
                    k = 0  # Reset to first neighborhood
                else:
                    k += 1  # Try next neighborhood

            # Progress report
            if iteration % 5 == 0:
                best_obj = min(self.archive, key=lambda x: x[1][0])[1]
                print(f"Iteration {iteration}: Archive size={len(self.archive)}, "
                      f"Improvements={self.counter_archive_improvement}")
                print(f"  Best: LU={-best_obj[0]:.1f}, SS={-best_obj[1]:.4f}, RSS={best_obj[2]:.1f}")

                if self.track_metrics:
                    metrics = self.calculate_metrics()
                    if metrics:
                        for key in self.metrics_history:
                            if key in metrics:
                                self.metrics_history[key].append(metrics[key])

                        current_hv = metrics.get('hypervolume', 0)
                        print(f"  HV={current_hv:.4f}, No-improvement={no_improvement}")

                        # Check convergence
                        if iteration >= MIN_ITERATIONS:
                            if current_hv > best_hv * (1 + IMPROVEMENT_THRESHOLD):
                                best_hv = current_hv
                                no_improvement = 0
                            else:
                                no_improvement += 1

                            if no_improvement >= self.min_no_improvement:
                                print(f"\nStopping: No improvement for {self.min_no_improvement} iterations")
                                break

        print(f"\nFinal archive: {len(self.archive)} solutions")
        print(f"Total archive improvements: {self.counter_archive_improvement}")

        # Convert to output format
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