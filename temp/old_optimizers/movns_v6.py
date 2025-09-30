"""
MOVNS v6 - VNS with Decomposition Neighborhood
Inherits from v2 and adds one decomposition-guided neighborhood
Maintains VNS identity while incorporating MOEA/D benefits
"""

import numpy as np
import sys
import os
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.movns_v2 import MOVNS_V2


class MOVNS_V6(MOVNS_V2):
    """
    MOVNS v6: Variable Neighborhood Search with Decomposition Operator
    
    Key innovation: One neighborhood uses decomposition to guide search
    Maintains VNS structure with 3 traditional + 1 decomposition neighborhood
    """

    def __init__(self, main_package, archive_size=50, max_iterations=30,
                 k_max=4, track_metrics=False, min_no_improvement=5,
                 n_weight_vectors=10):
        """
        Initialize MOVNS v6 with decomposition components
        
        Args:
            n_weight_vectors: Number of weight vectors for decomposition (default 10)
        """
        super().__init__(main_package, archive_size, max_iterations,
                        k_max, track_metrics, min_no_improvement)
        
        self.n_weight_vectors = n_weight_vectors
        self.weight_vectors = self.generate_uniform_weights(n_weight_vectors)
        self.z_star = np.array([1.0, 1.0, 0.0])
        
        self.neighborhoods = [
            self.n1_objective_guided,
            self.n2_size_adjustment, 
            self.n3_semantic_coherence,
            self.n4_decomposition_guided
        ]
        
        print(f"MOVNS v6 initialized with {n_weight_vectors} weight vectors")
        print("Using 3 VNS + 1 decomposition neighborhood")

    def generate_uniform_weights(self, n: int) -> np.ndarray:
        """Generate uniformly distributed weight vectors"""
        weights = []
        for i in range(n):
            w = np.random.dirichlet(np.ones(3))
            weights.append(w)
        return np.array(weights)

    def n1_objective_guided(self, solution: np.ndarray) -> np.ndarray:
        """Traditional VNS neighborhood: focus on weakest objective"""
        objectives = self.evaluate_objectives(solution)
        norm_obj = self.normalize_objectives(objectives)
        
        weakest_idx = np.argmin(norm_obj[:2])
        
        indices = np.where(solution == 1)[0]
        
        if weakest_idx == 0 and len(indices) < self.max_size:
            candidates = self.cooccur_candidates[:20]
            valid = [c for c in candidates if solution[c] == 0]
            if valid:
                new_solution = solution.copy()
                new_solution[np.random.choice(valid)] = 1
                return new_solution
                
        elif weakest_idx == 1 and len(indices) < self.max_size:
            candidates = self.semantic_candidates[:20]
            valid = [c for c in candidates if solution[c] == 0]
            if valid:
                new_solution = solution.copy()
                new_solution[np.random.choice(valid)] = 1
                return new_solution
        
        return solution

    def n2_size_adjustment(self, solution: np.ndarray) -> np.ndarray:
        """Traditional VNS neighborhood: adjust size toward ideal"""
        indices = np.where(solution == 1)[0]
        current_size = len(indices)
        
        if current_size > self.ideal_size and current_size > self.min_size:
            contributions = []
            for idx in indices:
                temp = solution.copy()
                temp[idx] = 0
                obj_before = self.evaluate_objectives(solution)
                obj_after = self.evaluate_objectives(temp)
                loss = np.sum(np.abs(obj_after[:2] - obj_before[:2]))
                contributions.append((idx, loss))
            
            contributions.sort(key=lambda x: x[1])
            new_solution = solution.copy()
            new_solution[contributions[0][0]] = 0
            return new_solution
            
        elif current_size < self.ideal_size and current_size < self.max_size:
            candidates = list(self.cooccur_candidates[:10]) + list(self.semantic_candidates[:10])
            valid = [c for c in candidates if solution[c] == 0]
            if valid:
                new_solution = solution.copy()
                new_solution[np.random.choice(valid)] = 1
                return new_solution
        
        return solution

    def n3_semantic_coherence(self, solution: np.ndarray) -> np.ndarray:
        """Traditional VNS neighborhood: improve semantic coherence"""
        indices = np.where(solution == 1)[0]
        
        if len(indices) < 2:
            return solution
        
        embeddings_subset = self.embeddings[indices]
        centroid = np.mean(embeddings_subset, axis=0)
        
        distances_internal = np.linalg.norm(embeddings_subset - centroid, axis=1)
        worst_internal = indices[np.argmax(distances_internal)]
        
        candidates = self.semantic_candidates[:30]
        valid = [c for c in candidates if solution[c] == 0]
        
        if valid and len(indices) > self.min_size:
            distances_external = np.linalg.norm(self.embeddings[valid] - centroid, axis=1)
            best_external = valid[np.argmin(distances_external)]
            
            new_solution = solution.copy()
            new_solution[worst_internal] = 0
            new_solution[best_external] = 1
            return new_solution
        
        return solution

    def n4_decomposition_guided(self, solution: np.ndarray) -> np.ndarray:
        """
        Decomposition-guided neighborhood.
        Uses weight vectors to identify and explore sparse regions.
        """
        sparse_weight = self.find_sparse_region_weight()
        
        indices = np.where(solution == 1)[0]
        current_size = len(indices)
        
        if sparse_weight[0] > 0.5 and current_size < self.max_size:
            candidates = self.cooccur_candidates[:10]
            best = self.select_by_decomposition(solution, candidates, sparse_weight, 'add')
            if best is not None:
                new_solution = solution.copy()
                new_solution[best] = 1
                return new_solution
                
        elif sparse_weight[1] > 0.5 and current_size < self.max_size:
            candidates = self.semantic_candidates[:10]
            best = self.select_by_decomposition(solution, candidates, sparse_weight, 'add')
            if best is not None:
                new_solution = solution.copy()
                new_solution[best] = 1
                return new_solution
                
        elif sparse_weight[2] > 0.5 and current_size > self.min_size:
            if len(indices) > 2:
                contributions = self.calculate_weighted_contributions(solution, sparse_weight)
                worst_idx = indices[np.argmin(contributions)]
                new_solution = solution.copy()
                new_solution[worst_idx] = 0
                return new_solution
        
        else:
            if current_size < self.max_size:
                all_candidates = list(set(list(self.cooccur_candidates[:5]) + 
                                        list(self.semantic_candidates[:5])))
                best = self.select_by_decomposition(solution, all_candidates, sparse_weight, 'add')
                if best is not None:
                    new_solution = solution.copy()
                    new_solution[best] = 1
                    return new_solution

        return solution

    def find_sparse_region_weight(self) -> np.ndarray:
        """
        Find weight vector for least explored region in archive.
        Simple and fast: divides objective space into sectors.
        """
        if len(self.archive) < 10:
            return np.array([0.33, 0.33, 0.34])

        archive_objectives = []
        for sol in self.archive:
            if isinstance(sol, dict) and 'objectives' in sol:
                obj = sol['objectives']
                if isinstance(obj, dict):
                    norm_obj = self.normalize_objectives(np.array([
                        obj['linked_usage'],
                        obj['semantic_similarity'],
                        obj['set_size']
                    ]))
                else:
                    norm_obj = self.normalize_objectives(obj)
            else:
                obj = self.evaluate_objectives(sol['chromosome'])
                norm_obj = self.normalize_objectives(obj)
            archive_objectives.append(norm_obj)

        archive_objectives = np.array(archive_objectives)

        sector_counts = np.zeros(len(self.weight_vectors))

        for obj in archive_objectives:
            obj_positive = np.abs(obj)
            if np.sum(obj_positive) == 0:
                continue

            obj_normalized = obj_positive / (np.sum(obj_positive) + 1e-8)

            distances = []
            for w in self.weight_vectors:
                cos_sim = np.dot(obj_normalized, w) / (np.linalg.norm(obj_normalized) * np.linalg.norm(w) + 1e-8)
                distances.append(1 - cos_sim)

            closest_weight_idx = np.argmin(distances)
            sector_counts[closest_weight_idx] += 1

        sparse_sector = np.argmin(sector_counts)
        return self.weight_vectors[sparse_sector]

    def select_by_decomposition(self, solution: np.ndarray, candidates: List[int],
                                weight: np.ndarray, operation: str) -> int:
        """
        Select best candidate using Tchebycheff decomposition.
        Fast: tests only provided candidates, not all 9997.
        """
        current_obj = self.evaluate_objectives(solution)
        current_norm = self.normalize_objectives(current_obj)
        current_score = np.max(weight * np.abs(current_norm - self.z_star))

        best_candidate = None
        best_score = current_score

        for candidate in candidates:
            if operation == 'add' and solution[candidate] == 1:
                continue
            elif operation == 'remove' and solution[candidate] == 0:
                continue

            test_solution = solution.copy()
            if operation == 'add':
                test_solution[candidate] = 1
            else:
                test_solution[candidate] = 0

            indices = np.where(test_solution == 1)[0]
            if len(indices) < self.min_size or len(indices) > self.max_size:
                continue

            test_obj = self.evaluate_objectives(test_solution)
            test_norm = self.normalize_objectives(test_obj)
            test_score = np.max(weight * np.abs(test_norm - self.z_star))

            if test_score < best_score:
                best_score = test_score
                best_candidate = candidate

        return best_candidate

    def calculate_weighted_contributions(self, solution: np.ndarray,
                                        weight: np.ndarray) -> np.ndarray:
        """
        Calculate contribution of each selected package weighted by decomposition.
        """
        indices = np.where(solution == 1)[0]
        contributions = np.zeros(len(indices))

        base_obj = self.evaluate_objectives(solution)
        base_norm = self.normalize_objectives(base_obj)
        base_score = np.max(weight * np.abs(base_norm - self.z_star))

        for i, idx in enumerate(indices):
            temp_solution = solution.copy()
            temp_solution[idx] = 0

            temp_obj = self.evaluate_objectives(temp_solution)
            temp_norm = self.normalize_objectives(temp_obj)
            temp_score = np.max(weight * np.abs(temp_norm - self.z_star))

            contributions[i] = temp_score - base_score

        return contributions

    def mobi_p_search(self, solution: np.ndarray, neighborhood_idx: int = 0) -> np.ndarray:
        """
        Modified MOBI/P search that adapts based on neighborhood type.
        """
        if neighborhood_idx < 3:
            return self.mobi_p_local_search(solution)
        else:
            return self.decomposition_local_search(solution)

    def decomposition_local_search(self, solution: np.ndarray) -> np.ndarray:
        """
        Specialized local search when using decomposition neighborhood.
        More focused and efficient.
        """
        current = solution.copy()
        current_obj = self.evaluate_objectives(current)

        weight = self.find_best_weight_for_solution(current_obj)

        for _ in range(5):
            improved = False
            indices = np.where(current == 1)[0]

            if len(indices) < self.max_size:
                add_candidates = list(self.cooccur_candidates[:5]) + list(self.semantic_candidates[:5])
                add_candidate = self.select_by_decomposition(current, add_candidates, weight, 'add')
                if add_candidate is not None:
                    test_solution = current.copy()
                    test_solution[add_candidate] = 1
                    test_obj = self.evaluate_objectives(test_solution)
                    if self.is_better_decomposition(test_obj, current_obj, weight):
                        current = test_solution
                        current_obj = test_obj
                        improved = True

            if len(indices) > self.min_size and not improved:
                remove_candidate = self.select_by_decomposition(current, indices.tolist(), weight, 'remove')
                if remove_candidate is not None:
                    test_solution = current.copy()
                    test_solution[remove_candidate] = 0
                    test_obj = self.evaluate_objectives(test_solution)
                    if self.is_better_decomposition(test_obj, current_obj, weight):
                        current = test_solution
                        current_obj = test_obj
                        improved = True

            if not improved:
                break

        return current

    def find_best_weight_for_solution(self, objectives: np.ndarray) -> np.ndarray:
        """
        Find weight vector that best represents the solution's position.
        """
        norm_obj = self.normalize_objectives(objectives)
        norm_obj_positive = np.abs(norm_obj)

        if np.sum(norm_obj_positive) == 0:
            return np.array([0.33, 0.33, 0.34])

        direction = norm_obj_positive / np.sum(norm_obj_positive)

        best_weight = self.weight_vectors[0]
        best_similarity = -1

        for w in self.weight_vectors:
            similarity = np.dot(direction, w)
            if similarity > best_similarity:
                best_similarity = similarity
                best_weight = w

        return best_weight

    def is_better_decomposition(self, obj1: np.ndarray, obj2: np.ndarray,
                               weight: np.ndarray) -> bool:
        """
        Check if obj1 is better than obj2 using decomposition.
        """
        norm1 = self.normalize_objectives(obj1)
        norm2 = self.normalize_objectives(obj2)

        score1 = np.max(weight * np.abs(norm1 - self.z_star))
        score2 = np.max(weight * np.abs(norm2 - self.z_star))

        return score1 < score2

    def run(self) -> List[Dict]:
        """
        Main VNS loop with decomposition neighborhood.
        """
        print("\nStarting MOVNS v6 with decomposition neighborhood...")

        no_improvement_counter = 0
        best_hv = 0

        for iteration in range(self.max_iterations):
            current_chromosome = self.select_unexplored_solution()
            current_solution = {
                'chromosome': current_chromosome,
                'objectives': {}
            }

            k = 0
            local_no_improvement = 0

            while k < self.k_max and local_no_improvement < 3:
                if k < 3:
                    x_prime = self.shake(current_solution['chromosome'], self.neighborhoods[k], intensity=k+1)
                else:
                    x_prime = current_solution['chromosome'].copy()

                x_neighbor = self.neighborhoods[k](x_prime)

                if x_neighbor is None or not isinstance(x_neighbor, np.ndarray):
                    x_neighbor = x_prime

                x_local = self.mobi_p_search(x_neighbor, k)

                if x_local is None or not isinstance(x_local, np.ndarray):
                    x_local = x_neighbor

                x_obj = self.evaluate_objectives(x_local)
                current_obj = self.evaluate_objectives(current_solution['chromosome'])

                if self.dominates(x_obj, current_obj) or \
                   (not self.dominates(current_obj, x_obj) and np.random.random() < 0.3):
                    current_solution['chromosome'] = x_local
                    current_solution['objectives'] = {
                        'linked_usage': x_obj[0],
                        'semantic_similarity': x_obj[1],
                        'set_size': x_obj[2]
                    }
                    k = 0
                    local_no_improvement = 0
                else:
                    k += 1
                    local_no_improvement += 1

            obj = self.evaluate_objectives(current_solution['chromosome'])
            self.update_archive(current_solution['chromosome'], obj)

            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])
                    current_hv = metrics.get('hypervolume', 0)

                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                     f"HV={current_hv:.4f}, Best_HV={best_hv:.4f}")

                if no_improvement_counter >= self.min_no_improvement:
                    print(f"\nEarly stopping: No improvement for {self.min_no_improvement} checks")
                    break

        print(f"\nMOVNS v6 completed: {len(self.archive)} solutions found")

        if self.track_metrics:
            final_hv = self.metrics_history['hypervolume'][-1] if self.metrics_history['hypervolume'] else 0
            print(f"Final hypervolume: {final_hv:.4f}")

        return self.archive