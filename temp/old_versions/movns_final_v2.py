"""
MOVNS Advanced - Aggressive Multi-Objective VNS with Advanced Local Search
Combines: Pareto Local Search, Simulated Annealing, Tabu Search, Adaptive Learning
No simplifications, no fallbacks - maximum performance focus
"""

import numpy as np
import pickle
import random
import time
from typing import List, Dict, Tuple, Set
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.quality_metrics import QualityMetrics
from optimizer.movns_v2 import MOVNS_V2


class MOVNS_Final_V2(MOVNS_V2):
    """
    Advanced MOVNS with multiple state-of-the-art local search methods
    """
    
    def __init__(self, main_package, archive_size=200, max_iterations=100,
                 track_metrics=True):
        super().__init__(main_package, archive_size, max_iterations,
                        k_max=6, track_metrics=track_metrics, min_no_improvement=20)
        
        self.temperature = 1.0
        self.cooling_rate = 0.995
        self.min_temperature = 0.01
        
        self.tabu_list = deque(maxlen=50)
        self.tabu_tenure = 20
        
        self.learning_rates = np.ones(6) * 0.5
        self.neighborhood_success = np.zeros(6)
        self.neighborhood_calls = np.zeros(6)
        
        self.pareto_queue = deque()
        self.intensification_memory = []
        self.diversification_memory = set()
        
        self.adaptive_params = {
            'local_search_intensity': 10,
            'perturbation_strength': 0.1,
            'archive_pressure': 0.3,
            'exploration_rate': 0.4
        }
        
        self.iteration_no_improvement = 0
        self.best_hypervolume = 0
        self.stagnation_counter = 0
        
        print(f"MOVNS Advanced initialized with aggressive settings")
        print(f"Archive: {archive_size}, Iterations: {max_iterations}, No fallbacks")
    
    def pareto_local_search(self, solution: np.ndarray, max_neighbors: int = 20) -> List[np.ndarray]:
        """
        Aggressive Pareto Local Search with queue management
        """
        pareto_set = []
        queue = deque([solution])
        evaluated = set()
        
        while queue and len(evaluated) < max_neighbors:
            current = queue.popleft()
            current_tuple = tuple(current)
            
            if current_tuple in evaluated:
                continue
            evaluated.add(current_tuple)
            
            current_obj = self.evaluate_objectives(current)
            
            indices = np.where(current == 1)[0]
            neighbors = []
            
            if len(indices) < self.max_size:
                cooccur_scores = self.rel_matrix[self.main_package_idx]
                if hasattr(cooccur_scores, 'toarray'):
                    cooccur_scores = cooccur_scores.toarray().flatten()
                else:
                    cooccur_scores = np.asarray(cooccur_scores).flatten()
                cooccur_scores[indices] = -np.inf
                top_candidates = np.argsort(cooccur_scores)[-30:]
                
                for candidate in top_candidates:
                    if current[candidate] == 0:
                        neighbor = current.copy()
                        neighbor[candidate] = 1
                        neighbors.append(neighbor)
            
            if len(indices) > self.min_size:
                for _ in range(min(10, len(indices))):
                    idx_to_remove = np.random.choice(indices)
                    neighbor = current.copy()
                    neighbor[idx_to_remove] = 0
                    neighbors.append(neighbor)
            
            if len(indices) >= 3 and len(indices) < self.max_size:
                for _ in range(10):
                    idx_to_remove = np.random.choice(indices)
                    candidates = self.semantic_candidates[:50]
                    valid = [c for c in candidates if current[c] == 0]
                    if valid:
                        idx_to_add = np.random.choice(valid)
                        neighbor = current.copy()
                        neighbor[idx_to_remove] = 0
                        neighbor[idx_to_add] = 1
                        neighbors.append(neighbor)
            
            non_dominated_neighbors = []
            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in evaluated and neighbor_tuple not in self.tabu_list:
                    neighbor_obj = self.evaluate_objectives(neighbor)
                    
                    is_dominated = False
                    dominates_current = self.dominates(neighbor_obj, current_obj)
                    
                    for sol, obj in pareto_set:
                        if self.dominates(obj, neighbor_obj):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        pareto_set = [(s, o) for s, o in pareto_set 
                                     if not self.dominates(neighbor_obj, o)]
                        pareto_set.append((neighbor, neighbor_obj))
                        
                        if dominates_current or len(queue) < 20:
                            queue.append(neighbor)
        
        return [sol for sol, _ in pareto_set]
    
    def simulated_annealing_accept(self, current_obj: np.ndarray, 
                                  candidate_obj: np.ndarray) -> bool:
        """
        Multi-objective simulated annealing acceptance criterion
        """
        if self.dominates(candidate_obj, current_obj):
            return True
        
        if self.dominates(current_obj, candidate_obj):
            if self.temperature > self.min_temperature:
                delta = np.sum(np.abs(candidate_obj - current_obj))
                probability = np.exp(-delta / self.temperature)
                return np.random.random() < probability
            return False
        
        return np.random.random() < 0.5
    
    def adaptive_perturbation(self, solution: np.ndarray, strength: float) -> np.ndarray:
        """
        Adaptive perturbation based on search progress
        """
        perturbed = solution.copy()
        indices = np.where(solution == 1)[0]
        
        n_changes = max(1, int(len(indices) * strength))
        
        if self.stagnation_counter > 10:
            n_changes *= 2
        
        for _ in range(n_changes):
            operation = np.random.choice(['add', 'remove', 'swap'], 
                                        p=[0.3, 0.3, 0.4])
            
            if operation == 'add' and len(indices) < self.max_size:
                pool = np.concatenate([
                    self.cooccur_candidates[:100],
                    self.semantic_candidates[:100],
                    np.random.choice(self.n_packages, 50, replace=False)
                ])
                valid = [p for p in pool if perturbed[p] == 0]
                if valid:
                    selected = np.random.choice(valid)
                    perturbed[selected] = 1
                    indices = np.where(perturbed == 1)[0]
            
            elif operation == 'remove' and len(indices) > self.min_size:
                idx = np.random.choice(indices)
                perturbed[idx] = 0
                indices = np.where(perturbed == 1)[0]
            
            elif operation == 'swap' and len(indices) >= 2:
                idx_remove = np.random.choice(indices)
                candidates = self.cluster_candidates[:100]
                valid = [c for c in candidates if perturbed[c] == 0]
                if valid:
                    idx_add = np.random.choice(valid)
                    perturbed[idx_remove] = 0
                    perturbed[idx_add] = 1
                    indices = np.where(perturbed == 1)[0]
        
        return self.repair_solution(perturbed)

    def iterated_local_search(self, initial: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Iterated Local Search with adaptive acceptance
        """
        best = initial.copy()
        best_obj = self.evaluate_objectives(best)
        current = initial.copy()

        for i in range(iterations):
            local_optimal = self.aggressive_local_search(current)
            local_obj = self.evaluate_objectives(local_optimal)

            if self.dominates(local_obj, best_obj):
                best = local_optimal.copy()
                best_obj = local_obj
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            strength = self.adaptive_params['perturbation_strength']
            if self.stagnation_counter > 5:
                strength *= 2

            current = self.adaptive_perturbation(local_optimal, strength)

            if self.simulated_annealing_accept(local_obj, self.evaluate_objectives(current)):
                pass
            else:
                current = local_optimal.copy()

        return best

    def aggressive_local_search(self, solution: np.ndarray) -> np.ndarray:
        """
        Aggressive multi-operator local search
        """
        current = solution.copy()
        current_obj = self.evaluate_objectives(current)

        improvement_found = True
        iterations_without_improvement = 0

        while improvement_found and iterations_without_improvement < 5:
            improvement_found = False
            indices = np.where(current == 1)[0]

            operators = [
                ('intensify_cooccurrence', 0.3),
                ('intensify_semantic', 0.3),
                ('diversify_cluster', 0.2),
                ('exchange_optimal', 0.2)
            ]

            for _ in range(self.adaptive_params['local_search_intensity']):
                op_name = np.random.choice(
                    [o[0] for o in operators],
                    p=[o[1] for o in operators]
                )

                if op_name == 'intensify_cooccurrence' and len(indices) < self.max_size:
                    scores = self.rel_matrix[indices].sum(axis=0)
                    if hasattr(scores, 'toarray'):
                        scores = scores.toarray().flatten()
                    else:
                        scores = np.asarray(scores).flatten()
                    scores[indices] = -np.inf
                    top_10 = np.argsort(scores)[-10:]

                    for candidate in top_10:
                        if current[candidate] == 0:
                            test = current.copy()
                            test[candidate] = 1
                            test_obj = self.evaluate_objectives(test)

                            if self.dominates(test_obj, current_obj):
                                current = test
                                current_obj = test_obj
                                improvement_found = True
                                iterations_without_improvement = 0
                                break

                elif op_name == 'intensify_semantic' and len(indices) < self.max_size:
                    embeddings_subset = self.embeddings[indices]
                    centroid = np.mean(embeddings_subset, axis=0)
                    distances = np.linalg.norm(self.embeddings - centroid, axis=1)
                    distances[indices] = np.inf
                    closest_10 = np.argsort(distances)[:10]

                    for candidate in closest_10:
                        if current[candidate] == 0:
                            test = current.copy()
                            test[candidate] = 1
                            test_obj = self.evaluate_objectives(test)

                            if self.dominates(test_obj, current_obj):
                                current = test
                                current_obj = test_obj
                                improvement_found = True
                                iterations_without_improvement = 0
                                break

                elif op_name == 'diversify_cluster' and len(indices) < self.max_size:
                    cluster_counts = np.bincount(self.cluster_labels[indices],
                                                minlength=200)
                    underrepresented = np.where(cluster_counts == 0)[0][:5]

                    for cluster_id in underrepresented:
                        candidates = np.where(self.cluster_labels == cluster_id)[0]
                        valid = [c for c in candidates if current[c] == 0]

                        if valid:
                            selected = np.random.choice(valid[:10] if len(valid) > 10 else valid)
                            test = current.copy()
                            test[selected] = 1
                            test_obj = self.evaluate_objectives(test)

                            if self.dominates(test_obj, current_obj):
                                current = test
                                current_obj = test_obj
                                improvement_found = True
                                iterations_without_improvement = 0
                                break

                elif op_name == 'exchange_optimal' and len(indices) >= 3:
                    contributions = []
                    for idx in indices:
                        temp = current.copy()
                        temp[idx] = 0
                        obj_after = self.evaluate_objectives(temp)
                        loss = np.sum(np.abs(current_obj[:2] - obj_after[:2]))
                        contributions.append((idx, loss))

                    contributions.sort(key=lambda x: x[1])
                    weakest_3 = [c[0] for c in contributions[:3]]

                    best_exchange = None
                    best_exchange_obj = current_obj

                    for weak_idx in weakest_3:
                        candidates = list(self.cooccur_candidates[:30]) + \
                                   list(self.semantic_candidates[:30])
                        valid = [c for c in candidates if current[c] == 0]

                        for replacement in valid[:10]:
                            test = current.copy()
                            test[weak_idx] = 0
                            test[replacement] = 1
                            test_obj = self.evaluate_objectives(test)

                            if self.dominates(test_obj, best_exchange_obj):
                                best_exchange = test
                                best_exchange_obj = test_obj

                    if best_exchange is not None:
                        current = best_exchange
                        current_obj = best_exchange_obj
                        improvement_found = True
                        iterations_without_improvement = 0

                indices = np.where(current == 1)[0]

            if not improvement_found:
                iterations_without_improvement += 1

        return current

    def adaptive_neighborhood_selection(self) -> int:
        """
        Select neighborhood based on learned success rates
        """
        epsilon = 0.1

        success_rates = []
        for i in range(self.k_max):
            if self.neighborhood_calls[i] > 0:
                rate = self.neighborhood_success[i] / self.neighborhood_calls[i]
                success_rates.append(rate * self.learning_rates[i])
            else:
                success_rates.append(epsilon)

        success_rates = np.array(success_rates)

        if np.sum(success_rates) == 0:
            return np.random.choice(self.k_max)

        success_rates = success_rates / np.sum(success_rates)

        return np.random.choice(self.k_max, p=success_rates)

    def update_learning_rates(self, neighborhood_idx: int, improved: bool):
        """
        Update learning rates based on performance
        """
        self.neighborhood_calls[neighborhood_idx] += 1

        if improved:
            self.neighborhood_success[neighborhood_idx] += 1
            self.learning_rates[neighborhood_idx] = min(1.0,
                self.learning_rates[neighborhood_idx] * 1.05)
        else:
            self.learning_rates[neighborhood_idx] = max(0.1,
                self.learning_rates[neighborhood_idx] * 0.95)

    def run(self) -> List[Dict]:
        """
        Main MOVNS loop with aggressive multi-method local search.
        No simplifications, maximum search intensity.
        """
        print(f"\nStarting MOVNS Advanced with aggressive optimization...")
        print(f"Settings: {self.max_iterations} iterations, {self.archive_limit} archive size")

        no_improvement_counter = 0
        best_hv = 0
        global_best_objectives = np.array([0, 0, float('inf')])

        for iteration in range(self.max_iterations):
            if iteration % 10 == 0:
                self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)

                if len(self.archive) > 10:
                    archive_objectives = []
                    for sol in self.archive:
                        obj = self.evaluate_objectives(sol['chromosome'])
                        archive_objectives.append(obj)

                    archive_objectives = np.array(archive_objectives)
                    self.z_star = np.min(archive_objectives, axis=0)

            if len(self.archive) > 0:
                if np.random.random() < self.adaptive_params['exploration_rate']:
                    current_chromosome = self.select_unexplored_solution()
                else:
                    archive_idx = np.random.choice(min(20, len(self.archive)))
                    current_chromosome = self.archive[archive_idx]['chromosome'].copy()
            else:
                current_chromosome = self.select_unexplored_solution()

            if np.random.random() < 0.3 and iteration > 10:
                pareto_solutions = self.pareto_local_search(current_chromosome)
                if pareto_solutions:
                    for ps in pareto_solutions[:5]:
                        self.pareto_queue.append(ps)
                        ps_obj = self.evaluate_objectives(ps)
                        self.update_archive(ps, ps_obj)
                    current_chromosome = pareto_solutions[0]

            improved_solution = self.iterated_local_search(current_chromosome,
                                                          iterations=5)

            k = self.adaptive_neighborhood_selection()

            x_prime = self.adaptive_perturbation(improved_solution,
                                                self.adaptive_params['perturbation_strength'] * (k + 1) / 6)

            x_local = self.aggressive_local_search(x_prime)

            current_obj = self.evaluate_objectives(improved_solution)
            local_obj = self.evaluate_objectives(x_local)

            if self.simulated_annealing_accept(current_obj, local_obj):
                final_solution = x_local
                final_obj = local_obj
                self.update_learning_rates(k, True)
            else:
                final_solution = improved_solution
                final_obj = current_obj
                self.update_learning_rates(k, False)

            if not self.dominates(global_best_objectives, final_obj):
                if self.dominates(final_obj, global_best_objectives):
                    global_best_objectives = final_obj
                    self.stagnation_counter = 0
                    print(f"New global best: LU={-final_obj[0]:.0f}, "
                          f"SS={-final_obj[1]:.4f}, RSS={final_obj[2]}")

            self.update_archive(final_solution, final_obj)

            if tuple(final_solution) not in self.diversification_memory:
                self.diversification_memory.add(tuple(final_solution))

            self.tabu_list.append(tuple(final_solution))

            if len(self.intensification_memory) < 10 and -final_obj[0] > 5000:
                self.intensification_memory.append({
                    'solution': final_solution,
                    'objectives': final_obj
                })

            if iteration > 0 and iteration % 20 == 0:
                self.adaptive_params['perturbation_strength'] = min(0.5,
                    self.adaptive_params['perturbation_strength'] * 1.1)
                self.adaptive_params['local_search_intensity'] = min(20,
                    self.adaptive_params['local_search_intensity'] + 2)

            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    current_hv = metrics.get('hypervolume', 0)

                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement_counter = 0
                        print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                              f"HV={current_hv:.4f} (improved), Temperature={self.temperature:.3f}")
                    else:
                        no_improvement_counter += 1

                        if iteration % 10 == 0:
                            print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                                  f"HV={current_hv:.4f}, Best={best_hv:.4f}, "
                                  f"No improvement: {no_improvement_counter}")

                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])

            if no_improvement_counter >= self.min_no_improvement * 2:
                print(f"\nStagnation detected at iteration {iteration}")
                print("Applying diversification restart...")

                if len(self.pareto_queue) > 0:
                    current_chromosome = self.pareto_queue.popleft()
                else:
                    current_chromosome = self.select_unexplored_solution()
                    for _ in range(5):
                        current_chromosome = self.adaptive_perturbation(
                            current_chromosome, 0.3)

                no_improvement_counter = self.min_no_improvement
                self.stagnation_counter = 0
                self.temperature = min(1.0, self.temperature * 10)

        print(f"\nMOVNS Advanced completed:")
        print(f"  Final archive: {len(self.archive)} solutions")
        print(f"  Best hypervolume: {best_hv:.4f}")
        print(f"  Global best LU: {-global_best_objectives[0]:.0f}")
        print(f"  Diversification memory: {len(self.diversification_memory)} unique solutions")
        print(f"  Intensification memory: {len(self.intensification_memory)} elite solutions")

        return self.archive