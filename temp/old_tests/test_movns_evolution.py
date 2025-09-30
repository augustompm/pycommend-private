"""
Systematic Testing and Evolution of MOVNS
Goal: Manually evolve MOVNS to improve HV compared to MOEA/D
Following rules.json: No inline comments, focus on results
"""

import numpy as np
import sys
import os
import time
import json
from typing import List, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v2 import MOVNS_V2
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


class MOVNSEvolved(MOVNS_V2):
    """
    Evolved MOVNS with manual improvements based on testing
    """
    
    def __init__(self, main_package, archive_size=100, max_iterations=40,
                 k_max=4, track_metrics=True, min_no_improvement=10):
        super().__init__(main_package, archive_size, max_iterations,
                        k_max, track_metrics, min_no_improvement)
        
        self.neighborhood_success = [0, 0, 0, 0]
        self.neighborhood_calls = [0, 0, 0, 0]
        self.adaptive_intensity = True
        self.elite_size = 10
        self.elite_archive = []
        
    def n1_exploitation(self, solution):
        """Exploit best known regions"""
        indices = np.where(solution == 1)[0]
        
        if len(indices) < self.max_size:
            cooccur_scores = self.rel_matrix[indices].sum(axis=0)
            if hasattr(cooccur_scores, 'toarray'):
                cooccur_scores = cooccur_scores.toarray().flatten()
            else:
                cooccur_scores = np.asarray(cooccur_scores).flatten()
            cooccur_scores[indices] = -np.inf
            
            top_5 = np.argsort(cooccur_scores)[-5:]
            if len(top_5) > 0:
                best = np.random.choice(top_5)
                new_solution = solution.copy()
                new_solution[best] = 1
                return new_solution
        
        return solution
    
    def n2_exploration(self, solution):
        """Explore new regions via semantic similarity"""
        indices = np.where(solution == 1)[0]
        
        if len(indices) > 0 and len(indices) < self.max_size:
            embeddings_subset = self.embeddings[indices]
            centroid = np.mean(embeddings_subset, axis=0)
            
            distances = np.linalg.norm(self.embeddings - centroid, axis=1)
            distances[indices] = np.inf
            
            closest_10 = np.argsort(distances)[:10]
            if len(closest_10) > 0:
                selected = np.random.choice(closest_10)
                new_solution = solution.copy()
                new_solution[selected] = 1
                return new_solution
        
        return solution
    
    def n3_refinement(self, solution):
        """Refine solution by swapping weak packages"""
        indices = np.where(solution == 1)[0]
        
        if len(indices) >= 3:
            contributions = []
            for idx in indices:
                temp = solution.copy()
                temp[idx] = 0
                obj_before = self.evaluate_objectives(solution)
                obj_after = self.evaluate_objectives(temp)
                loss = np.sum(np.abs(obj_after[:2] - obj_before[:2]))
                contributions.append((idx, loss))
            
            contributions.sort(key=lambda x: x[1])
            weakest = contributions[0][0]
            
            candidates = list(self.cooccur_candidates[:20]) + list(self.semantic_candidates[:20])
            candidates = [c for c in candidates if solution[c] == 0]
            
            if candidates:
                replacement = np.random.choice(candidates)
                new_solution = solution.copy()
                new_solution[weakest] = 0
                new_solution[replacement] = 1
                return new_solution
        
        return solution
    
    def n4_diversification(self, solution):
        """Diversify by adding cluster-based packages"""
        indices = np.where(solution == 1)[0]
        
        if len(indices) < self.max_size:
            cluster_counts = np.bincount(self.cluster_labels[indices], minlength=200)
            underrepresented = np.where(cluster_counts < 2)[0]
            
            if len(underrepresented) > 0:
                target_cluster = np.random.choice(underrepresented)
                candidates = np.where(self.cluster_labels == target_cluster)[0]
                candidates = candidates[solution[candidates] == 0]
                
                if len(candidates) > 0:
                    selected = np.random.choice(candidates[:10] if len(candidates) > 10 else candidates)
                    new_solution = solution.copy()
                    new_solution[selected] = 1
                    return new_solution
        
        return solution
    
    def adaptive_shake(self, solution, neighborhood_func, intensity):
        """Adaptive shaking based on search progress"""
        if self.adaptive_intensity:
            archive_size_ratio = len(self.archive) / self.archive_limit
            if archive_size_ratio > 0.8:
                intensity = max(1, intensity - 1)
            elif archive_size_ratio < 0.3:
                intensity = min(3, intensity + 1)
        
        shaken = solution.copy()
        for _ in range(intensity):
            shaken = neighborhood_func(shaken)
            shaken = self.repair_solution(shaken)
        
        return shaken
    
    def enhanced_mobi_p(self, solution, neighborhood_idx):
        """Enhanced MOBI/P with elite guidance"""
        best_solution = solution
        best_objectives = self.evaluate_objectives(solution)
        
        local_archive = []
        
        max_neighbors = 15 if neighborhood_idx < 2 else 10
        
        for _ in range(max_neighbors):
            neighbor = self.neighborhoods[neighborhood_idx](solution)
            
            if np.array_equal(neighbor, solution):
                continue
            
            neighbor_obj = self.evaluate_objectives(neighbor)
            
            if self.dominates(neighbor_obj, best_objectives):
                best_solution = neighbor
                best_objectives = neighbor_obj
                local_archive = [{'chromosome': neighbor, 'objectives': neighbor_obj}]
                self.neighborhood_success[neighborhood_idx] += 1
            elif not self.dominates(best_objectives, neighbor_obj):
                local_archive.append({'chromosome': neighbor, 'objectives': neighbor_obj})
        
        self.neighborhood_calls[neighborhood_idx] += 1
        
        if local_archive:
            non_dominated = []
            for sol in local_archive:
                is_dominated = False
                for other in local_archive:
                    if self.dominates(other['objectives'], sol['objectives']):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(sol)

            if non_dominated:
                return non_dominated[0]['chromosome']

        return best_solution
    
    def run_evolved(self):
        """Enhanced VNS main loop"""
        self.neighborhoods = [
            self.n1_exploitation,
            self.n2_exploration, 
            self.n3_refinement,
            self.n4_diversification
        ]
        
        print(f"\nStarting Evolved MOVNS for {self.main_package}...")
        print("="*60)
        
        no_improvement_count = 0
        best_hv = 0
        
        for iteration in range(self.max_iterations):
            current = self.select_unexplored_solution()
            
            success_rates = []
            for i in range(4):
                if self.neighborhood_calls[i] > 0:
                    success_rates.append(self.neighborhood_success[i] / self.neighborhood_calls[i])
                else:
                    success_rates.append(0.25)
            
            neighborhood_order = np.argsort(success_rates)[::-1]
            
            k_idx = 0
            local_no_improvement = 0
            
            while k_idx < len(neighborhood_order) and local_no_improvement < 2:
                k = neighborhood_order[k_idx]
                
                x_prime = self.adaptive_shake(current, self.neighborhoods[k], k_idx + 1)
                
                x_local = self.enhanced_mobi_p(x_prime, k)
                
                x_obj = self.evaluate_objectives(x_local)
                current_obj = self.evaluate_objectives(current)
                
                if self.dominates(x_obj, current_obj) or \
                   (not self.dominates(current_obj, x_obj) and len(self.archive) < self.archive_limit):
                    current = x_local
                    k_idx = 0
                    local_no_improvement = 0
                else:
                    k_idx += 1
                    local_no_improvement += 1
            
            obj = self.evaluate_objectives(current)
            self.update_archive(current, obj)
            
            if len(self.elite_archive) < self.elite_size or \
               any(self.dominates(obj, e['objectives']) for e in self.elite_archive):
                self.elite_archive.append({'chromosome': current, 'objectives': obj})

                non_dominated_elite = []
                for sol in self.elite_archive:
                    is_dominated = False
                    for other in self.elite_archive:
                        if self.dominates(other['objectives'], sol['objectives']):
                            is_dominated = True
                            break
                    if not is_dominated:
                        non_dominated_elite.append(sol)

                self.elite_archive = non_dominated_elite[:self.elite_size]
            
            if self.track_metrics and iteration % 5 == 0:
                metrics = self.calculate_metrics()
                if metrics:
                    for key in self.metrics_history:
                        if key in metrics and metrics[key] is not None:
                            self.metrics_history[key].append(metrics[key])
                    
                    current_hv = metrics.get('hypervolume', 0)
                    
                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    print(f"Iteration {iteration}: Archive={len(self.archive)}, "
                          f"HV={current_hv:.4f}, Best_HV={best_hv:.4f}")
                    
                    if no_improvement_count >= self.min_no_improvement:
                        print(f"\nEarly stopping: No improvement for {self.min_no_improvement} checks")
                        break
        
        print(f"\nEvolved MOVNS completed: {len(self.archive)} solutions")
        print(f"Neighborhood success rates:")
        for i, name in enumerate(['Exploitation', 'Exploration', 'Refinement', 'Diversification']):
            rate = self.neighborhood_success[i] / max(1, self.neighborhood_calls[i])
            print(f"  {name}: {rate:.2%} ({self.neighborhood_success[i]}/{self.neighborhood_calls[i]})")
        
        return self.archive


def unit_test_neighborhoods():
    """Unit test individual neighborhood operators"""
    print("\n" + "="*60)
    print("UNIT TESTING NEIGHBORHOODS")
    print("="*60)

    algo = MOVNSEvolved('fastapi', archive_size=50, max_iterations=10)

    test_solution = np.zeros(algo.n_packages)
    test_solution[algo.main_package_idx] = 1
    test_solution[algo.cooccur_candidates[:3]] = 1

    initial_obj = algo.evaluate_objectives(test_solution)
    print(f"\nInitial solution: {np.sum(test_solution)} packages")
    print(f"Objectives: LU={initial_obj[0]:.1f}, SS={initial_obj[1]:.4f}, RSS={initial_obj[2]:.1f}")

    neighborhoods = [
        ('n1_exploitation', algo.n1_exploitation),
        ('n2_exploration', algo.n2_exploration),
        ('n3_refinement', algo.n3_refinement),
        ('n4_diversification', algo.n4_diversification)
    ]

    results = []

    for name, func in neighborhoods:
        print(f"\nTesting {name}...")
        improved = 0
        total_tests = 10

        for _ in range(total_tests):
            neighbor = func(test_solution)
            neighbor_obj = algo.evaluate_objectives(neighbor)

            if algo.dominates(neighbor_obj, initial_obj):
                improved += 1
                print(f"  ✓ Improvement: LU={neighbor_obj[0]:.1f}, SS={neighbor_obj[1]:.4f}, RSS={neighbor_obj[2]:.1f}")
            elif not np.array_equal(neighbor, test_solution):
                print(f"  - Change without dominance")

        success_rate = improved / total_tests
        results.append((name, success_rate))
        print(f"  Success rate: {success_rate:.1%}")

    print("\n" + "-"*40)
    print("Neighborhood effectiveness ranking:")
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (name, rate) in enumerate(results, 1):
        print(f"{i}. {name}: {rate:.1%}")

    return results


def compare_algorithms(package='fastapi', iterations=30, runs=3):
    """Compare evolved MOVNS with MOEA/D"""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    print(f"Package: {package}, Iterations: {iterations}, Runs: {runs}")

    results = {'movns_evolved': [], 'movns_v2': [], 'moead': []}

    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")

        print("\nTesting Evolved MOVNS...")
        algo = MOVNSEvolved(package, archive_size=100, max_iterations=iterations)
        start = time.time()
        solutions = algo.run_evolved()
        exec_time = time.time() - start

        metrics = algo.get_metrics_history()
        if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0:
            final_hv = metrics['hypervolume'][-1]
        else:
            final_hv = 0

        results['movns_evolved'].append({
            'hv': final_hv,
            'time': exec_time,
            'solutions': len(solutions)
        })
        print(f"  HV: {final_hv:.4f}, Time: {exec_time:.1f}s, Solutions: {len(solutions)}")

        print("\nTesting MOVNS v2...")
        algo = MOVNS_V2(package, archive_size=100, max_iterations=iterations)
        start = time.time()
        solutions = algo.run()
        exec_time = time.time() - start

        metrics = algo.get_metrics_history()
        if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0:
            final_hv = metrics['hypervolume'][-1]
        else:
            final_hv = 0

        results['movns_v2'].append({
            'hv': final_hv,
            'time': exec_time,
            'solutions': len(solutions)
        })
        print(f"  HV: {final_hv:.4f}, Time: {exec_time:.1f}s, Solutions: {len(solutions)}")

        print("\nTesting MOEA/D...")
        algo = MOEAD_Normalized(package, pop_size=100, max_gen=iterations)
        start = time.time()
        solutions = algo.run()
        exec_time = time.time() - start

        metrics = algo.get_metrics_history()
        if metrics and 'hypervolume' in metrics and len(metrics['hypervolume']) > 0:
            final_hv = metrics['hypervolume'][-1]
        else:
            final_hv = 0

        results['moead'].append({
            'hv': final_hv,
            'time': exec_time,
            'solutions': len(solutions)
        })
        print(f"  HV: {final_hv:.4f}, Time: {exec_time:.1f}s, Solutions: {len(solutions)}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for algo_name in results:
        hvs = [r['hv'] for r in results[algo_name]]
        times = [r['time'] for r in results[algo_name]]
        solutions = [r['solutions'] for r in results[algo_name]]

        avg_hv = np.mean(hvs)
        std_hv = np.std(hvs)
        avg_time = np.mean(times)
        avg_solutions = np.mean(solutions)

        print(f"\n{algo_name.upper()}:")
        print(f"  HV: {avg_hv:.4f} ± {std_hv:.4f}")
        print(f"  Time: {avg_time:.1f}s")
        print(f"  Solutions: {avg_solutions:.1f}")

    evolved_hv = np.mean([r['hv'] for r in results['movns_evolved']])
    v2_hv = np.mean([r['hv'] for r in results['movns_v2']])
    moead_hv = np.mean([r['hv'] for r in results['moead']])

    print("\n" + "-"*40)
    print("Performance Comparison:")
    print(f"  Evolved vs v2: {(evolved_hv/v2_hv - 1)*100:+.1f}%")
    print(f"  Evolved vs MOEA/D: {(evolved_hv/moead_hv - 1)*100:+.1f}%")
    print(f"  Evolved as % of MOEA/D: {(evolved_hv/moead_hv)*100:.1f}%")

    if evolved_hv > v2_hv * 1.05:
        print("\n✓ SUCCESS: Evolved MOVNS improves over v2")

    if evolved_hv > moead_hv * 0.8:
        print("✓ COMPETITIVE: Evolved MOVNS achieves >80% of MOEA/D")

    return results


def main():
    """Main testing and evolution pipeline"""
    print("="*60)
    print("MOVNS EVOLUTION AND TESTING")
    print("="*60)
    print("Goal: Evolve MOVNS to improve HV performance")

    unit_results = unit_test_neighborhoods()

    comparison_results = compare_algorithms('fastapi', iterations=30, runs=2)

    print("\n" + "="*60)
    print("EVOLUTION INSIGHTS")
    print("="*60)

    print("\nKey improvements in Evolved MOVNS:")
    print("1. Adaptive neighborhood selection based on success rates")
    print("2. Enhanced MOBI/P with larger local search")
    print("3. Problem-specific neighborhoods (exploitation, exploration, refinement, diversification)")
    print("4. Elite archive for preserving best solutions")
    print("5. Adaptive shaking intensity based on archive fullness")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    evolved_hv = np.mean([r['hv'] for r in comparison_results['movns_evolved']])
    moead_hv = np.mean([r['hv'] for r in comparison_results['moead']])

    if evolved_hv < moead_hv * 0.7:
        print("\nFurther improvements needed:")
        print("- Consider collaborative VNS with multiple search threads")
        print("- Implement incremental objective evaluation")
        print("- Add learning mechanisms for parameter adaptation")
        print("- Design more problem-specific neighborhoods")
    else:
        print("\nEvolved MOVNS shows competitive performance")
        print("Continue refinement of successful strategies")

    with open('movns_evolution_results.json', 'w') as f:
        json.dump({
            'unit_tests': dict(unit_results),
            'comparison': comparison_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    print("\nResults saved to movns_evolution_results.json")

if __name__ == "__main__":
    main()