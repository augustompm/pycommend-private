"""
Convergence Detection Test for MOVNS Advanced vs MOEA/D
Detects when algorithms converge (10 iterations without improvement)
Proper HV calculation with normalization
"""

import numpy as np
import sys
import os
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


class ConvergenceDetector:
    """Detect convergence and compare algorithms at that point"""

    def __init__(self, no_improvement_threshold: int = 10):
        self.no_improvement_threshold = no_improvement_threshold
        self.qm = QualityMetrics()

    def calculate_proper_hv(self, solutions: List, algo) -> float:
        """Calculate hypervolume with proper normalization"""
        if not solutions:
            return 0.0

        objectives = []
        for sol in solutions:
            if isinstance(sol, dict) and 'chromosome' in sol:
                obj = algo.evaluate_objectives(sol['chromosome'])
            else:
                obj = algo.evaluate_objectives(sol)
            objectives.append(obj)

        objectives = np.array(objectives)

        normalized = []
        for obj in objectives:
            norm_obj = algo.normalize_objectives(obj)
            normalized.append(norm_obj)

        normalized = np.array(normalized)

        ref_point = np.array([0, 0, 1.0])
        hv = self.qm.hypervolume(normalized, ref_point)

        return hv

    def run_until_convergence(self, algo_class, package: str, **kwargs) -> Dict:
        """Run algorithm until convergence detected"""

        algo = algo_class(package, track_metrics=True, **kwargs)
        algo_name = algo_class.__name__

        print(f"\nRunning {algo_name} until convergence...")
        print(f"Convergence criteria: {self.no_improvement_threshold} iterations without HV improvement")

        results = {
            'algorithm': algo_name,
            'hv_history': [],
            'convergence_iteration': None,
            'final_hv': 0,
            'total_time': 0,
            'archive_sizes': [],
            'best_objectives': {'lu': 0, 'ss': 0, 'rss': float('inf')}
        }

        start_time = time.time()

        best_hv = 0
        no_improvement_count = 0
        iteration = 0
        max_iterations = kwargs.get('max_iterations', 100) if 'MOVNS' in algo_name else kwargs.get('max_gen', 100)

        if 'MOVNS' in algo_name:
            self.run_movns_with_convergence_detection(algo, results)
        else:
            self.run_moead_with_convergence_detection(algo, results)

        results['total_time'] = time.time() - start_time

        return results

    def run_movns_with_convergence_detection(self, algo, results: Dict):
        """Run MOVNS with iteration-by-iteration convergence detection"""

        best_hv = 0
        no_improvement_count = 0

        print(f"\nStarting {algo.__class__.__name__}...")
        print(f"Settings: {algo.max_iterations} max iterations, {algo.archive_limit} archive size")

        for iteration in range(algo.max_iterations):
            if len(algo.archive) > 0:
                if np.random.random() < algo.adaptive_params['exploration_rate']:
                    current_chromosome = algo.select_unexplored_solution()
                else:
                    archive_idx = np.random.choice(min(20, len(algo.archive)))
                    current_chromosome = algo.archive[archive_idx]['chromosome'].copy()
            else:
                current_chromosome = algo.select_unexplored_solution()

            if np.random.random() < 0.3 and iteration > 10:
                pareto_solutions = algo.pareto_local_search(current_chromosome)
                if pareto_solutions:
                    for ps in pareto_solutions[:5]:
                        algo.pareto_queue.append(ps)
                        ps_obj = algo.evaluate_objectives(ps)
                        algo.update_archive(ps, ps_obj)
                    current_chromosome = pareto_solutions[0]

            improved_solution = algo.iterated_local_search(current_chromosome, iterations=5)

            k = algo.adaptive_neighborhood_selection()
            x_prime = algo.adaptive_perturbation(improved_solution,
                                                algo.adaptive_params['perturbation_strength'] * (k + 1) / 6)
            x_local = algo.aggressive_local_search(x_prime)

            current_obj = algo.evaluate_objectives(improved_solution)
            local_obj = algo.evaluate_objectives(x_local)

            if algo.simulated_annealing_accept(current_obj, local_obj):
                final_solution = x_local
                final_obj = local_obj
                algo.update_learning_rates(k, True)
            else:
                final_solution = improved_solution
                final_obj = current_obj
                algo.update_learning_rates(k, False)

            algo.update_archive(final_solution, final_obj)

            if iteration % 2 == 0:
                current_hv = self.calculate_proper_hv(algo.archive, algo)
                results['hv_history'].append(current_hv)
                results['archive_sizes'].append(len(algo.archive))

                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improvement_count = 0
                    print(f"  Iteration {iteration}: HV={current_hv:.4f} ↑, Archive={len(algo.archive)}")
                else:
                    no_improvement_count += 1
                    if iteration % 5 == 0:
                        print(f"  Iteration {iteration}: HV={current_hv:.4f}, No improvement: {no_improvement_count}")

                if no_improvement_count >= self.no_improvement_threshold:
                    results['convergence_iteration'] = iteration
                    results['final_hv'] = current_hv
                    print(f"\nConvergence detected at iteration {iteration}")
                    print(f"Final HV: {current_hv:.4f}")
                    break

        for sol in algo.archive:
            obj = algo.evaluate_objectives(sol['chromosome'])
            if -obj[0] > results['best_objectives']['lu']:
                results['best_objectives']['lu'] = -obj[0]
            if -obj[1] > results['best_objectives']['ss']:
                results['best_objectives']['ss'] = -obj[1]
            if obj[2] < results['best_objectives']['rss']:
                results['best_objectives']['rss'] = obj[2]

    def run_moead_with_convergence_detection(self, algo, results: Dict):
        """Run MOEA/D with generation-by-generation convergence detection"""

        best_hv = 0
        no_improvement_count = 0

        print("\nInitializing population...")
        algo.initialize_population()

        print(f"Starting evolution with {algo.max_gen} max generations...")

        for generation in range(algo.max_gen):
            for i in range(algo.pop_size):
                if np.random.random() < algo.delta:
                    neighbor_indices = algo.neighborhood[i]
                else:
                    neighbor_indices = list(range(algo.pop_size))

                k = np.random.choice(neighbor_indices)
                l = np.random.choice(neighbor_indices)
                while l == k:
                    l = np.random.choice(neighbor_indices)

                offspring = algo.genetic_operator(
                    algo.population[k]['chromosome'],
                    algo.population[l]['chromosome']
                )

                offspring = algo.repair(offspring)
                y_obj = algo.evaluate_objectives(offspring)

                algo.update_reference(y_obj)
                algo.update_neighbors(offspring, y_obj, neighbor_indices)
                algo.update_external_population(offspring, y_obj)

            if generation % 2 == 0:
                current_hv = self.calculate_proper_hv(algo.external_population, algo)
                results['hv_history'].append(current_hv)
                results['archive_sizes'].append(len(algo.external_population))

                if current_hv > best_hv:
                    best_hv = current_hv
                    no_improvement_count = 0
                    print(f"  Generation {generation}: HV={current_hv:.4f} ↑, Archive={len(algo.external_population)}")
                else:
                    no_improvement_count += 1
                    if generation % 5 == 0:
                        print(f"  Generation {generation}: HV={current_hv:.4f}, No improvement: {no_improvement_count}")

                if no_improvement_count >= self.no_improvement_threshold:
                    results['convergence_iteration'] = generation
                    results['final_hv'] = current_hv
                    print(f"\nConvergence detected at generation {generation}")
                    print(f"Final HV: {current_hv:.4f}")
                    break

        for sol in algo.external_population:
            obj = algo.evaluate_objectives(sol['chromosome'])
            if -obj[0] > results['best_objectives']['lu']:
                results['best_objectives']['lu'] = -obj[0]
            if -obj[1] > results['best_objectives']['ss']:
                results['best_objectives']['ss'] = -obj[1]
            if obj[2] < results['best_objectives']['rss']:
                results['best_objectives']['rss'] = obj[2]


def main():
    """Main convergence comparison"""

    print("="*70)
    print("CONVERGENCE DETECTION TEST")
    print("="*70)
    print("\nComparing MOVNS Advanced and MOEA/D Normalized")
    print("Convergence criterion: 10 iterations without HV improvement")

    detector = ConvergenceDetector(no_improvement_threshold=10)
    package = 'fastapi'

    print("\n" + "-"*70)
    print("1. MOVNS ADVANCED")
    print("-"*70)

    movns_results = detector.run_until_convergence(
        MOVNS_Advanced,
        package,
        archive_size=100,
        max_iterations=100
    )

    print("\n" + "-"*70)
    print("2. MOEA/D NORMALIZED")
    print("-"*70)

    moead_results = detector.run_until_convergence(
        MOEAD_Normalized,
        package,
        pop_size=100,
        max_gen=100
    )

    print("\n" + "="*70)
    print("CONVERGENCE COMPARISON RESULTS")
    print("="*70)

    print(f"\n1. Convergence Points:")
    print(f"   MOVNS Advanced: Iteration {movns_results['convergence_iteration']}")
    print(f"   MOEA/D:         Generation {moead_results['convergence_iteration']}")

    print(f"\n2. Final Hypervolume at Convergence:")
    print(f"   MOVNS Advanced: {movns_results['final_hv']:.4f}")
    print(f"   MOEA/D:         {moead_results['final_hv']:.4f}")

    if moead_results['final_hv'] > 0:
        ratio = movns_results['final_hv'] / moead_results['final_hv']
        print(f"   Ratio: {ratio*100:.1f}%")

        if ratio > 1.0:
            print(f"   MOVNS Advanced superior by {(ratio-1)*100:.1f}%")
        else:
            print(f"   MOEA/D superior by {(1-ratio)*100:.1f}%")

    print(f"\n3. Archive Size at Convergence:")
    if movns_results['archive_sizes']:
        print(f"   MOVNS Advanced: {movns_results['archive_sizes'][-1]} solutions")
    if moead_results['archive_sizes']:
        print(f"   MOEA/D:         {moead_results['archive_sizes'][-1]} solutions")

    print(f"\n4. Time to Convergence:")
    print(f"   MOVNS Advanced: {movns_results['total_time']:.1f}s")
    print(f"   MOEA/D:         {moead_results['total_time']:.1f}s")

    print(f"\n5. Best Objectives at Convergence:")
    print(f"   MOVNS Advanced:")
    print(f"     LU: {movns_results['best_objectives']['lu']:.0f}")
    print(f"     SS: {movns_results['best_objectives']['ss']:.4f}")
    print(f"     RSS: {movns_results['best_objectives']['rss']}")
    print(f"   MOEA/D:")
    print(f"     LU: {moead_results['best_objectives']['lu']:.0f}")
    print(f"     SS: {moead_results['best_objectives']['ss']:.4f}")
    print(f"     RSS: {moead_results['best_objectives']['rss']}")

    print(f"\n6. HV Progress (last 5 measurements):")
    if len(movns_results['hv_history']) >= 5:
        print(f"   MOVNS: {movns_results['hv_history'][-5:]}")
    if len(moead_results['hv_history']) >= 5:
        print(f"   MOEA/D: {moead_results['hv_history'][-5:]}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if movns_results['convergence_iteration'] and moead_results['convergence_iteration']:
        if movns_results['convergence_iteration'] < moead_results['convergence_iteration']:
            print(f"MOVNS converges faster ({movns_results['convergence_iteration']} vs {moead_results['convergence_iteration']} iterations)")
        else:
            print(f"MOEA/D converges faster ({moead_results['convergence_iteration']} vs {movns_results['convergence_iteration']} iterations)")

    if movns_results['final_hv'] > moead_results['final_hv']:
        print(f"MOVNS achieves better final quality (HV={movns_results['final_hv']:.4f})")
    else:
        print(f"MOEA/D achieves better final quality (HV={moead_results['final_hv']:.4f})")


if __name__ == "__main__":
    main()