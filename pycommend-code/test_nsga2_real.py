"""
Real unit tests for NSGA-II v5 - No artificial fallbacks
Testing actual convergence and performance
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.append('src/optimizer')
from nsga2_v5 import NSGA2_V5


def test_real_convergence(package_name, expected_packages, pop_size=100, max_gen=50):
    """
    Test real NSGA-II convergence without shortcuts
    """
    print(f"\n{'='*70}")
    print(f"REAL CONVERGENCE TEST: {package_name.upper()}")
    print(f"Population: {pop_size}, Generations: {max_gen}")
    print(f"Expected packages: {expected_packages}")
    print(f"{'='*70}")

    nsga2 = NSGA2_V5(package_name, pop_size=pop_size, max_gen=max_gen)

    start_time = time.time()
    solutions = nsga2.run()
    execution_time = time.time() - start_time

    print(f"\nExecution time: {execution_time:.2f}s")
    print(f"Pareto front size: {len(solutions)}")

    if not solutions:
        print("ERROR: No solutions found")
        return {
            'success_rate': 0,
            'matches': [],
            'time': execution_time,
            'pareto_size': 0
        }

    best_by_colink = min(solutions, key=lambda x: x['objectives'][0])
    best_by_similarity = min(solutions, key=lambda x: x['objectives'][1])
    best_by_coherence = min(solutions, key=lambda x: x['objectives'][2])

    print("\nBest solutions by objective:")
    print(f"Best by F1 (colink): {-best_by_colink['objectives'][0]:.2f}")
    print(f"Best by F2 (similarity): {-best_by_similarity['objectives'][1]:.4f}")
    print(f"Best by F3 (coherence): {-best_by_coherence['objectives'][2]:.4f}")

    all_matches = set()
    for sol in solutions:
        indices = np.where(sol['chromosome'] == 1)[0]
        packages = [nsga2.package_names[idx] for idx in indices]
        matches = [pkg for pkg in expected_packages if pkg in packages]
        all_matches.update(matches)

    best_solution = best_by_colink
    indices = np.where(best_solution['chromosome'] == 1)[0]
    found_packages = [nsga2.package_names[idx] for idx in indices]

    print(f"\nBest solution packages ({len(found_packages)}):")
    for i, pkg in enumerate(found_packages[:15]):
        marker = " [EXPECTED]" if pkg in expected_packages else ""
        colink = nsga2.rel_matrix[nsga2.main_package_idx, indices[i]]
        sim = nsga2.sim_matrix[nsga2.main_package_idx, indices[i]]
        print(f"  {pkg}: colink={colink:.2f}, sim={sim:.4f}{marker}")

    direct_matches = [pkg for pkg in expected_packages if pkg in found_packages]
    success_rate = len(direct_matches) / len(expected_packages) * 100
    pareto_coverage = len(all_matches) / len(expected_packages) * 100

    print(f"\nDirect matches in best: {direct_matches}")
    print(f"All matches in Pareto: {list(all_matches)}")
    print(f"Success rate (best): {success_rate:.1f}%")
    print(f"Success rate (Pareto): {pareto_coverage:.1f}%")

    return {
        'success_rate': success_rate,
        'pareto_coverage': pareto_coverage,
        'matches': direct_matches,
        'all_matches': list(all_matches),
        'time': execution_time,
        'pareto_size': len(solutions),
        'objectives': best_solution['objectives'].tolist()
    }


def test_objective_values():
    """
    Test that all 4 objectives produce meaningful values
    """
    print(f"\n{'='*70}")
    print("OBJECTIVE VALUES TEST")
    print(f"{'='*70}")

    nsga2 = NSGA2_V5('numpy', pop_size=50, max_gen=20)

    test_cases = [
        ('empty', np.zeros(nsga2.n_packages, dtype=np.int8)),
        ('single', np.zeros(nsga2.n_packages, dtype=np.int8)),
        ('small', np.zeros(nsga2.n_packages, dtype=np.int8)),
        ('medium', np.zeros(nsga2.n_packages, dtype=np.int8)),
        ('large', np.zeros(nsga2.n_packages, dtype=np.int8))
    ]

    test_cases[1][1][nsga2.semantic_candidates[0]] = 1

    for idx in nsga2.cooccur_candidates[:3]:
        test_cases[2][1][idx] = 1

    for idx in nsga2.semantic_candidates[:7]:
        test_cases[3][1][idx] = 1

    for idx in nsga2.cluster_candidates[:15]:
        test_cases[4][1][idx] = 1

    for name, chromosome in test_cases:
        size = np.sum(chromosome)
        objectives = nsga2.evaluate_objectives(chromosome)

        print(f"\n{name.upper()} (size={size}):")

        if np.any(np.isinf(objectives)):
            print("  Contains infinity (invalid solution)")
        else:
            print(f"  F1 (colink): {-objectives[0]:.2f}")
            print(f"  F2 (similarity): {-objectives[1]:.4f}")
            print(f"  F3 (coherence): {-objectives[2]:.4f}")
            print(f"  F4 (size): {objectives[3]:.2f}")

            if size > 0:
                indices = np.where(chromosome == 1)[0]
                packages = [nsga2.package_names[idx] for idx in indices[:3]]
                print(f"  First packages: {packages}")

    return True


def test_initialization_quality():
    """
    Test quality of different initialization strategies
    """
    print(f"\n{'='*70}")
    print("INITIALIZATION QUALITY TEST")
    print(f"{'='*70}")

    nsga2 = NSGA2_V5('flask', pop_size=30, max_gen=10)
    expected = ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe']

    strategies = ['cooccur', 'semantic', 'cluster', 'hybrid']
    results = {}

    for strategy in strategies:
        matches_found = []
        objective_values = []

        for _ in range(10):
            chromosome = nsga2.smart_initialization(strategy)
            indices = np.where(chromosome == 1)[0]
            packages = [nsga2.package_names[idx] for idx in indices]

            matches = [pkg for pkg in expected if pkg in packages]
            matches_found.append(len(matches))

            objectives = nsga2.evaluate_objectives(chromosome)
            if not np.any(np.isinf(objectives)):
                objective_values.append(objectives)

        avg_matches = np.mean(matches_found)
        max_matches = max(matches_found)

        print(f"\n{strategy.upper()}:")
        print(f"  Avg matches: {avg_matches:.2f}/{len(expected)}")
        print(f"  Max matches: {max_matches}/{len(expected)}")

        if objective_values:
            avg_objectives = np.mean(objective_values, axis=0)
            print(f"  Avg F1: {-avg_objectives[0]:.2f}")
            print(f"  Avg F3 (coherence): {-avg_objectives[2]:.4f}")

        results[strategy] = {
            'avg_matches': avg_matches,
            'max_matches': max_matches
        }

    best_strategy = max(results, key=lambda k: results[k]['avg_matches'])
    print(f"\nBest initialization strategy: {best_strategy}")

    return results


def test_population_evolution():
    """
    Test how population evolves over generations
    """
    print(f"\n{'='*70}")
    print("POPULATION EVOLUTION TEST")
    print(f"{'='*70}")

    nsga2 = NSGA2_V5('pandas', pop_size=50, max_gen=30)
    expected = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'openpyxl']

    population = nsga2.initialize_population()

    generation_stats = []

    for gen in range(0, 30, 5):
        fronts = nsga2.fast_non_dominated_sort(population)

        if fronts and fronts[0]:
            pareto_size = len(fronts[0])

            best = min([population[i] for i in fronts[0]], key=lambda x: x['objectives'][0])
            indices = np.where(best['chromosome'] == 1)[0]
            packages = [nsga2.package_names[idx] for idx in indices]
            matches = [pkg for pkg in expected if pkg in packages]

            avg_f1 = np.mean([population[i]['objectives'][0] for i in fronts[0]])
            avg_f3 = np.mean([population[i]['objectives'][2] for i in fronts[0]])

            print(f"\nGeneration {gen}:")
            print(f"  Pareto size: {pareto_size}")
            print(f"  Matches: {matches}")
            print(f"  Avg F1: {-avg_f1:.2f}")
            print(f"  Avg F3: {-avg_f3:.4f}")

            generation_stats.append({
                'gen': gen,
                'pareto_size': pareto_size,
                'matches': len(matches),
                'avg_f1': -avg_f1,
                'avg_f3': -avg_f3
            })

        offspring = []
        for _ in range(nsga2.pop_size):
            parent1 = nsga2.tournament_selection(population)
            parent2 = nsga2.tournament_selection(population)

            child_chromosome = nsga2.crossover(parent1, parent2)
            child_chromosome = nsga2.mutation(child_chromosome)

            objectives = nsga2.evaluate_objectives(child_chromosome)

            offspring.append({
                'chromosome': child_chromosome,
                'objectives': objectives,
                'rank': None,
                'crowding_distance': 0
            })

        population.extend(offspring)
        fronts = nsga2.fast_non_dominated_sort(population)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= nsga2.pop_size:
                new_population.extend([population[i] for i in front])
            else:
                remaining = nsga2.pop_size - len(new_population)
                front_individuals = [population[i] for i in front]
                nsga2.crowding_distance_assignment(front_individuals)
                front_individuals.sort(key=lambda x: x['crowding_distance'], reverse=True)
                new_population.extend(front_individuals[:remaining])
                break

        population = new_population

    if generation_stats:
        improvement = generation_stats[-1]['avg_f1'] / generation_stats[0]['avg_f1']
        print(f"\nF1 improvement: {improvement:.2f}x")
        print(f"Final matches: {generation_stats[-1]['matches']}/{len(expected)}")

    return generation_stats


def main():
    """
    Run complete real tests
    """
    print("="*70)
    print("NSGA-II v5 REAL UNIT TESTS")
    print("No artificial fallbacks - Testing actual performance")
    print("="*70)

    all_results = {}

    print("\n1. OBJECTIVE VALUES TEST")
    test_objective_values()

    print("\n2. INITIALIZATION QUALITY TEST")
    init_results = test_initialization_quality()

    print("\n3. POPULATION EVOLUTION TEST")
    evolution = test_population_evolution()

    print("\n4. REAL CONVERGENCE TESTS")

    test_packages = {
        'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
        'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
        'requests': ['urllib3', 'certifi', 'idna', 'charset-normalizer', 'chardet']
    }

    for package, expected in test_packages.items():
        result = test_real_convergence(package, expected, pop_size=100, max_gen=50)
        all_results[package] = result

    print("\n" + "="*70)
    print("REAL PERFORMANCE SUMMARY")
    print("="*70)

    for package, result in all_results.items():
        print(f"\n{package.upper()}:")
        print(f"  Direct success: {result['success_rate']:.1f}%")
        print(f"  Pareto coverage: {result['pareto_coverage']:.1f}%")
        print(f"  Execution time: {result['time']:.2f}s")
        print(f"  Pareto size: {result['pareto_size']}")
        print(f"  Matches: {result['matches']}")

    avg_success = np.mean([r['success_rate'] for r in all_results.values()])
    avg_pareto = np.mean([r['pareto_coverage'] for r in all_results.values()])
    avg_time = np.mean([r['time'] for r in all_results.values()])

    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"Average success rate (best solution): {avg_success:.1f}%")
    print(f"Average success rate (Pareto front): {avg_pareto:.1f}%")
    print(f"Average execution time: {avg_time:.2f}s")

    print("\nComparison with previous versions:")
    print(f"  v1 (random): 4.0%")
    print(f"  v4 (weighted): 26.7%")
    print(f"  v5 (SBERT): {avg_success:.1f}% (best) / {avg_pareto:.1f}% (Pareto)")

    if avg_success > 26.7:
        improvement = avg_success / 26.7
        print(f"\nIMPROVEMENT: {improvement:.2f}x over v4")
    else:
        print(f"\nNO IMPROVEMENT: Need to debug convergence")

    return all_results


if __name__ == '__main__':
    results = main()