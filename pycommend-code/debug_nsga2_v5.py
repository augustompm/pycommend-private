"""
Debug NSGA-II v5 to find the issue
"""

import sys
import numpy as np

sys.path.append('src/optimizer')
from nsga2_v5 import NSGA2_V5


def debug_basic():
    """Debug basic functionality"""
    print("DEBUG: Basic Functionality Test")
    print("-"*50)

    nsga2 = NSGA2_V5('numpy', pop_size=10, max_gen=2)

    print("\n1. Initialize population:")
    population = nsga2.initialize_population()
    print(f"   Population size: {len(population)}")

    print("\n2. Check objectives:")
    for i, ind in enumerate(population[:3]):
        print(f"   Individual {i}: {ind['objectives']}")
        if np.any(np.isinf(ind['objectives'])):
            print(f"      WARNING: Contains infinity!")

    print("\n3. Non-dominated sort:")
    fronts = nsga2.fast_non_dominated_sort(population)
    print(f"   Number of fronts: {len(fronts)}")
    for i, front in enumerate(fronts[:3]):
        print(f"   Front {i}: {len(front)} individuals")

    print("\n4. Check if fronts[0] exists and has content:")
    if fronts:
        print(f"   fronts exists: True")
        print(f"   len(fronts): {len(fronts)}")
        if len(fronts) > 0:
            print(f"   fronts[0] exists: True")
            print(f"   len(fronts[0]): {len(fronts[0])}")
            if len(fronts[0]) > 0:
                print(f"   fronts[0] has content: True")
                print(f"   fronts[0]: {fronts[0]}")
            else:
                print(f"   fronts[0] is EMPTY - THIS IS THE PROBLEM")
    else:
        print(f"   fronts is EMPTY")

    print("\n5. Test dominance:")
    obj1 = population[0]['objectives']
    obj2 = population[1]['objectives']
    print(f"   Obj1: {obj1}")
    print(f"   Obj2: {obj2}")
    print(f"   Obj1 dominates Obj2: {nsga2.dominates(obj1, obj2)}")
    print(f"   Obj2 dominates Obj1: {nsga2.dominates(obj2, obj1)}")

    print("\n6. Check for invalid objectives:")
    invalid_count = 0
    for i, ind in enumerate(population):
        if np.any(np.isinf(ind['objectives'])):
            invalid_count += 1
            indices = np.where(ind['chromosome'] == 1)[0]
            size = len(indices)
            print(f"   Individual {i} invalid: size={size}")

    if invalid_count > 0:
        print(f"\n   WARNING: {invalid_count}/{len(population)} individuals have invalid objectives!")
        print(f"   This might cause empty Pareto fronts")

    print("\n7. Test evaluate_objectives directly:")
    test_chromosome = nsga2.smart_initialization('hybrid')
    test_size = np.sum(test_chromosome)
    test_objectives = nsga2.evaluate_objectives(test_chromosome)
    print(f"   Test chromosome size: {test_size}")
    print(f"   Test objectives: {test_objectives}")
    print(f"   Valid: {not np.any(np.isinf(test_objectives))}")

    if test_size < nsga2.min_size:
        print(f"   ERROR: Size {test_size} < min_size {nsga2.min_size}")

    return fronts


def debug_convergence():
    """Debug convergence issue"""
    print("\n\nDEBUG: Convergence Test")
    print("-"*50)

    nsga2 = NSGA2_V5('flask', pop_size=20, max_gen=5)
    population = nsga2.initialize_population()

    for gen in range(5):
        print(f"\nGeneration {gen}:")

        fronts = nsga2.fast_non_dominated_sort(population)
        print(f"  Fronts: {len(fronts)}")

        if fronts and len(fronts) > 0 and len(fronts[0]) > 0:
            print(f"  Pareto front size: {len(fronts[0])}")
            pareto = [population[i] for i in fronts[0]]
            best = min(pareto, key=lambda x: x['objectives'][0])
            print(f"  Best F1: {-best['objectives'][0]:.2f}")
        else:
            print(f"  WARNING: No valid Pareto front!")
            print(f"  fronts: {fronts}")

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

        population = population + offspring

        fronts = nsga2.fast_non_dominated_sort(population)

        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= nsga2.pop_size:
                new_population.extend([population[i] for i in front])
            else:
                break

        population = new_population

        if len(population) < nsga2.pop_size:
            print(f"  WARNING: Population shrunk to {len(population)}")


def main():
    print("="*60)
    print("NSGA-II v5 DEBUG")
    print("="*60)

    fronts = debug_basic()
    debug_convergence()

    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)

    if fronts and len(fronts) > 0 and len(fronts[0]) > 0:
        print("Pareto front is valid")
    else:
        print("ERROR: Empty Pareto front causing IndexError")
        print("Likely cause: All individuals have invalid (inf) objectives")
        print("Solution: Check min_size constraint and initialization")


if __name__ == '__main__':
    main()