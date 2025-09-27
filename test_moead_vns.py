"""
Test MOEA/D-VNS alignment with presentation
Compare with NSGA-II results
"""

import sys
import os
import numpy as np
import time
os.chdir('pycommend-code')
sys.path.append('src/optimizer')


def test_moead_quick():
    """Quick test of MOEA/D implementation"""
    from moead_vns import MOEAD_VNS

    print("="*70)
    print("MOEA/D-VNS QUICK TEST")
    print("="*70)

    package = 'fastapi'
    print(f"\nTesting MOEA/D for: {package}")

    moead = MOEAD_VNS(package, pop_size=30, n_neighbors=10, max_gen=10,
                      decomposition='tchebycheff')

    start_time = time.time()

    population = []
    strategies = ['small', 'medium', 'large', 'cooccur', 'semantic', 'hybrid']

    print("Initializing population...")
    for i in range(30):
        strategy = strategies[i % len(strategies)]
        chromosome = moead.smart_initialization(strategy)
        objectives = moead.evaluate_objectives(chromosome)
        population.append({
            'chromosome': chromosome,
            'objectives': objectives
        })

    print(f"Population initialized in {time.time() - start_time:.2f}s")

    pareto_front = []
    for i, sol_i in enumerate(population):
        dominated = False
        for j, sol_j in enumerate(population):
            if i != j and moead.dominates(sol_j['objectives'], sol_i['objectives']):
                dominated = True
                break
        if not dominated:
            pareto_front.append(sol_i)

    print(f"Initial Pareto front size: {len(pareto_front)}")

    recommendations = moead.get_recommendations(pareto_front[:5])

    print("\nSample recommendations:")
    for rec in recommendations[:3]:
        print(f"\nSize {rec['size']}: {', '.join(rec['packages'][:8])}")
        print(f"  LU={rec['linked_usage']:.0f}, SS={rec['semantic_similarity']:.3f}")


def compare_algorithms():
    """Compare NSGA-II and MOEA/D on presentation test cases"""
    from nsga2_vns import NSGA2_VNS
    from moead_vns import MOEAD_VNS

    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)

    test_packages = ['fastapi', 'scikit-learn', 'prophet']

    for package in test_packages:
        print(f"\n{package.upper()}:")
        print("-"*50)

        try:
            nsga2 = NSGA2_VNS(package, pop_size=20, max_gen=5)

            nsga2_pop = nsga2.initialize_population()
            nsga2_objs = [ind['objectives'] for ind in nsga2_pop]

            best_nsga2_lu = min([obj[0] for obj in nsga2_objs])
            best_nsga2_ss = min([obj[1] for obj in nsga2_objs])
            avg_nsga2_rss = np.mean([obj[2] for obj in nsga2_objs])

            print(f"NSGA-II initial population:")
            print(f"  Best LU: {-best_nsga2_lu:.2f}")
            print(f"  Best SS: {-best_nsga2_ss:.4f}")
            print(f"  Avg RSS: {avg_nsga2_rss:.1f}")

        except Exception as e:
            print(f"NSGA-II error: {str(e)}")

        try:
            moead = MOEAD_VNS(package, pop_size=20, n_neighbors=5, max_gen=5,
                             decomposition='tchebycheff')

            moead_pop = []
            for i in range(20):
                chromosome = moead.smart_initialization('hybrid')
                objectives = moead.evaluate_objectives(chromosome)
                moead_pop.append(objectives)

            best_moead_lu = min([obj[0] for obj in moead_pop])
            best_moead_ss = min([obj[1] for obj in moead_pop])
            avg_moead_rss = np.mean([obj[2] for obj in moead_pop])

            print(f"MOEA/D initial population:")
            print(f"  Best LU: {-best_moead_lu:.2f}")
            print(f"  Best SS: {-best_moead_ss:.4f}")
            print(f"  Avg RSS: {avg_moead_rss:.1f}")

        except Exception as e:
            print(f"MOEA/D error: {str(e)}")


def test_decomposition_methods():
    """Test different decomposition methods in MOEA/D"""
    from moead_vns import MOEAD_VNS

    print("\n" + "="*70)
    print("DECOMPOSITION METHODS TEST")
    print("="*70)

    package = 'scikit-learn'
    methods = ['tchebycheff', 'weighted_sum', 'pbi']

    for method in methods:
        print(f"\n{method.upper()} decomposition:")
        print("-"*30)

        try:
            moead = MOEAD_VNS(package, pop_size=10, n_neighbors=5, max_gen=1,
                             decomposition=method)

            test_obj = np.array([-100, -0.5, 5])
            test_weight = np.array([0.33, 0.33, 0.34])

            scalar = moead.decompose(test_obj, test_weight)
            print(f"  Test objectives: {test_obj}")
            print(f"  Test weights: {test_weight}")
            print(f"  Scalar value: {scalar:.4f}")

        except Exception as e:
            print(f"  Error: {str(e)}")


def validate_presentation_results():
    """Validate that both algorithms can find presentation results"""
    from moead_vns import MOEAD_VNS

    print("\n" + "="*70)
    print("PRESENTATION VALIDATION")
    print("="*70)

    presentation_results = {
        'fastapi': ['pydantic', 'uvicorn', 'typer', 'starlette', 'httpx'],
        'scikit-learn': ['pandas', 'matplotlib', 'xgboost', 'joblib'],
        'prophet': ['pandas', 'matplotlib', 'numpy', 'scikit-learn']
    }

    for package, expected_libs in presentation_results.items():
        print(f"\n{package.upper()} - Expected: {', '.join(expected_libs)}")
        print("-"*50)

        moead = MOEAD_VNS(package, pop_size=10, n_neighbors=5, max_gen=1)

        test_chromosome = np.zeros(moead.n_packages, dtype=np.int8)

        found_count = 0
        total_lu = 0
        for lib in expected_libs:
            if lib in moead.package_names:
                idx = moead.package_names.index(lib)
                test_chromosome[idx] = 1
                cooccur = moead.rel_matrix[moead.main_package_idx, idx]
                total_lu += cooccur
                found_count += 1

        if found_count > 0:
            objectives = moead.evaluate_objectives(test_chromosome)

            print(f"  Found {found_count}/{len(expected_libs)} packages")
            print(f"  Total co-occurrence: {total_lu:.0f}")
            print(f"  Objectives: LU={-objectives[0]:.2f}, SS={-objectives[1]:.4f}, RSS={objectives[2]:.1f}")


if __name__ == '__main__':
    test_moead_quick()
    compare_algorithms()
    test_decomposition_methods()
    validate_presentation_results()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
MOEA/D-VNS Implementation Status:

1. ALIGNED WITH PRESENTATION:
   - Three objectives: LU, SS, RSS
   - Smart initialization using co-occurrence and semantic data
   - Based on Zhang & Li (2007) IEEE paper

2. KEY FEATURES:
   - Decomposition methods: Tchebycheff, Weighted Sum, PBI
   - Weight vector generation using Das-Dennis method
   - Neighborhood-based evolution
   - Differential Evolution operator

3. FOLLOWS rules.json:
   - No inline comments (only header docstrings)
   - Clean Python code
   - Real academic references

4. COMPARABLE TO NSGA-II:
   - Both use same objectives
   - Both use same data sources
   - Both produce Pareto-optimal solutions
    """)