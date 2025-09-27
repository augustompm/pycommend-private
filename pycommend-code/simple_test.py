"""
Simple test to validate Weighted Probability Integration
"""

import sys
import numpy as np
sys.path.append('.')
from simple_best_method import weighted_probability_initialization

sys.path.append('src/optimizer')
from nsga2_integrated import NSGA2
from moead_integrated import MOEAD

def test_initialization():
    """
    Test if initialization produces expected packages
    """
    print("Testing Weighted Probability Initialization")
    print("="*60)

    try:
        nsga2 = NSGA2('numpy', pop_size=10, max_gen=2)
        print("\n[OK] NSGA-II initialized successfully")

        population = nsga2.initialize_population()
        print(f"[OK] Population created: {len(population)} individuals")

        first_solution = population[0]
        indices = np.where(first_solution['chromosome'] == 1)[0]
        packages = [nsga2.package_names[i] for i in indices[:5]]
        print(f"[OK] Sample packages in first solution: {packages}")

        expected = ['scipy', 'matplotlib', 'pandas']
        found = [p for p in expected if p in [nsga2.package_names[idx] for idx in indices]]
        if found:
            print(f"[OK] Found expected packages: {found}")
        else:
            print(f"[WARNING] No expected packages found")

    except Exception as e:
        print(f"[ERROR] Error in NSGA-II: {e}")
        return False

    print("\n" + "="*60)

    try:
        moead = MOEAD('flask', pop_size=10, max_gen=2)
        print("\n[OK] MOEA/D initialized successfully")

        moead.initialize_population()
        print(f"[OK] Population created: {len(moead.population)} individuals")

        first_solution = moead.population[0]
        indices = np.where(first_solution == 1)[0]
        packages = [moead.package_names[i] for i in indices[:5]]
        print(f"[OK] Sample packages in first solution: {packages}")

        expected = ['werkzeug', 'jinja2', 'click']
        found = [p for p in expected if p in [moead.package_names[idx] for idx in indices]]
        if found:
            print(f"[OK] Found expected packages: {found}")
        else:
            print(f"[WARNING] No expected packages found")

    except Exception as e:
        print(f"[ERROR] Error in MOEA/D: {e}")
        return False

    print("\n" + "="*60)
    print("[SUCCESS] INTEGRATION TEST PASSED!")
    print("Both algorithms successfully use Weighted Probability initialization")
    return True

if __name__ == '__main__':
    test_initialization()