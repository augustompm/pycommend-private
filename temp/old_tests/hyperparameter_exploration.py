"""
Hyperparameter exploration for MOVNS v4
Systematic grid search to find optimal configuration
"""

import numpy as np
import sys
import os
import time
import itertools
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v4 import MOVNS_V4
from optimizer.moead_normalized import MOEAD_Normalized

def test_configuration(config, package='fastapi', iterations=50):
    """Test a specific configuration"""
    start = time.time()

    movns = MOVNS_V4(
        package,
        archive_size=config['archive_size'],
        secondary_archive_size=config['secondary_size'],
        max_iterations=iterations,
        mobi_p_neighbors=config['mobi_p'],
        shaking_base_intensity=config['shaking_base'],
        shaking_max_intensity=config['shaking_max'],
        adaptive=config['adaptive'],
        track_metrics=True,
        min_no_improvement=15
    )

    solutions = movns.run()
    exec_time = time.time() - start

    metrics = movns.get_metrics_history()

    result = {
        'config': config,
        'solutions': len(solutions),
        'time': exec_time
    }

    if metrics and 'hypervolume' in metrics:
        hv = metrics['hypervolume']
        if len(hv) > 0:
            result['initial_hv'] = hv[0]
            result['final_hv'] = hv[-1]
            result['max_hv'] = max(hv)
            result['improvement'] = ((hv[-1] - hv[0]) / hv[0] * 100) if hv[0] > 0 else 0

    return result

def grid_search_phase1():
    """Phase 1: Coarse grid search"""
    print("="*60)
    print("PHASE 1: COARSE GRID SEARCH")
    print("="*60)

    # Define parameter grid
    param_grid = {
        'archive_size': [100, 150, 200],
        'secondary_size': [30, 50],
        'mobi_p': [30, 50],
        'shaking_base': [2],
        'shaking_max': [8],
        'adaptive': [True]
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"Testing {len(configs)} configurations...")
    results = []

    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}/{len(configs)}:")
        print(f"  Archive: {config['archive_size']}, Secondary: {config['secondary_size']}")
        print(f"  MOBI/P: {config['mobi_p']}, Adaptive: {config['adaptive']}")

        result = test_configuration(config, iterations=30)  # Quick test
        results.append(result)

        print(f"  Result: HV={result.get('final_hv', 0):.4f}, Time={result['time']:.1f}s")

    # Sort by final HV
    results.sort(key=lambda x: x.get('final_hv', 0), reverse=True)

    print("\n" + "="*60)
    print("PHASE 1 RESULTS - TOP 5")
    print("="*60)

    for i, result in enumerate(results[:5]):
        config = result['config']
        print(f"\n{i+1}. HV={result.get('final_hv', 0):.4f}")
        print(f"   Archive={config['archive_size']}, Secondary={config['secondary_size']}")
        print(f"   MOBI/P={config['mobi_p']}, Adaptive={config['adaptive']}")

    return results[0]['config']  # Return best config

def fine_tune_phase2(base_config):
    """Phase 2: Fine tuning around best config"""
    print("\n" + "="*60)
    print("PHASE 2: FINE TUNING")
    print("="*60)
    print(f"Base config: {base_config}")

    # Fine tune around best config
    param_variations = {
        'archive_size': [base_config['archive_size'] - 25,
                        base_config['archive_size'],
                        base_config['archive_size'] + 25],
        'mobi_p': [base_config['mobi_p'] - 10,
                  base_config['mobi_p'],
                  base_config['mobi_p'] + 10],
        'shaking_base': [1, 2, 3],
        'shaking_max': [8, 10, 12]
    }

    # Fix other parameters
    fixed_params = {
        'secondary_size': base_config['secondary_size'],
        'adaptive': True
    }

    keys = param_variations.keys()
    values = param_variations.values()
    configs = []

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config.update(fixed_params)
        configs.append(config)

    print(f"Fine tuning with {len(configs)} configurations...")
    results = []

    for i, config in enumerate(configs):
        if i % 5 == 0:
            print(f"  Testing {i+1}/{len(configs)}...")

        result = test_configuration(config, iterations=40)
        results.append(result)

    # Sort by final HV
    results.sort(key=lambda x: x.get('final_hv', 0), reverse=True)

    print("\n" + "="*60)
    print("PHASE 2 RESULTS - TOP 3")
    print("="*60)

    for i, result in enumerate(results[:3]):
        config = result['config']
        print(f"\n{i+1}. HV={result.get('final_hv', 0):.4f}, Improvement={result.get('improvement', 0):.1f}%")
        print(f"   Archive={config['archive_size']}, MOBI/P={config['mobi_p']}")
        print(f"   Shaking: {config['shaking_base']}-{config['shaking_max']}")

    return results[0]['config']

def final_validation(best_config):
    """Phase 3: Final validation against MOEA/D"""
    print("\n" + "="*60)
    print("PHASE 3: FINAL VALIDATION")
    print("="*60)
    print(f"Best configuration: {best_config}")

    iterations = 75
    runs = 3

    movns_results = []
    moead_results = []

    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")

        # Test MOVNS v4 with best config
        print("Testing MOVNS v4...")
        result = test_configuration(best_config, iterations=iterations)
        movns_results.append(result.get('final_hv', 0))
        print(f"  MOVNS v4 HV: {result.get('final_hv', 0):.4f}")

        # Test MOEA/D
        print("Testing MOEA/D Normalized...")
        start = time.time()
        moead = MOEAD_Normalized('fastapi', pop_size=50, max_gen=iterations,
                                track_metrics=True)
        solutions = moead.run()
        moead_time = time.time() - start

        metrics = moead.get_metrics_history()
        if metrics and 'hypervolume' in metrics:
            hv = metrics['hypervolume']
            if len(hv) > 0:
                moead_results.append(hv[-1])
                print(f"  MOEA/D HV: {hv[-1]:.4f}")

    # Statistics
    movns_mean = np.mean(movns_results)
    movns_std = np.std(movns_results)
    moead_mean = np.mean(moead_results)
    moead_std = np.std(moead_results)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nMOVNS v4 (calibrated):")
    print(f"  Mean HV: {movns_mean:.4f} ± {movns_std:.4f}")
    print(f"  Best HV: {max(movns_results):.4f}")

    print(f"\nMOEA/D Normalized:")
    print(f"  Mean HV: {moead_mean:.4f} ± {moead_std:.4f}")
    print(f"  Best HV: {max(moead_results):.4f}")

    if movns_mean > moead_mean:
        improvement = (movns_mean / moead_mean - 1) * 100
        print(f"\n✓ SUCCESS: MOVNS v4 beats MOEA/D by {improvement:.1f}%!")
    else:
        deficit = (moead_mean / movns_mean - 1) * 100
        print(f"\n✗ MOVNS v4 is {deficit:.1f}% behind MOEA/D")
        print(f"  Consider further tuning or longer runs")

    return movns_mean, moead_mean, best_config

def main():
    """Main hyperparameter exploration"""
    print("="*60)
    print("MOVNS v4 HYPERPARAMETER EXPLORATION")
    print("="*60)
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("This will take approximately 10-15 minutes")

    # Phase 1: Coarse grid search
    best_coarse = grid_search_phase1()

    # Phase 2: Fine tuning
    best_fine = fine_tune_phase2(best_coarse)

    # Phase 3: Final validation
    movns_hv, moead_hv, final_config = final_validation(best_fine)

    # Save results
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}")

    print(f"\nOptimal MOVNS v4 Configuration:")
    for key, value in final_config.items():
        print(f"  {key}: {value}")

    print(f"\nFinal Performance:")
    print(f"  MOVNS v4: {movns_hv:.4f}")
    print(f"  MOEA/D: {moead_hv:.4f}")

    if movns_hv > moead_hv:
        print(f"\n✓ MOVNS v4 successfully calibrated to beat MOEA/D!")
    else:
        print(f"\n✗ Further optimization needed")
        print(f"  Consider:")
        print(f"  - Increasing archive size further (250+)")
        print(f"  - More MOBI/P neighbors (70+)")
        print(f"  - Longer runs (100+ iterations)")
        print(f"  - Problem-specific neighborhoods")

if __name__ == "__main__":
    main()