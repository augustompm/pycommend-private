"""
Quick calibration test for MOVNS v4
Find configuration that beats MOEA/D
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v4 import MOVNS_V4
from optimizer.moead_normalized import MOEAD_Normalized

def test_config(archive_size, mobi_p, iterations=30):
    """Test a specific configuration"""
    print(f"\nTesting: Archive={archive_size}, MOBI/P={mobi_p}")

    # Test MOVNS v4
    movns = MOVNS_V4('fastapi',
                    archive_size=archive_size,
                    secondary_archive_size=50,
                    max_iterations=iterations,
                    mobi_p_neighbors=mobi_p,
                    shaking_base_intensity=2,
                    shaking_max_intensity=8,
                    adaptive=True,
                    track_metrics=True,
                    min_no_improvement=10)

    start = time.time()
    movns_sol = movns.run()
    movns_time = time.time() - start

    movns_metrics = movns.get_metrics_history()
    movns_hv = 0
    if movns_metrics and 'hypervolume' in movns_metrics:
        hv = movns_metrics['hypervolume']
        if len(hv) > 0:
            movns_hv = hv[-1]

    print(f"  MOVNS v4: HV={movns_hv:.4f}, Time={movns_time:.1f}s")

    return movns_hv

def main():
    print("="*60)
    print("MOVNS v4 QUICK CALIBRATION")
    print("="*60)

    # Test MOEA/D baseline
    print("\nBaseline: MOEA/D Normalized")
    moead = MOEAD_Normalized('fastapi', pop_size=50, max_gen=30, track_metrics=True)
    start = time.time()
    moead_sol = moead.run()
    moead_time = time.time() - start

    moead_metrics = moead.get_metrics_history()
    moead_hv = 0
    if moead_metrics and 'hypervolume' in moead_metrics:
        hv = moead_metrics['hypervolume']
        if len(hv) > 0:
            moead_hv = hv[-1]
            print(f"  MOEA/D: HV={moead_hv:.4f}, Time={moead_time:.1f}s")

    print(f"\nTarget: Beat HV={moead_hv:.4f}")
    print("-"*40)

    # Test configurations
    configs = [
        (150, 50),   # Large archive, moderate MOBI/P
        (200, 60),   # Very large archive, high MOBI/P
        (150, 70),   # Large archive, very high MOBI/P
        (100, 80),   # Moderate archive, extreme MOBI/P
        (175, 55),   # Balanced configuration
    ]

    best_config = None
    best_hv = 0

    for archive, mobi_p in configs:
        hv = test_config(archive, mobi_p, iterations=30)

        if hv > best_hv:
            best_hv = hv
            best_config = (archive, mobi_p)

        if hv > moead_hv:
            print(f"  ✓ BEATS MOEA/D by {(hv/moead_hv - 1)*100:.1f}%")

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    if best_hv > moead_hv:
        improvement = (best_hv / moead_hv - 1) * 100
        print(f"\n✓ SUCCESS: MOVNS v4 beats MOEA/D by {improvement:.1f}%")
        print(f"  Best config: Archive={best_config[0]}, MOBI/P={best_config[1]}")
        print(f"  MOVNS v4 HV: {best_hv:.4f}")
        print(f"  MOEA/D HV: {moead_hv:.4f}")
    else:
        deficit = (moead_hv / best_hv - 1) * 100
        print(f"\n✗ MOVNS v4 is {deficit:.1f}% behind MOEA/D")
        print(f"  Best attempt: Archive={best_config[0]}, MOBI/P={best_config[1]}")
        print(f"  MOVNS v4 HV: {best_hv:.4f}")
        print(f"  MOEA/D HV: {moead_hv:.4f}")

        print("\nRecommendations:")
        print("  1. Increase iterations to 50-75")
        print("  2. Try archive size 250+")
        print("  3. Increase MOBI/P to 100+")
        print("  4. Adjust neighborhood strategies")

if __name__ == "__main__":
    main()