"""
Quick test of MOVNS performance after optimizations
"""

import sys
import time

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2


print("="*80)
print("MOVNS PERFORMANCE TEST AFTER OPTIMIZATIONS")
print("="*80)

# Test MOVNS with 1 iteration
print("\n1. Testing MOVNS (1 iteration):")
print("-"*60)
start = time.time()
movns = MOVNS_VNS('numpy', archive_size=20, max_iterations=1, track_metrics=False)
movns_sols = movns.run()
movns_time = time.time() - start

print(f"✓ MOVNS completed")
print(f"  Solutions: {len(movns_sols)}")
print(f"  Time: {movns_time:.2f}s")

if movns_time < 2:
    print(f"  ✓ GOOD: Under 2 seconds for 1 iteration")
else:
    print(f"  ⚠ SLOW: {movns_time:.1f}s is too slow for practical use")

# Compare with NSGA-II
print("\n2. Reference: NSGA-II (1 generation):")
print("-"*60)
start = time.time()
nsga2 = NSGA2('numpy', pop_size=20, max_gen=1)
nsga2_sols = nsga2.run()
nsga2_time = time.time() - start

print(f"✓ NSGA-II completed")
print(f"  Solutions: {len(nsga2_sols)}")
print(f"  Time: {nsga2_time:.2f}s")

# Summary
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"MOVNS (1 iter):  {movns_time:.2f}s")
print(f"NSGA-II (1 gen): {nsga2_time:.2f}s")

if movns_time > 0:
    ratio = movns_time / nsga2_time
    print(f"Ratio: MOVNS is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than NSGA-II")

if movns_time < 2:
    print("\n✓ MOVNS is now practical for real-time recommendations!")
    print("  A full 30-iteration run should take ~1 minute")
else:
    print(f"\n⚠ MOVNS still needs optimization")
    print(f"  Current: {movns_time:.1f}s per iteration")
    print(f"  Target: <1s per iteration for practical use")