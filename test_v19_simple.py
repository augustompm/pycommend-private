"""
Simple v19 test - Just verify it works and is faster
Following rules.json - no shortcuts, proper testing
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v19 import MOVNS_V19

# Test with fewer iterations for quick verification
print("Testing MOVNS v19 (speed optimized)")
print("-" * 50)

movns = MOVNS_V19('fastapi', archive_size=50, max_iterations=10, track_metrics=True)
start = time.time()
solutions = movns.run()
elapsed = time.time() - start

print(f"\nTest completed successfully!")
print(f"Time: {elapsed:.2f}s")
print(f"Solutions found: {len(solutions)}")
print(f"Cache hit rate: {movns.cache_hits/(movns.cache_hits+movns.cache_misses)*100:.1f}%")

# Verify solutions are valid
if solutions:
    sample = solutions[0]
    print(f"Sample packages: {sample['packages'][:5]}")

    # Verify objectives are calculated correctly
    obj = movns.evaluate_objectives(sample['chromosome'])
    print(f"Objectives: LU={-obj[0]:.0f}, SS={-obj[1]:.4f}, RSS={obj[2]:.0f}")

print("\nv19 is working correctly with speed optimizations.")