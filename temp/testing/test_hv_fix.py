"""
Test and fix hypervolume calculation
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from evaluation.quality_metrics import QualityMetrics

# Test with sample objectives (LU, SS, RSS)
# LU and SS are negative (to maximize), RSS is positive (to minimize)
test_objectives = np.array([
    [-1000, -0.8, 5],    # Good solution
    [-800, -0.7, 6],     # Medium solution
    [-500, -0.6, 7],     # Another solution
    [-300, -0.5, 8],     # Worse solution
])

print("Test objectives (LU, SS, RSS):")
print(test_objectives)

# Calculate HV with different approaches
qm = QualityMetrics()

# Test 1: Default reference point
print("\n1. Default reference point:")
try:
    hv = qm.hypervolume(test_objectives)
    print(f"HV = {hv}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Specified reference point
print("\n2. With ref_point [0, 0, 15]:")
try:
    hv = qm.hypervolume(test_objectives, ref_point=[0, 0, 15])
    print(f"HV = {hv}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Transform objectives to all positive for minimization
print("\n3. Transform to all positive minimization:")
transformed = test_objectives.copy()
transformed[:, 0] = -transformed[:, 0]  # LU: negate to make positive (lower is better)
transformed[:, 1] = -transformed[:, 1]  # SS: negate to make positive (lower is better)
print("Transformed objectives:")
print(transformed)

try:
    qm2 = QualityMetrics()
    hv = qm2.hypervolume(transformed, ref_point=[2000, 1, 15])
    print(f"HV = {hv}")
except Exception as e:
    print(f"Error: {e}")