"""
Debug test for MOVNS v15 Fast metrics
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v15_fast import MOVNS_V15_Fast
from evaluation.quality_metrics import QualityMetrics

print("Testing MOVNS v15 Fast metrics calculation")
print("="*60)

# Create optimizer with tracking enabled
movns = MOVNS_V15_Fast(
    'fastapi',
    archive_size=100,
    max_iterations=10,
    track_metrics=True
)

# Run optimizer
solutions = movns.run()

print(f"\nArchive size: {len(solutions)}")

# Check metrics history
metrics = movns.get_metrics_history()
if metrics:
    print(f"Metrics history exists: {list(metrics.keys())}")
    for key in metrics:
        print(f"  {key}: {len(metrics[key])} values")
        if metrics[key]:
            print(f"    Last value: {metrics[key][-1]:.4f}")
else:
    print("No metrics history!")

# Calculate metrics manually
qm = QualityMetrics()
objectives = []
for sol in solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    objectives.append(obj)
objectives = np.array(objectives)

# Normalize for HV calculation
obj_min = np.min(objectives, axis=0)
obj_max = np.max(objectives, axis=0)

normalized = []
for obj in objectives:
    norm_obj = (obj - obj_min) / (obj_max - obj_min + 1e-10)
    normalized.append(norm_obj)
normalized = np.array(normalized)

manual_hv = qm.hypervolume(normalized)
print(f"\nManual HV calculation: {manual_hv:.4f}")

# Check archive objectives
print(f"\nSample objectives (first 3):")
for i, obj in enumerate(objectives[:3]):
    print(f"  {i}: LU={-obj[0]:.0f}, SS={-obj[1]:.4f}, RSS={obj[2]:.1f}")