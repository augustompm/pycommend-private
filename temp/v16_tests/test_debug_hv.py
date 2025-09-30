"""
Debug do HV do MOVNS Final
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_final import MOVNS_Final

print("Debug HV do MOVNS Final")
print("="*60)

movns = MOVNS_Final('fastapi', archive_size=100, max_iterations=5, track_metrics=True)
solutions = movns.run()

print(f"\nSoluções: {len(solutions)}")

# Verificar get_metrics_history
metrics = movns.get_metrics_history()
print(f"\nget_metrics_history retorna: {metrics}")

if metrics:
    print(f"Keys: {list(metrics.keys())}")
    if 'hypervolume' in metrics:
        print(f"HV list: {metrics['hypervolume']}")
        if len(metrics['hypervolume']) > 0:
            print(f"Último HV: {metrics['hypervolume'][-1]:.4f}")
else:
    print("get_metrics_history retornou None!")

# Verificar se metrics_history existe
print(f"\nTem metrics_history? {hasattr(movns, 'metrics_history')}")
if hasattr(movns, 'metrics_history'):
    print(f"metrics_history: {movns.metrics_history}")