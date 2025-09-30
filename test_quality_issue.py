"""
Teste para entender por que MOVNS v15 Fast gera soluções fracas
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v15_fast import MOVNS_V15_Fast
from optimizer.movns_v2 import MOVNS_V2

print("="*70)
print("Comparar v15 Fast vs v2 base")
print("="*70)

# Teste v15 Fast
print("\n1. MOVNS v15 Fast (5 iter)")
movns_fast = MOVNS_V15_Fast('fastapi', archive_size=100, max_iterations=5, track_metrics=True)
fast_solutions = movns_fast.run()

fast_objectives = []
for sol in fast_solutions[:5]:
    obj = movns_fast.evaluate_objectives(sol['chromosome'])
    fast_objectives.append(obj)
    active = np.where(sol['chromosome'] == 1)[0]
    print(f"  Solução: {len(active)} packages, LU={-obj[0]:.0f}, SS={-obj[1]:.4f}")

# Teste v2 base
print("\n2. MOVNS v2 base (5 iter)")
movns_v2 = MOVNS_V2('fastapi', archive_size=100, max_iterations=5, track_metrics=True)
v2_solutions = movns_v2.run()

v2_objectives = []
for sol in v2_solutions[:5]:
    obj = movns_v2.evaluate_objectives(sol['chromosome'])
    v2_objectives.append(obj)
    indices = np.where(sol['chromosome'] == 1)[0]
    print(f"  Solução: {len(indices)} packages, LU={-obj[0]:.0f}, SS={-obj[1]:.4f}")

print("\n" + "="*70)
print("Análise da inicialização")
print("="*70)

# Verificar se os candidatos estão sendo usados
print(f"\nv15 Fast tem cooccur_candidates? {hasattr(movns_fast, 'cooccur_candidates')}")
if hasattr(movns_fast, 'cooccur_candidates'):
    print(f"  Tamanho: {len(movns_fast.cooccur_candidates)}")
    print(f"  Primeiros 5: {movns_fast.cooccur_candidates[:5]}")

print(f"\nv15 Fast tem semantic_candidates? {hasattr(movns_fast, 'semantic_candidates')}")
if hasattr(movns_fast, 'semantic_candidates'):
    print(f"  Tamanho: {len(movns_fast.semantic_candidates)}")
    print(f"  Primeiros 5: {movns_fast.semantic_candidates[:5]}")

# Verificar método simple_neighbor
print("\n" + "="*70)
print("Teste do simple_neighbor")
print("="*70)

test_solution = fast_solutions[0]['chromosome']
print(f"Solução original: {np.sum(test_solution)} packages ativos")

for i in range(5):
    neighbor = movns_fast.simple_neighbor(test_solution)
    diff = np.sum(np.abs(neighbor - test_solution))
    print(f"  Neighbor {i}: {np.sum(neighbor)} packages, mudança de {diff} bits")
    test_solution = neighbor