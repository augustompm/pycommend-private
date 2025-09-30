"""
Test incremental para entender problema do HV
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_v15_fast import MOVNS_V15_Fast
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("TESTE 1: Verificar HV do MOVNS v15 Fast")
print("="*70)

movns = MOVNS_V15_Fast('fastapi', archive_size=100, max_iterations=5, track_metrics=True)
movns_solutions = movns.run()

print(f"\nSoluções: {len(movns_solutions)}")

# Pegar métricas do histórico
movns_metrics = movns.get_metrics_history()
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    print(f"HV do histórico: {movns_metrics['hypervolume'][-1]:.4f}")
else:
    print("HV do histórico: ERRO - sem dados")

# Calcular HV manualmente
qm = QualityMetrics()
movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

# Normalizar como o MOVNS_V2 faz
obj_min = movns.obj_min
obj_max = movns.obj_max
print(f"\nBounds: min={obj_min}, max={obj_max}")

normalized = []
for obj in movns_objectives:
    norm_obj = (obj - obj_min) / (obj_max - obj_min + 1e-10)
    normalized.append(norm_obj)
normalized = np.array(normalized)

manual_hv = qm.hypervolume(normalized)
print(f"HV manual (com bounds do MOVNS): {manual_hv:.4f}")

# Calcular com bounds dos próprios dados
data_min = np.min(movns_objectives, axis=0)
data_max = np.max(movns_objectives, axis=0)
print(f"\nBounds dos dados: min={data_min}, max={data_max}")

normalized2 = []
for obj in movns_objectives:
    norm_obj = (obj - data_min) / (data_max - data_min + 1e-10)
    normalized2.append(norm_obj)
normalized2 = np.array(normalized2)

manual_hv2 = qm.hypervolume(normalized2)
print(f"HV manual (com bounds dos dados): {manual_hv2:.4f}")

print("\n" + "="*70)
print("TESTE 2: Comparar com MOEA/D")
print("="*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=5, track_metrics=True)
moead_solutions = moead.run()

print(f"\nSoluções MOEA/D: {len(moead_solutions)}")

moead_metrics = moead.get_metrics_history()
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    print(f"HV do histórico MOEA/D: {moead_metrics['hypervolume'][-1]:.4f}")
else:
    print("HV do histórico MOEA/D: ERRO")

# Verificar se os bounds são os mesmos
print(f"\nMOEA/D bounds: min={moead.obj_min}, max={moead.obj_max}")

print("\n" + "="*70)
print("TESTE 3: Verificar valores dos objetivos")
print("="*70)

print("\nPrimeiras 3 soluções MOVNS:")
for i in range(min(3, len(movns_objectives))):
    print(f"  {i}: {movns_objectives[i]}")

moead_objectives = []
for sol in moead_solutions[:3]:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)

print("\nPrimeiras 3 soluções MOEA/D:")
for i, obj in enumerate(moead_objectives):
    print(f"  {i}: {obj}")