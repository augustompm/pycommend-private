"""
Validação manual dos cálculos de métricas
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("VALIDAÇÃO MANUAL DAS MÉTRICAS")
print("="*70)

# Executar MOVNS Advanced com apenas 5 iterações para ser rápido
print("\n1. Executando MOVNS Advanced (5 iterações)")
movns = MOVNS_Advanced('fastapi', archive_size=100, max_iterations=5, track_metrics=True)
movns_solutions = movns.run()

# Pegar métricas do histórico
movns_metrics = movns.get_metrics_history()
hv_historico = None
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    hv_historico = movns_metrics['hypervolume'][-1]
    print(f"\nHV do histórico: {hv_historico:.4f}")

# Calcular HV manualmente passo a passo
print("\n2. Validação manual do HV:")

# Pegar objetivos
objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    objectives.append(obj)
objectives = np.array(objectives)

print(f"   Número de soluções: {len(objectives)}")
print(f"   Primeiras 3 soluções (raw):")
for i in range(min(3, len(objectives))):
    print(f"      {objectives[i]}")

# Verificar bounds
print(f"\n   Bounds do MOVNS:")
print(f"      obj_min: {movns.obj_min}")
print(f"      obj_max: {movns.obj_max}")

# Normalizar manualmente
normalized = []
for obj in objectives:
    norm_obj = (obj - movns.obj_min) / (movns.obj_max - movns.obj_min + 1e-10)
    normalized.append(norm_obj)
normalized = np.array(normalized)

print(f"\n   Primeiras 3 soluções (normalizadas):")
for i in range(min(3, len(normalized))):
    print(f"      {normalized[i]}")

# Calcular HV com QualityMetrics
qm = QualityMetrics()
hv_manual = qm.hypervolume(normalized)
print(f"\n   HV calculado manualmente: {hv_manual:.4f}")

# Comparar
print(f"\n3. Comparação:")
print(f"   HV do histórico: {hv_historico:.4f}" if hv_historico else "   HV do histórico: None")
print(f"   HV manual: {hv_manual:.4f}")
if hv_historico:
    diff = abs(hv_historico - hv_manual)
    print(f"   Diferença: {diff:.6f}")
    print(f"   Match: {'SIM' if diff < 0.001 else 'NÃO'}")

# Calcular Spacing
print(f"\n4. Cálculo do Spacing:")
spacing = qm.spacing(objectives)
print(f"   Spacing: {spacing:.4f}")

print("\n" + "="*70)
print("TESTE COM MOEA/D")
print("="*70)

# Executar MOEA/D
print("\n1. Executando MOEA/D (5 gerações)")
moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=5, track_metrics=True)
moead_solutions = moead.run()

# Verificar bounds do MOEA/D
print(f"\n   Bounds do MOEA/D:")
print(f"      obj_min: {moead.obj_min}")
print(f"      obj_max: {moead.obj_max}")

# Verificar se são iguais
bounds_match = np.allclose(movns.obj_min, moead.obj_min) and np.allclose(movns.obj_max, moead.obj_max)
print(f"\n   Bounds são iguais? {bounds_match}")

# Métricas MOEA/D
moead_metrics = moead.get_metrics_history()
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv = moead_metrics['hypervolume'][-1]
    print(f"   HV MOEA/D: {moead_hv:.4f}")

# Spacing MOEA/D
moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

moead_spacing = qm.spacing(moead_objectives)
print(f"   Spacing MOEA/D: {moead_spacing:.4f}")