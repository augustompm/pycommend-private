"""
Teste MOVNS Robust vs MOEA/D
Objetivo: Vencer HV + Spacing com 80-100 soluções
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_robust import MOVNS_Robust
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("MOVNS ROBUST vs MOEA/D - TESTE FINAL")
print("="*70)

iterations = 10  # Reduzido para teste mais rápido

# 1. MOVNS Robust
print("\n1. MOVNS Robust")
print("-"*70)

movns = MOVNS_Robust('fastapi', archive_size=100, max_iterations=iterations, track_metrics=True)
start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

# Métricas MOVNS
movns_metrics = movns.get_metrics_history()
movns_hv = 0
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    movns_hv = movns_metrics['hypervolume'][-1]

# Spacing com instância própria
qm_movns = QualityMetrics()
movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')

print(f"Tempo: {movns_time:.1f}s")
print(f"Soluções: {len(movns_solutions)}")
print(f"HV: {movns_hv:.4f}")
print(f"Spacing: {movns_spacing:.4f}")

# Amostra de qualidade
if len(movns_objectives) > 0:
    best_lu = np.max(-movns_objectives[:, 0])
    print(f"Melhor LU: {best_lu:.0f}")

# 2. MOEA/D
print("\n2. MOEA/D Normalized")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

# Métricas MOEA/D
moead_metrics = moead.get_metrics_history()
moead_hv = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv = moead_metrics['hypervolume'][-1]

# Spacing com instância própria
qm_moead = QualityMetrics()
moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

print(f"Tempo: {moead_time:.1f}s")
print(f"Soluções: {len(moead_solutions)}")
print(f"HV: {moead_hv:.4f}")
print(f"Spacing: {moead_spacing:.4f}")

if len(moead_objectives) > 0:
    best_lu = np.max(-moead_objectives[:, 0])
    print(f"Melhor LU: {best_lu:.0f}")

# 3. RESULTADO
print("\n" + "="*70)
print("RESULTADO FINAL")
print("="*70)

movns_wins = 0
moead_wins = 0

print("\n1. HYPERVOLUME (maior é melhor)")
print(f"   MOVNS: {movns_hv:.4f}")
print(f"   MOEA/D: {moead_hv:.4f}")
if movns_hv > moead_hv:
    print("   VENCEDOR: MOVNS")
    movns_wins += 1
else:
    print("   VENCEDOR: MOEA/D")
    moead_wins += 1

print("\n2. SPACING (menor é melhor)")
print(f"   MOVNS: {movns_spacing:.4f}")
print(f"   MOEA/D: {moead_spacing:.4f}")
if movns_spacing < moead_spacing:
    print("   VENCEDOR: MOVNS")
    movns_wins += 1
else:
    print("   VENCEDOR: MOEA/D")
    moead_wins += 1

print("\n3. ANÁLISE DO ARQUIVO")
print(f"   MOVNS: {len(movns_solutions)} soluções")
print(f"   MOEA/D: {len(moead_solutions)} soluções")

ratio = len(movns_solutions) / len(moead_solutions) if len(moead_solutions) > 0 else 0
print(f"   Ratio: {ratio:.2f}")

print("\n" + "="*70)
if movns_wins == 2:
    print("*** SUCESSO: MOVNS VENCE 2/2 ***")
    print("Objetivo alcançado!")
elif movns_wins == 1:
    print(f"MOVNS vence {movns_wins}/2 métricas")
    print("Próximo passo: ajustar parâmetros")
else:
    print("MOVNS não venceu")
    print("Análise necessária dos parâmetros")