"""
Teste: MOEA/D com arquivo limitado a 50 vs MOVNS
SEM FALLBACKS - Usando MOEAD_50 com archive_limit=50
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_final_v2 import MOVNS_Final_V2
from optimizer.moead_50 import MOEAD_50
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("TESTE REAL: MOEA/D_50 (archive_limit=50) vs MOVNS")
print("="*70)

iterations = 30

# 1. MOVNS Final V2
print("\n1. MOVNS Final V2")
print("-"*70)

movns = MOVNS_Final_V2('fastapi', archive_size=100, max_iterations=iterations, track_metrics=True)
start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

print(f"\nTempo: {movns_time:.1f}s")
print(f"Soluções: {len(movns_solutions)}")

movns_metrics = movns.get_metrics_history()
movns_hv_interno = 0
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    movns_hv_interno = movns_metrics['hypervolume'][-1]
    print(f"HV interno: {movns_hv_interno:.4f}")

# 2. MOEA/D_50 (archive_limit=50)
print("\n2. MOEA/D_50 (archive_limit=50)")
print("-"*70)

moead = MOEAD_50('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

print(f"\nTempo: {moead_time:.1f}s")
print(f"Soluções: {len(moead_solutions)}")
print(f"Archive limit: {moead.archive_limit}")

moead_metrics = moead.get_metrics_history()
moead_hv_interno = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv_interno = moead_metrics['hypervolume'][-1]
    print(f"HV interno: {moead_hv_interno:.4f}")

# 3. Calcular métricas
print("\n3. Calculando Métricas")
print("-"*70)

# Usar métricas internas quando disponíveis
movns_hv = movns_hv_interno if movns_hv_interno > 0 else 0
moead_hv = moead_hv_interno if moead_hv_interno > 0 else 0

# Spacing com QualityMetrics separados
qm_movns = QualityMetrics()
qm_moead = QualityMetrics()

movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

# Epsilon-indicator
combined = np.vstack([movns_objectives, moead_objectives])
non_dominated = []
for i in range(len(combined)):
    dominated = False
    for j in range(len(combined)):
        if i != j:
            if np.all(combined[j] <= combined[i]) and np.any(combined[j] < combined[i]):
                dominated = True
                break
    if not dominated:
        non_dominated.append(combined[i])
reference_set = np.array(non_dominated) if len(non_dominated) > 0 else combined

qm3 = QualityMetrics()
movns_epsilon = qm3.epsilon_indicator(movns_objectives, reference_set)
qm4 = QualityMetrics()
moead_epsilon = qm4.epsilon_indicator(moead_objectives, reference_set)

# 4. Resultados
print("\n" + "="*70)
print("RESULTADOS")
print("="*70)

print("\n1. TAMANHO DOS ARQUIVOS")
print(f"   MOVNS: {len(movns_solutions)} soluções")
print(f"   MOEA/D_50: {len(moead_solutions)} soluções (limite: 50)")
print(f"   Diferença: {abs(len(movns_solutions) - len(moead_solutions))} soluções")

print("\n2. HYPERVOLUME (maior é melhor)")
print(f"   MOVNS: {movns_hv:.4f}")
print(f"   MOEA/D_50: {moead_hv:.4f}")
if movns_hv > moead_hv:
    print("   VENCEDOR: MOVNS")
    if moead_hv > 0:
        print(f"   Vantagem: {((movns_hv - moead_hv) / moead_hv * 100):.1f}%")
else:
    print("   VENCEDOR: MOEA/D_50")
    if movns_hv > 0:
        print(f"   Vantagem: {((moead_hv - movns_hv) / movns_hv * 100):.1f}%")

print("\n3. SPACING (menor é melhor)")
print(f"   MOVNS: {movns_spacing:.4f}")
print(f"   MOEA/D_50: {moead_spacing:.4f}")
if movns_spacing < moead_spacing:
    print("   VENCEDOR: MOVNS")
else:
    print("   VENCEDOR: MOEA/D_50")

print("\n4. EPSILON-INDICATOR (menor é melhor)")
print(f"   MOVNS: {movns_epsilon:.4f}")
print(f"   MOEA/D_50: {moead_epsilon:.4f}")
if movns_epsilon < moead_epsilon:
    print("   VENCEDOR: MOVNS")
else:
    print("   VENCEDOR: MOEA/D_50")

# Resumo
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)

movns_wins = 0
moead_wins = 0

if movns_hv > moead_hv: movns_wins += 1
else: moead_wins += 1

if movns_spacing < moead_spacing: movns_wins += 1
else: moead_wins += 1

if movns_epsilon < moead_epsilon: movns_wins += 1
else: moead_wins += 1

print(f"\nMOVNS vence {movns_wins}/3 métricas")
print(f"MOEA/D_50 vence {moead_wins}/3 métricas")

if movns_wins >= 2:
    print("\nCONCLUSÃO: MOVNS demonstra superioridade!")
elif movns_wins == moead_wins:
    print("\nCONCLUSÃO: Empate nas métricas")
else:
    print("\nCONCLUSÃO: MOEA/D_50 mantém competitividade")

print("\nIMPORTANTE: MOEA/D_50 usa archive_limit=50 real (sem fallbacks)")
print("="*70)