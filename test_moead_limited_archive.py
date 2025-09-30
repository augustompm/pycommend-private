"""
Teste: MOEA/D com arquivo limitado a 50 soluções vs MOVNS
Objetivo: Comparação mais justa limitando soluções finais
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_final_v2 import MOVNS_Final_V2
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("="*70)
print("TESTE: MOEA/D COM ARQUIVO LIMITADO A 50 SOLUÇÕES")
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

# 2. MOEA/D
print("\n2. MOEA/D Normalized")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=iterations, track_metrics=True)
start = time.time()
moead_solutions_full = moead.run()
moead_time = time.time() - start

print(f"\nTempo: {moead_time:.1f}s")
print(f"Soluções originais: {len(moead_solutions_full)}")

# LIMITAR MOEA/D A 50 SOLUÇÕES (seleção por qualidade)
if len(moead_solutions_full) > 50:
    print("Limitando MOEA/D a 50 soluções...")

    # Calcular HV individual de cada solução
    moead_quality_scores = []
    for sol in moead_solutions_full:
        obj = moead.evaluate_objectives(sol['chromosome'])
        # Score baseado em normalização dos objetivos
        lu_score = min(1.0, -obj[0] / 10000) if obj[0] < 0 else 0
        ss_score = min(1.0, -obj[1]) if obj[1] < 0 else 0
        rss_score = max(0, 1.0 - (obj[2] - 2) / 13) if obj[2] >= 2 else 0
        quality = lu_score * 0.4 + ss_score * 0.3 + rss_score * 0.3
        moead_quality_scores.append(quality)

    # Selecionar top 50
    top_indices = np.argsort(moead_quality_scores)[::-1][:50]
    moead_solutions = [moead_solutions_full[i] for i in top_indices]
    print(f"MOEA/D limitado a: {len(moead_solutions)} soluções")
else:
    moead_solutions = moead_solutions_full

moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

moead_metrics = moead.get_metrics_history()
moead_hv_interno = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv_interno = moead_metrics['hypervolume'][-1]
    print(f"HV interno (antes do filtro): {moead_hv_interno:.4f}")

# 3. Calcular métricas com arquivos comparáveis
print("\n3. Métricas com Arquivos Comparáveis")
print("-"*70)

qm_movns = QualityMetrics()
qm_moead = QualityMetrics()

# HV recalculado após filtro
movns_hv = qm_movns.hypervolume(movns_objectives, ref_point=[0, 0, 15])
moead_hv = qm_moead.hypervolume(moead_objectives, ref_point=[0, 0, 15])

# Spacing
movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

# Epsilon-indicator com referência combinada
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
reference_set = np.array(non_dominated)

qm3 = QualityMetrics()
movns_epsilon = qm3.epsilon_indicator(movns_objectives, reference_set)
qm4 = QualityMetrics()
moead_epsilon = qm4.epsilon_indicator(moead_objectives, reference_set)

# 4. Resultados
print("\n" + "="*70)
print("RESULTADOS COM ARQUIVOS COMPARÁVEIS")
print("="*70)

print("\n1. TAMANHO DOS ARQUIVOS (comparável)")
print(f"   MOVNS: {len(movns_solutions)} soluções")
print(f"   MOEA/D: {len(moead_solutions)} soluções (limitado)")

print("\n2. HYPERVOLUME (maior é melhor)")
print(f"   MOVNS: {movns_hv:.4f}")
print(f"   MOEA/D: {moead_hv:.4f}")
if movns_hv > moead_hv:
    print("   VENCEDOR: MOVNS")
    vantagem = ((movns_hv - moead_hv) / moead_hv * 100) if moead_hv > 0 else 0
    print(f"   Vantagem MOVNS: {vantagem:.1f}%")
else:
    print("   VENCEDOR: MOEA/D")
    vantagem = ((moead_hv - movns_hv) / movns_hv * 100) if movns_hv > 0 else 0
    print(f"   Vantagem MOEA/D: {vantagem:.1f}%")

print("\n3. SPACING (menor é melhor)")
print(f"   MOVNS: {movns_spacing:.4f}")
print(f"   MOEA/D: {moead_spacing:.4f}")
if movns_spacing < moead_spacing:
    print("   VENCEDOR: MOVNS")
else:
    print("   VENCEDOR: MOEA/D")

print("\n4. EPSILON-INDICATOR (menor é melhor)")
print(f"   MOVNS: {movns_epsilon:.4f}")
print(f"   MOEA/D: {moead_epsilon:.4f}")
if movns_epsilon < moead_epsilon:
    print("   VENCEDOR: MOVNS")
else:
    print("   VENCEDOR: MOEA/D")

# Resumo
print("\n" + "="*70)
print("CONCLUSÃO")
print("="*70)

movns_wins = 0
moead_wins = 0

if movns_hv > moead_hv: movns_wins += 1
else: moead_wins += 1

if movns_spacing < moead_spacing: movns_wins += 1
else: moead_wins += 1

if movns_epsilon < moead_epsilon: movns_wins += 1
else: moead_wins += 1

print(f"\nCom arquivos de tamanho comparável:")
print(f"MOVNS vence {movns_wins}/3 métricas")
print(f"MOEA/D vence {moead_wins}/3 métricas")

if movns_wins >= 2:
    print("\nMOVNS demonstra superioridade com comparação justa!")
else:
    print("\nMOEA/D mantém competitividade mesmo com arquivo reduzido")

print("\nNOTA: Este teste limita MOEA/D às 50 melhores soluções")
print("para comparação mais justa com o MOVNS")

print("="*70)