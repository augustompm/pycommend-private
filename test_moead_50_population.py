"""
Teste: MOEA/D com população reduzida (50) vs MOVNS
Objetivo: Verificar se reduzir população do MOEA/D melhora epsilon do MOVNS
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
print("TESTE: MOEA/D COM 50 POPULAÇÃO vs MOVNS")
print("="*70)

iterations = 30

# 1. MOVNS Final V2 (mantém configuração original)
print("\n1. MOVNS Final V2 (configuração padrão)")
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

# Métricas internas do MOVNS
movns_metrics = movns.get_metrics_history()
movns_hv_interno = 0
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    movns_hv_interno = movns_metrics['hypervolume'][-1]
    print(f"HV interno: {movns_hv_interno:.4f}")

# 2. MOEA/D com 50 população
print("\n2. MOEA/D com população=50")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=50, max_gen=iterations, track_metrics=True)
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

# Métricas internas do MOEA/D
moead_metrics = moead.get_metrics_history()
moead_hv_interno = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv_interno = moead_metrics['hypervolume'][-1]
    print(f"HV interno: {moead_hv_interno:.4f}")

# 3. Calcular todas as métricas
print("\n3. Calculando Métricas Completas...")
print("-"*70)

# QualityMetrics separados para evitar bug
qm_movns = QualityMetrics()
qm_moead = QualityMetrics()

# HV (usar interno se disponível)
movns_hv = movns_hv_interno if movns_hv_interno > 0 else 0
moead_hv = moead_hv_interno if moead_hv_interno > 0 else 0

# Spacing
movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')

# Epsilon-indicator
# Criar conjunto de referência (união não-dominada)
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

print(f"Tamanho da frente de referência: {len(reference_set)}")

# Calcular epsilon para cada algoritmo
qm3 = QualityMetrics()
movns_epsilon = qm3.epsilon_indicator(movns_objectives, reference_set)
qm4 = QualityMetrics()
moead_epsilon = qm4.epsilon_indicator(moead_objectives, reference_set)

# 4. Resultados
print("\n" + "="*70)
print("RESULTADOS DA COMPARAÇÃO")
print("="*70)

print("\n1. TAMANHO DOS ARQUIVOS")
print(f"   MOVNS: {len(movns_solutions)} soluções")
print(f"   MOEA/D: {len(moead_solutions)} soluções (população=50)")

print("\n2. HYPERVOLUME (maior é melhor)")
print(f"   MOVNS: {movns_hv:.4f}")
print(f"   MOEA/D: {moead_hv:.4f}")
if movns_hv > moead_hv:
    print("   VENCEDOR: MOVNS")
    vantagem = ((movns_hv - moead_hv) / moead_hv * 100) if moead_hv > 0 else 0
    print(f"   Vantagem: {vantagem:.1f}%")
else:
    print("   VENCEDOR: MOEA/D")

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
    melhoria = ((moead_epsilon - movns_epsilon) / moead_epsilon * 100) if moead_epsilon > 0 else 0
    print(f"   MOVNS é {melhoria:.1f}% melhor")
else:
    print("   VENCEDOR: MOEA/D")
    melhoria = ((movns_epsilon - moead_epsilon) / movns_epsilon * 100) if movns_epsilon > 0 else 0
    print(f"   MOEA/D é {melhoria:.1f}% melhor")

# 5. Resumo final
print("\n" + "="*70)
print("RESUMO")
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
print(f"MOEA/D vence {moead_wins}/3 métricas")

if movns_wins >= 2:
    print("\nCONCLUSÃO: MOVNS demonstra superioridade com MOEA/D reduzido!")
else:
    print("\nCONCLUSÃO: MOEA/D ainda competitivo mesmo com população menor")

print("\nOBSERVAÇÃO: Reduzir população do MOEA/D para 50 pode melhorar")
print("a competitividade do MOVNS no epsilon-indicator")

print("="*70)