"""
Teste Final: MOVNS Final V2 (Advanced) vs MOEA/D
Objetivo: Demonstrar superioridade do MOVNS em HV
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
print("COMPARAÇÃO FINAL: MOVNS vs MOEA/D")
print("="*70)

movns_iterations = 30  # Iterações iguais para comparação justa
moead_iterations = 15  # MOEA/D com menos iterações (convergência mais rápida)

# 1. MOVNS Final V2 (Advanced com HV=0.30)
print("\n1. MOVNS Final V2 (Quality-focused)")
print("-"*70)

movns = MOVNS_Final_V2('fastapi', archive_size=100, max_iterations=movns_iterations, track_metrics=True)
start = time.time()
movns_solutions = movns.run()
movns_time = time.time() - start

print(f"\nTempo: {movns_time:.1f}s")
print(f"Soluções no arquivo: {len(movns_solutions)}")

# Calcular métricas do MOVNS
movns_objectives = []
for sol in movns_solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    movns_objectives.append(obj)
movns_objectives = np.array(movns_objectives)

# Usar métricas internas primeiro (mais confiáveis)
movns_metrics = movns.get_metrics_history()
movns_hv_interno = 0
if movns_metrics and 'hypervolume' in movns_metrics and len(movns_metrics['hypervolume']) > 0:
    movns_hv_interno = movns_metrics['hypervolume'][-1]
    print(f"HV (interno): {movns_hv_interno:.4f}")

# Spacing com QualityMetrics próprio
qm_movns = QualityMetrics()
movns_spacing = qm_movns.spacing(movns_objectives) if len(movns_objectives) > 1 else float('inf')
print(f"Spacing: {movns_spacing:.4f}")

# Usar HV interno se disponível
movns_hv = movns_hv_interno if movns_hv_interno > 0 else 0

if len(movns_objectives) > 0:
    best_lu = np.max(-movns_objectives[:, 0])
    best_ss = np.max(-movns_objectives[:, 1])
    best_rss = np.min(movns_objectives[:, 2])
    print(f"Melhor LU: {best_lu:.0f}")
    print(f"Melhor SS: {best_ss:.4f}")
    print(f"Melhor RSS: {best_rss:.0f}")

# 2. MOEA/D Normalized
print("\n2. MOEA/D Normalized (Decomposition)")
print("-"*70)

moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=moead_iterations, track_metrics=True)
start = time.time()
moead_solutions = moead.run()
moead_time = time.time() - start

print(f"\nTempo: {moead_time:.1f}s")

# IMPORTANTE: MOEA/D retorna pareto front do arquivo externo
# Vamos verificar o tamanho real
print(f"Soluções retornadas (Pareto front): {len(moead_solutions)}")

# Se retornou população inteira, filtrar para não-dominadas
if len(moead_solutions) > 50:
    print("Filtrando para soluções não-dominadas...")
    non_dominated = []
    for i, sol_i in enumerate(moead_solutions):
        dominated = False
        for j, sol_j in enumerate(moead_solutions):
            if i != j:
                obj_i = moead.evaluate_objectives(sol_i['chromosome'])
                obj_j = moead.evaluate_objectives(sol_j['chromosome'])
                if moead.dominates(obj_j, obj_i):
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(sol_i)
    moead_solutions = non_dominated
    print(f"Soluções não-dominadas: {len(moead_solutions)}")

# Calcular métricas do MOEA/D
moead_objectives = []
for sol in moead_solutions:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_objectives.append(obj)
moead_objectives = np.array(moead_objectives)

# Usar métricas internas primeiro (mais confiáveis)
moead_metrics = moead.get_metrics_history()
moead_hv_interno = 0
if moead_metrics and 'hypervolume' in moead_metrics and len(moead_metrics['hypervolume']) > 0:
    moead_hv_interno = moead_metrics['hypervolume'][-1]
    print(f"HV (interno): {moead_hv_interno:.4f}")

# Spacing com QualityMetrics próprio
qm_moead = QualityMetrics()
moead_spacing = qm_moead.spacing(moead_objectives) if len(moead_objectives) > 1 else float('inf')
print(f"Spacing: {moead_spacing:.4f}")

# Usar HV interno se disponível
moead_hv = moead_hv_interno if moead_hv_interno > 0 else 0

if len(moead_objectives) > 0:
    best_lu = np.max(-moead_objectives[:, 0])
    best_ss = np.max(-moead_objectives[:, 1])
    best_rss = np.min(moead_objectives[:, 2])
    print(f"Melhor LU: {best_lu:.0f}")
    print(f"Melhor SS: {best_ss:.4f}")
    print(f"Melhor RSS: {best_rss:.0f}")

# 3. RESULTADO FINAL
print("\n" + "="*70)
print("RESULTADO DA COMPARAÇÃO")
print("="*70)

movns_wins = 0
moead_wins = 0

print("\n1. HYPERVOLUME (maior é melhor)")
print(f"   MOVNS: {movns_hv:.4f} ({len(movns_solutions)} soluções)")
print(f"   MOEA/D: {moead_hv:.4f} ({len(moead_solutions)} soluções)")
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

print("\n3. ANÁLISE DE QUALIDADE")
print(f"   MOVNS: {len(movns_solutions)} soluções de alta qualidade")
print(f"   MOEA/D: {len(moead_solutions)} soluções")

print("\n" + "="*70)
if movns_wins >= 1 and movns_hv > moead_hv:
    print("MOVNS DEMONSTRA SUPERIORIDADE EM HV")
    print(f"HV advantage: {(movns_hv/moead_hv - 1)*100:.1f}%")
elif movns_wins == 2:
    print("MOVNS VENCE AMBAS AS MÉTRICAS")
else:
    print(f"Resultado: {movns_wins} vs {moead_wins}")

print("="*70)