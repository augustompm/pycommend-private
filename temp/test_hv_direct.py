"""
Teste direto de hypervolume - MOVNS vs NSGA-II
"""

import sys
import numpy as np
sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2
from evaluation.metrics import calculate_hypervolume

print("="*80)
print("TESTE DIRETO DE HYPERVOLUME")
print("="*80)

# Configuração igual para comparação justa
iterations = 10
archive_size = 100

# 1. MOVNS
print("\n1. MOVNS:")
print("-"*60)
movns = MOVNS_VNS('numpy', archive_size=archive_size, max_iterations=iterations, track_metrics=False)
movns_sols = movns.run()

print(f"  Soluções geradas: {len(movns_sols)}")
if movns_sols:
    # Mostrar ranges
    movns_lu = [s['objectives']['linked_usage'] for s in movns_sols]
    movns_ss = [s['objectives']['semantic_similarity'] for s in movns_sols]
    movns_size = [s['objectives']['set_size'] for s in movns_sols]

    print(f"  LU range: {min(movns_lu):.0f} - {max(movns_lu):.0f}")
    print(f"  SS range: {min(movns_ss):.3f} - {max(movns_ss):.3f}")
    print(f"  Size range: {min(movns_size)} - {max(movns_size)}")

    # Calcular hypervolume manualmente
    # Normalizar para escala similar
    max_lu = 50000  # Valor máximo esperado
    max_ss = 1.0
    max_size = 20

    movns_objs = []
    for sol in movns_sols:
        movns_objs.append([
            sol['objectives']['linked_usage'] / max_lu,  # Normalizar
            sol['objectives']['semantic_similarity'],  # Já está entre 0-1
            1.0 - (sol['objectives']['set_size'] / max_size)  # Inverter e normalizar
        ])

    reference = [0, 0, 0]  # Ponto de referência normalizado
    movns_hv = calculate_hypervolume(movns_objs, reference)
    print(f"  Hypervolume (normalizado): {movns_hv:.4f}")

# 2. NSGA-II
print("\n2. NSGA-II:")
print("-"*60)
nsga2 = NSGA2('numpy', pop_size=archive_size, max_gen=iterations)
nsga2_sols = nsga2.run()

print(f"  Soluções geradas: {len(nsga2_sols)}")
if nsga2_sols:
    # NSGA-II retorna lista de listas
    nsga2_objs = []
    nsga2_lu = []
    nsga2_ss = []
    nsga2_size = []

    for sol in nsga2_sols:
        # NSGA-II retorna dicionários com 'objectives' como MOVNS
        if isinstance(sol, dict) and 'objectives' in sol:
            lu = sol['objectives']['linked_usage']
            ss = sol['objectives']['semantic_similarity']
            size = sol['objectives']['set_size']
        else:
            # Fallback se for lista simples
            lu = sol[0] if isinstance(sol, (list, tuple)) else sol
            ss = sol[1] if isinstance(sol, (list, tuple)) and len(sol) > 1 else 0
            size = sol[2] if isinstance(sol, (list, tuple)) and len(sol) > 2 else 5

        nsga2_lu.append(lu)
        nsga2_ss.append(ss)
        nsga2_size.append(size)

        # Normalizar igual ao MOVNS
        nsga2_objs.append([
            lu / max_lu,  # Normalizar LU
            ss,  # SS já está normalizado
            1.0 - (size / max_size)  # Inverter e normalizar size
        ])

    print(f"  LU range: {min(nsga2_lu):.0f} - {max(nsga2_lu):.0f}")
    print(f"  SS range: {min(nsga2_ss):.3f} - {max(nsga2_ss):.3f}")
    print(f"  Size range: {min(nsga2_size):.0f} - {max(nsga2_size):.0f}")

    nsga2_hv = calculate_hypervolume(nsga2_objs, reference)
    print(f"  Hypervolume (normalizado): {nsga2_hv:.4f}")

# 3. Comparação
print("\n" + "="*80)
print("COMPARAÇÃO FINAL")
print("="*80)

if movns_sols and nsga2_sols:
    print(f"\nHypervolume normalizado:")
    print(f"  MOVNS:   {movns_hv:.4f}")
    print(f"  NSGA-II: {nsga2_hv:.4f}")

    ratio = movns_hv / nsga2_hv if nsga2_hv > 0 else 0
    print(f"  Ratio:   {ratio:.2%}")

    if movns_hv > nsga2_hv:
        print("\n[SUCESSO] MOVNS SUPEROU NSGA-II!")
        print(f"  Vantagem: {((movns_hv - nsga2_hv) / nsga2_hv * 100):.1f}%")
    elif ratio >= 0.9:
        print("\n[COMPETITIVO] MOVNS está próximo do NSGA-II")
        print(f"  Performance: {ratio:.0%} do NSGA-II")
    else:
        print("\n[MELHORAR] MOVNS precisa otimização")
        print(f"  Performance: {ratio:.0%} do NSGA-II")

    # Análise de qualidade das soluções
    print("\n" + "="*80)
    print("ANÁLISE DE QUALIDADE")
    print("="*80)

    # MOVNS tem valores de LU maiores?
    max_movns_lu = max(movns_lu)
    max_nsga2_lu = max(nsga2_lu)

    if max_movns_lu > max_nsga2_lu:
        print(f"\n✓ MOVNS encontra soluções com maior linked usage:")
        print(f"  MOVNS max LU: {max_movns_lu:.0f}")
        print(f"  NSGA-II max LU: {max_nsga2_lu:.0f}")
        print(f"  Diferença: {max_movns_lu - max_nsga2_lu:.0f} ({((max_movns_lu - max_nsga2_lu)/max_nsga2_lu*100):.0f}% maior)")

    # Diversidade
    print(f"\n✓ Diversidade de soluções:")
    print(f"  MOVNS: {len(movns_sols)} soluções")
    print(f"  NSGA-II: {len(nsga2_sols)} soluções")