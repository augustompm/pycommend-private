"""
Teste final de Hypervolume para MOVNS vs NSGA-II
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2
from evaluation.metrics import calculate_hypervolume


def test_hypervolume_comparison():
    """Teste comparativo de hypervolume entre MOVNS e NSGA-II"""

    print("="*80)
    print("TESTE DE HYPERVOLUME - MOVNS vs NSGA-II")
    print("="*80)

    package = 'numpy'
    iterations = 10
    reference_point = [0, 0, -20]  # Para LU (min), SS (min), RSS (max negativo)

    # 1. MOVNS
    print("\n1. MOVNS com tracking de métricas:")
    print("-"*60)
    start = time.time()
    movns = MOVNS_VNS(package, archive_size=30, max_iterations=iterations, track_metrics=False)
    movns_sols = movns.run()
    movns_time = time.time() - start

    # Converter soluções para formato de objetivos
    movns_objs = []
    for sol in movns_sols:
        # Converter para formato de minimização (hypervolume assume minimização)
        movns_objs.append([
            -sol['objectives']['linked_usage'],  # Negativo pois queremos maximizar
            -sol['objectives']['semantic_similarity'],  # Negativo pois queremos maximizar
            sol['objectives']['set_size']  # Já é minimização
        ])

    movns_hv = calculate_hypervolume(movns_objs, reference_point) if movns_objs else 0

    print(f"MOVNS completado:")
    print(f"  Soluções: {len(movns_sols)}")
    print(f"  Tempo: {movns_time:.2f}s")
    print(f"  Hypervolume: {movns_hv:.4f}")

    # 2. NSGA-II
    print("\n2. NSGA-II baseline:")
    print("-"*60)
    start = time.time()
    nsga2 = NSGA2(package, pop_size=30, max_gen=iterations)
    nsga2_sols = nsga2.run()
    nsga2_time = time.time() - start

    # Converter soluções NSGA-II
    nsga2_objs = []
    for sol in nsga2_sols:
        nsga2_objs.append([
            -sol['objectives']['linked_usage'],
            -sol['objectives']['semantic_similarity'],
            sol['objectives']['set_size']
        ])

    nsga2_hv = calculate_hypervolume(nsga2_objs, reference_point) if nsga2_objs else 0

    print(f"NSGA-II completado:")
    print(f"  Soluções: {len(nsga2_sols)}")
    print(f"  Tempo: {nsga2_time:.2f}s")
    print(f"  Hypervolume: {nsga2_hv:.4f}")

    # 3. Comparação
    print("\n" + "="*80)
    print("COMPARAÇÃO DE HYPERVOLUME")
    print("="*80)

    print(f"\nHypervolume (quanto maior, melhor):")
    print(f"  MOVNS:   {movns_hv:.4f}")
    print(f"  NSGA-II: {nsga2_hv:.4f}")

    if nsga2_hv > 0:
        ratio = (movns_hv / nsga2_hv) * 100
        print(f"  Ratio:   {ratio:.1f}%")

        if ratio >= 90:
            print(f"\n[SUCESSO] MOVNS alcançou {ratio:.1f}% do hypervolume do NSGA-II")
            print("MOVNS é competitivo para publicação!")
        elif ratio >= 70:
            print(f"\n[ACEITÁVEL] MOVNS alcançou {ratio:.1f}% do hypervolume do NSGA-II")
            print("MOVNS tem desempenho razoável")
        else:
            print(f"\n[ATENÇÃO] MOVNS alcançou apenas {ratio:.1f}% do hypervolume do NSGA-II")
            print("MOVNS precisa de melhorias")
    else:
        print("\n[ERRO] NSGA-II não gerou soluções válidas")

    # 4. Mostrar melhores soluções
    print("\n" + "="*80)
    print("MELHORES SOLUÇÕES")
    print("="*80)

    if movns_sols:
        print("\nMOVNS - Top 3:")
        for i, sol in enumerate(movns_sols[:3], 1):
            print(f"  {i}. LU={sol['objectives']['linked_usage']:.0f}, "
                  f"SS={sol['objectives']['semantic_similarity']:.3f}, "
                  f"Size={sol['objectives']['set_size']:.0f}")

    if nsga2_sols:
        print("\nNSGA-II - Top 3:")
        for i, sol in enumerate(nsga2_sols[:3], 1):
            print(f"  {i}. LU={sol['objectives']['linked_usage']:.0f}, "
                  f"SS={sol['objectives']['semantic_similarity']:.3f}, "
                  f"Size={sol['objectives']['set_size']:.0f}")

    return {
        'movns_hv': movns_hv,
        'nsga2_hv': nsga2_hv,
        'ratio': (movns_hv / nsga2_hv * 100) if nsga2_hv > 0 else 0
    }


if __name__ == '__main__':
    result = test_hypervolume_comparison()

    print("\n" + "="*80)
    print("RESULTADO FINAL")
    print("="*80)
    print(f"Hypervolume MOVNS: {result['movns_hv']:.4f}")
    print(f"Hypervolume NSGA-II: {result['nsga2_hv']:.4f}")
    print(f"Performance relativa: {result['ratio']:.1f}%")