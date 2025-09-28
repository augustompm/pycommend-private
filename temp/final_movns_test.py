"""
TESTE FINAL MOVNS - Sistema completo para artigo
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2


def test_complete_system():
    """Teste completo do sistema MOVNS para o artigo"""

    print("="*80)
    print("TESTE FINAL - SISTEMA COMPLETO MOVNS")
    print("="*80)

    package = 'numpy'
    iterations = 10  # Número razoável para teste completo

    # 1. Testar MOVNS
    print(f"\n1. MOVNS com {iterations} iterações:")
    print("-"*60)
    start = time.time()
    movns = MOVNS_VNS(package, archive_size=30, max_iterations=iterations, track_metrics=False)
    movns_sols = movns.run()
    movns_time = time.time() - start

    print(f"[OK] MOVNS completado")
    print(f"  Soluções: {len(movns_sols)}")
    print(f"  Tempo: {movns_time:.2f}s")
    print(f"  Tempo/iteração: {movns_time/iterations:.2f}s")

    # Mostrar top 3 soluções
    print(f"\n  Top 3 soluções MOVNS:")
    for i, sol in enumerate(movns_sols[:3], 1):
        print(f"    {i}. {', '.join(sol['packages'][:5])}... (tamanho: {len(sol['packages'])})")
        print(f"       LU={sol['objectives']['linked_usage']:.0f}, "
              f"SS={sol['objectives']['semantic_similarity']:.3f}, "
              f"Size={sol['objectives']['set_size']}")

    # 2. Testar NSGA-II para comparação
    print(f"\n2. NSGA-II com {iterations} gerações (baseline):")
    print("-"*60)
    start = time.time()
    nsga2 = NSGA2(package, pop_size=30, max_gen=iterations)
    nsga2_sols = nsga2.run()
    nsga2_time = time.time() - start

    print(f"[OK] NSGA-II completado")
    print(f"  Soluções: {len(nsga2_sols)}")
    print(f"  Tempo: {nsga2_time:.2f}s")
    print(f"  Tempo/geração: {nsga2_time/iterations:.2f}s")

    # 3. Análise comparativa
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA")
    print("="*80)

    print(f"\n> Métricas de Performance:")
    print(f"  MOVNS:   {movns_time:.2f}s ({len(movns_sols)} soluções)")
    print(f"  NSGA-II: {nsga2_time:.2f}s ({len(nsga2_sols)} soluções)")

    if movns_time > 0 and nsga2_time > 0:
        ratio = movns_time / nsga2_time
        if ratio > 1:
            print(f"  MOVNS é {ratio:.1f}x mais lento que NSGA-II")
        else:
            print(f"  MOVNS é {1/ratio:.1f}x mais rápido que NSGA-II")

    # 4. Validação de qualidade
    print(f"\n> Validação de Qualidade:")

    # Verificar se as soluções são relevantes
    expected_packages = ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'tensorflow', 'torch']

    movns_found = []
    for sol in movns_sols[:5]:
        for pkg in expected_packages:
            if pkg in sol['packages']:
                movns_found.append(pkg)
    movns_found = list(set(movns_found))

    print(f"  MOVNS encontrou {len(movns_found)}/{len(expected_packages)} pacotes esperados:")
    print(f"    {', '.join(movns_found)}")

    # 5. Veredicto final
    print("\n" + "="*80)
    print("VEREDICTO FINAL")
    print("="*80)

    success = True
    issues = []

    # Verificar tempo de execução
    if movns_time > iterations * 10:  # Mais de 10s por iteração
        success = False
        issues.append(f"Muito lento: {movns_time/iterations:.1f}s por iteração")

    # Verificar número de soluções
    if len(movns_sols) < 5:
        success = False
        issues.append(f"Poucas soluções: apenas {len(movns_sols)}")

    # Verificar relevância
    if len(movns_found) < 3:
        success = False
        issues.append(f"Soluções irrelevantes: apenas {len(movns_found)} pacotes esperados")

    if success:
        print("[SUCESSO] SISTEMA COMPLETO E FUNCIONAL!")
        print("   - Performance adequada")
        print("   - Soluções diversas e relevantes")
        print("   - Pronto para uso no artigo científico")
    else:
        print("[FALHOU] SISTEMA PRECISA DE AJUSTES:")
        for issue in issues:
            print(f"   - {issue}")

    return {
        'success': success,
        'movns_time': movns_time,
        'nsga2_time': nsga2_time,
        'movns_solutions': len(movns_sols),
        'nsga2_solutions': len(nsga2_sols),
        'relevant_packages_found': len(movns_found)
    }


if __name__ == '__main__':
    result = test_complete_system()

    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)
    print(f"Status: {'[SUCESSO] APROVADO' if result['success'] else '[FALHOU] REPROVADO'}")
    print(f"MOVNS: {result['movns_time']:.1f}s, {result['movns_solutions']} soluções")
    print(f"NSGA-II: {result['nsga2_time']:.1f}s, {result['nsga2_solutions']} soluções")
    print(f"Qualidade: {result['relevant_packages_found']} pacotes relevantes encontrados")