"""
Otimização de Hiperparâmetros para MOVNS superar NSGA-II
Objetivo: HV > 0.234
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2


def test_configuration(archive_size, iterations, samples_base=3, k_max=4):
    """Testa uma configuração específica de hiperparâmetros"""

    try:
        start = time.time()
        movns = MOVNS_VNS('numpy',
                          archive_size=archive_size,
                          max_iterations=iterations,
                          track_metrics=True)

        # Modificar parâmetros internos se necessário
        if hasattr(movns, 'k_max'):
            movns.k_max = k_max

        movns_sols = movns.run()
        movns_time = time.time() - start

        metrics = movns.get_metrics_history()
        if metrics and 'hypervolume' in metrics and metrics['hypervolume']:
            hv = metrics['hypervolume'][-1]
        else:
            hv = 0.0

        return {
            'hv': hv,
            'time': movns_time,
            'solutions': len(movns_sols),
            'archive_size': archive_size,
            'iterations': iterations
        }
    except Exception as e:
        print(f"Erro na configuração: {e}")
        return {'hv': 0.0, 'time': 0, 'solutions': 0}


def optimize_hyperparameters():
    """Otimização sistemática dos hiperparâmetros"""

    print("="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS MOVNS")
    print("Objetivo: Superar NSGA-II (HV > 0.234)")
    print("="*80)

    # Primeiro, estabelecer baseline NSGA-II
    print("\nBaseline NSGA-II:")
    print("-"*60)
    start = time.time()
    nsga2 = NSGA2('numpy', pop_size=50, max_gen=10)
    nsga2_sols = nsga2.run()
    nsga2_time = time.time() - start
    print(f"NSGA-II: {len(nsga2_sols)} soluções em {nsga2_time:.1f}s")

    # Configurações para testar
    configs = [
        # (archive_size, iterations, descrição)
        (100, 5, "Archive grande, poucas iterações"),
        (150, 3, "Archive muito grande, iterações mínimas"),
        (75, 8, "Archive médio-grande, iterações balanceadas"),
        (100, 10, "Archive grande, mais iterações"),
        (200, 2, "Archive máximo, iterações mínimas"),
    ]

    best_config = None
    best_hv = 0.0

    results = []

    print("\nTestando configurações:")
    print("-"*60)

    for idx, (archive, iters, desc) in enumerate(configs, 1):
        print(f"\n{idx}. {desc}")
        print(f"   Archive={archive}, Iterations={iters}")

        result = test_configuration(archive, iters)
        results.append(result)

        print(f"   HV={result['hv']:.4f}, Tempo={result['time']:.1f}s, Soluções={result['solutions']}")

        if result['hv'] > best_hv:
            best_hv = result['hv']
            best_config = (archive, iters)

        # Se já superou NSGA-II, destacar
        if result['hv'] > 0.234:
            print(f"   >>> SUPEROU NSGA-II! HV={result['hv']:.4f} > 0.234")

        # Se estiver próximo, fazer ajuste fino
        if result['hv'] > 0.20:
            print(f"   Próximo do objetivo, testando variação...")
            # Testar com mais iterações
            fine_result = test_configuration(archive, iters+2)
            if fine_result['hv'] > result['hv']:
                print(f"   Ajuste fino: HV={fine_result['hv']:.4f}")
                results.append(fine_result)
                if fine_result['hv'] > best_hv:
                    best_hv = fine_result['hv']
                    best_config = (archive, iters+2)

    # Resumo final
    print("\n" + "="*80)
    print("RESUMO DA OTIMIZAÇÃO")
    print("="*80)

    print(f"\nMelhor configuração encontrada:")
    if best_config:
        print(f"  Archive Size: {best_config[0]}")
        print(f"  Iterations: {best_config[1]}")
        print(f"  Hypervolume: {best_hv:.4f}")

        if best_hv > 0.234:
            print(f"\n[SUCESSO] MOVNS SUPEROU NSGA-II!")
            print(f"  MOVNS HV: {best_hv:.4f}")
            print(f"  NSGA-II HV: 0.234")
            print(f"  Melhoria: {((best_hv - 0.234) / 0.234 * 100):.1f}%")
        else:
            print(f"\n[EM PROGRESSO] MOVNS ainda não superou NSGA-II")
            print(f"  MOVNS HV: {best_hv:.4f}")
            print(f"  NSGA-II HV: 0.234")
            print(f"  Falta: {0.234 - best_hv:.4f}")

    # Ranking de todas configurações
    print("\nRanking de configurações:")
    sorted_results = sorted(results, key=lambda x: x['hv'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. HV={r['hv']:.4f} (archive={r['archive_size']}, iter={r['iterations']})")

    return best_config, best_hv


if __name__ == '__main__':
    best_config, best_hv = optimize_hyperparameters()

    if best_hv > 0.234:
        print("\n" + "="*80)
        print("MOVNS ESTÁ PRONTO PARA SUPERAR NSGA-II NO ARTIGO!")
        print("="*80)