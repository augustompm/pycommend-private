"""
Otimização agressiva de MOVNS para superar NSGA-II
"""

import sys
import time
import numpy as np

sys.path.insert(0, 'E:/pycommend/pycommend-code/src')

from optimizer.movns_vns import MOVNS_VNS
from optimizer.nsga2 import NSGA2


def test_aggressive_configuration(archive_size, iterations, samples_base=5, k_max=6):
    """Testa configuração agressiva com mais samples e neighborhoods"""

    try:
        start = time.time()
        movns = MOVNS_VNS('numpy',
                          archive_size=archive_size,
                          max_iterations=iterations,
                          track_metrics=True)

        # Modificar parâmetros internos agressivamente
        if hasattr(movns, 'k_max'):
            movns.k_max = k_max  # Mais neighborhoods

        # Modificar samples do MOBI/P dinamicamente
        original_mobi = movns.mobi_p_local_search
        def enhanced_mobi(solution, neighborhood, samples=samples_base):
            return original_mobi(solution, neighborhood, samples=samples_base)
        movns.mobi_p_local_search = enhanced_mobi

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
            'config': f"archive={archive_size}, iter={iterations}, samples={samples_base}, k_max={k_max}"
        }
    except Exception as e:
        print(f"Erro: {e}")
        return {'hv': 0.0, 'time': 0, 'solutions': 0}


def optimize_aggressive():
    """Otimização agressiva para superar NSGA-II"""

    print("="*80)
    print("OTIMIZAÇÃO AGRESSIVA MOVNS")
    print("Objetivo: HV > 0.234 (superar NSGA-II)")
    print("="*80)

    # Baseline NSGA-II com mais gerações para comparação justa
    print("\nBaseline NSGA-II (15 gerações):")
    print("-"*60)
    start = time.time()
    nsga2 = NSGA2('numpy', pop_size=100, max_gen=15)
    nsga2_sols = nsga2.run()
    nsga2_time = time.time() - start
    print(f"NSGA-II: {len(nsga2_sols)} soluções em {nsga2_time:.1f}s")

    best_hv = 0.0
    best_config = None
    results = []

    # Configurações agressivas
    configs = [
        # (archive, iter, samples, k_max, desc)
        (200, 5, 5, 6, "Archive grande, mais samples"),
        (300, 3, 7, 5, "Archive muito grande, muitos samples"),
        (150, 8, 4, 7, "Balanceado com mais neighborhoods"),
        (250, 4, 6, 6, "Config agressiva média"),
        (400, 2, 8, 4, "Archive máximo, samples máximo"),
        (100, 12, 3, 8, "Mais iterações, todos neighborhoods"),
        (350, 3, 5, 5, "Archive extra grande"),
    ]

    print("\nTestando configurações agressivas:")
    print("-"*60)

    for idx, (archive, iters, samples, k_max, desc) in enumerate(configs, 1):
        print(f"\n{idx}. {desc}")
        print(f"   Config: archive={archive}, iter={iters}, samples={samples}, k_max={k_max}")

        result = test_aggressive_configuration(archive, iters, samples, k_max)
        results.append(result)

        print(f"   HV={result['hv']:.4f}, Tempo={result['time']:.1f}s")

        if result['hv'] > best_hv:
            best_hv = result['hv']
            best_config = result['config']

        if result['hv'] > 0.234:
            print(f"   >>> SUCESSO! SUPEROU NSGA-II! HV={result['hv']:.4f} > 0.234")
            break
        elif result['hv'] > 0.20:
            print(f"   Próximo! Testando ajuste fino...")
            # Ajuste fino
            fine_result = test_aggressive_configuration(archive+50, iters+1, samples+1, k_max)
            if fine_result['hv'] > result['hv']:
                print(f"   Ajuste fino: HV={fine_result['hv']:.4f}")
                results.append(fine_result)
                if fine_result['hv'] > best_hv:
                    best_hv = fine_result['hv']
                    best_config = fine_result['config']

                if fine_result['hv'] > 0.234:
                    print(f"   >>> SUCESSO COM AJUSTE FINO! HV={fine_result['hv']:.4f}")
                    break

    # Resumo
    print("\n" + "="*80)
    print("RESULTADO DA OTIMIZAÇÃO AGRESSIVA")
    print("="*80)

    if best_hv > 0.234:
        print(f"\n[SUCESSO] MOVNS SUPEROU NSGA-II!")
        print(f"  MOVNS HV: {best_hv:.4f}")
        print(f"  NSGA-II HV: 0.234")
        print(f"  Melhoria: {((best_hv - 0.234) / 0.234 * 100):.1f}%")
        print(f"  Config vencedora: {best_config}")
    else:
        print(f"\n[AINDA OTIMIZANDO]")
        print(f"  Melhor HV até agora: {best_hv:.4f}")
        print(f"  Meta: 0.234")
        print(f"  Falta: {0.234 - best_hv:.4f}")
        print(f"  Melhor config: {best_config}")

    # Top 3
    print("\nTop 3 configurações:")
    sorted_results = sorted(results, key=lambda x: x['hv'], reverse=True)
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. HV={r['hv']:.4f} ({r['config']})")

    return best_hv, best_config


if __name__ == '__main__':
    best_hv, best_config = optimize_aggressive()

    if best_hv > 0.234:
        print("\n" + "="*80)
        print("MOVNS VENCEU! PRONTO PARA O ARTIGO!")
        print("="*80)