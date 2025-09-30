"""
Análise detalhada das métricas IGD+ e Spacing
Investigar por que valores estão tão similares entre MOVNS e MOEA/D
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics


def analyze_metrics_in_detail(solutions, algo, algo_name):
    """Análise detalhada das métricas"""

    print(f"\n{'='*70}")
    print(f"ANÁLISE DETALHADA: {algo_name}")
    print(f"{'='*70}")

    if not solutions:
        print("Sem soluções para analisar")
        return

    # Extrair objetivos
    objectives = []
    for sol in solutions:
        if isinstance(sol, dict) and 'chromosome' in sol:
            obj = algo.evaluate_objectives(sol['chromosome'])
        else:
            obj = algo.evaluate_objectives(sol)
        objectives.append(obj)

    objectives = np.array(objectives)
    print(f"\n1. DADOS BRUTOS:")
    print(f"   Total de soluções: {len(objectives)}")
    print(f"   Objetivos shape: {objectives.shape}")

    # Estatísticas dos objetivos
    print(f"\n2. ESTATÍSTICAS DOS OBJETIVOS:")
    print(f"   LU: min={-objectives[:, 0].max():.0f}, max={-objectives[:, 0].min():.0f}, mean={-objectives[:, 0].mean():.0f}")
    print(f"   SS: min={-objectives[:, 1].max():.4f}, max={-objectives[:, 1].min():.4f}, mean={-objectives[:, 1].mean():.4f}")
    print(f"   RSS: min={objectives[:, 2].min():.1f}, max={objectives[:, 2].max():.1f}, mean={objectives[:, 2].mean():.1f}")

    # Normalização
    normalized = []
    for obj in objectives:
        norm_obj = algo.normalize_objectives(obj)
        normalized.append(norm_obj)
    normalized = np.array(normalized)

    print(f"\n3. APÓS NORMALIZAÇÃO:")
    print(f"   LU norm: min={normalized[:, 0].min():.4f}, max={normalized[:, 0].max():.4f}")
    print(f"   SS norm: min={normalized[:, 1].min():.4f}, max={normalized[:, 1].max():.4f}")
    print(f"   RSS norm: min={normalized[:, 2].min():.4f}, max={normalized[:, 2].max():.4f}")

    # Calcular métricas
    qm = QualityMetrics()

    # Hypervolume
    ref_point = np.array([0, 0, 1.0])
    hv = qm.hypervolume(normalized, ref_point)
    print(f"\n4. HYPERVOLUME:")
    print(f"   Valor: {hv:.4f}")
    print(f"   Ponto de referência: {ref_point}")

    # IGD+ com reference set gerado
    print(f"\n5. IGD+ (Inverted Generational Distance Plus):")

    # Gerar conjunto de referência ideal (frente de Pareto aproximada)
    reference = []
    for i in range(100):
        w1 = np.random.random()
        w2 = np.random.random() * (1 - w1)
        w3 = 1 - w1 - w2

        # Pontos ideais ponderados
        lu = -10000 * w1  # Máximo LU
        ss = -1.0 * w2    # Máximo SS
        rss = 2 + 13 * w3  # Entre min e max RSS

        reference.append([lu, ss, rss])

    reference = np.array(reference)

    # Calcular IGD+
    igd_plus = qm.igd_plus(objectives, reference)
    print(f"   IGD+: {igd_plus:.4f}")
    print(f"   Quanto menor, melhor (distância média do reference set)")
    print(f"   Reference set: {len(reference)} pontos")

    # Spacing
    print(f"\n6. SPACING (Uniformidade da distribuição):")
    spacing = qm.spacing(objectives)
    print(f"   Spacing: {spacing:.4f}")
    print(f"   Quanto menor, mais uniforme a distribuição")
    print(f"   Mede o desvio padrão das distâncias mínimas entre soluções")

    # Análise de diversidade
    print(f"\n7. ANÁLISE DE DIVERSIDADE:")

    # Calcular distâncias entre todas as soluções
    n = len(normalized)
    if n > 1:
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(normalized[i] - normalized[j])
                distances.append(dist)

        print(f"   Distâncias entre soluções:")
        print(f"   Min: {np.min(distances):.4f}")
        print(f"   Max: {np.max(distances):.4f}")
        print(f"   Mean: {np.mean(distances):.4f}")
        print(f"   Std: {np.std(distances):.4f}")

    # Análise de dominância
    print(f"\n8. ANÁLISE DE DOMINÂNCIA:")
    dominated_count = 0
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j:
                # Checa se j domina i
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated = True
                    dominated_count += 1
                    break

    non_dominated = n - dominated_count
    print(f"   Soluções não-dominadas: {non_dominated}/{n} ({100*non_dominated/n:.1f}%)")

    return {
        'hv': hv,
        'igd_plus': igd_plus,
        'spacing': spacing,
        'n_solutions': n,
        'non_dominated': non_dominated
    }


def main():
    """Comparação detalhada das métricas"""

    print("\n" + "="*70)
    print("ANÁLISE DETALHADA DAS MÉTRICAS: POR QUE TÃO SIMILARES?")
    print("="*70)

    package = 'fastapi'

    # MOVNS
    print("\nExecutando MOVNS Advanced...")
    movns = MOVNS_Advanced(package, archive_size=100, max_iterations=10, track_metrics=True)
    movns_solutions = movns.run()
    movns_metrics = analyze_metrics_in_detail(movns_solutions, movns, "MOVNS Advanced")

    # MOEA/D
    print("\nExecutando MOEA/D...")
    moead = MOEAD_Normalized(package, pop_size=100, max_gen=10)
    moead_solutions = moead.run()
    moead_metrics = analyze_metrics_in_detail(moead_solutions, moead, "MOEA/D Normalized")

    # Comparação final
    print("\n" + "="*70)
    print("COMPARAÇÃO FINAL E EXPLICAÇÃO")
    print("="*70)

    print("\n1. POR QUE IGD+ TÃO SIMILAR?")
    print(f"   MOVNS IGD+: {movns_metrics['igd_plus']:.4f}")
    print(f"   MOEA/D IGD+: {moead_metrics['igd_plus']:.4f}")
    print(f"   Diferença: {abs(movns_metrics['igd_plus'] - moead_metrics['igd_plus']):.4f}")
    print("\n   Possíveis razões:")
    print("   - Ambos algoritmos convergem para região similar do espaço objetivo")
    print("   - Reference set pode não estar bem calibrado")
    print("   - Normalização pode estar comprimindo as diferenças")

    print("\n2. O QUE É SPACING?")
    print(f"   MOVNS Spacing: {movns_metrics['spacing']:.4f}")
    print(f"   MOEA/D Spacing: {moead_metrics['spacing']:.4f}")
    print("\n   Explicação:")
    print("   - Spacing mede a uniformidade da distribuição das soluções")
    print("   - Calcula o desvio padrão das distâncias mínimas entre soluções")
    print("   - Spacing baixo = distribuição mais uniforme (melhor)")
    print("   - Spacing alto = soluções agrupadas em clusters (pior)")

    print("\n3. DIFERENÇA REAL ESTÁ NO HYPERVOLUME:")
    print(f"   MOVNS HV: {movns_metrics['hv']:.4f}")
    print(f"   MOEA/D HV: {moead_metrics['hv']:.4f}")
    if movns_metrics['hv'] > 0 and moead_metrics['hv'] > 0:
        ratio = movns_metrics['hv'] / moead_metrics['hv']
        print(f"   Ratio: {ratio:.2f}x")
    print("\n   HV é mais confiável porque:")
    print("   - Não depende de reference set externo")
    print("   - Mede volume dominado (qualidade absoluta)")
    print("   - É Pareto compliant")

    print("\n4. QUALIDADE DAS FRENTES:")
    print(f"   MOVNS: {movns_metrics['non_dominated']}/{movns_metrics['n_solutions']} não-dominadas")
    print(f"   MOEA/D: {moead_metrics['non_dominated']}/{moead_metrics['n_solutions']} não-dominadas")


if __name__ == "__main__":
    main()