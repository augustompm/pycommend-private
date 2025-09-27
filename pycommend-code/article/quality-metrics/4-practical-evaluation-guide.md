# Guia Prático de Avaliação de Algoritmos Multi-Objetivo

## Framework Completo de Avaliação

### 1. Preparação dos Dados

#### Normalização
```python
def prepare_data_for_evaluation(solutions, ideal=None, nadir=None):
    """
    Prepara dados para avaliação justa
    """
    if ideal is None:
        ideal = np.min(solutions, axis=0)
    if nadir is None:
        nadir = np.max(solutions, axis=0)

    # Normalizar para [0, 1]
    normalized = (solutions - ideal) / (nadir - ideal + 1e-10)

    return normalized, ideal, nadir
```

#### Filtragem de Dominados
```python
def filter_dominated(solutions):
    """
    Remove soluções dominadas (mantém apenas Pareto front)
    """
    n = len(solutions)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i != j and not is_dominated[i]:
                # Verifica se i é dominado por j
                if np.all(solutions[j] <= solutions[i]) and
                   np.any(solutions[j] < solutions[i]):
                    is_dominated[i] = True
                    break

    return solutions[~np.array(is_dominated)]
```

### 2. Suite de Métricas Essenciais

```python
class QualityMetrics:
    """
    Suite completa de métricas de qualidade
    """

    def __init__(self, reference_set=None):
        self.reference_set = reference_set
        self.metrics_computed = {}

    def evaluate_all(self, solutions):
        """
        Calcula todas as métricas principais
        """
        results = {}

        # Convergência
        if self.reference_set is not None:
            results['IGD'] = self.calculate_igd(solutions)
            results['IGD+'] = self.calculate_igd_plus(solutions)
            results['GD'] = self.calculate_gd(solutions)

        # Diversidade
        results['Spacing'] = self.calculate_spacing(solutions)
        results['Spread'] = self.calculate_spread(solutions)
        results['Diversity'] = self.calculate_diversity(solutions)

        # Volume
        results['Hypervolume'] = self.calculate_hypervolume(solutions)

        # Cardinalidade
        results['N_solutions'] = len(solutions)
        results['N_nondominated'] = len(filter_dominated(solutions))

        # Estatísticas
        results['Stats'] = self.calculate_statistics(solutions)

        return results

    def calculate_statistics(self, solutions):
        """
        Estatísticas básicas das soluções
        """
        return {
            'mean': np.mean(solutions, axis=0).tolist(),
            'std': np.std(solutions, axis=0).tolist(),
            'min': np.min(solutions, axis=0).tolist(),
            'max': np.max(solutions, axis=0).tolist(),
            'range': (np.max(solutions, axis=0) -
                     np.min(solutions, axis=0)).tolist()
        }
```

### 3. Protocolo de Comparação

```python
def compare_algorithms(algo1_solutions, algo2_solutions,
                      algo1_name="Algorithm 1",
                      algo2_name="Algorithm 2",
                      n_runs=30):
    """
    Protocolo completo para comparar dois algoritmos
    """
    results = {
        algo1_name: [],
        algo2_name: []
    }

    # Múltiplas execuções para significância estatística
    for run in range(n_runs):
        # Executar algoritmos (ou usar resultados salvos)
        sols1 = algo1_solutions[run] if isinstance(algo1_solutions, list)
                else algo1_solutions
        sols2 = algo2_solutions[run] if isinstance(algo2_solutions, list)
                else algo2_solutions

        # Normalizar conjuntamente
        all_sols = np.vstack([sols1, sols2])
        normalized, ideal, nadir = prepare_data_for_evaluation(all_sols)

        n1 = len(sols1)
        sols1_norm = normalized[:n1]
        sols2_norm = normalized[n1:]

        # Calcular métricas
        metrics = QualityMetrics()
        results[algo1_name].append(metrics.evaluate_all(sols1_norm))
        results[algo2_name].append(metrics.evaluate_all(sols2_norm))

    # Análise estatística
    return statistical_analysis(results, algo1_name, algo2_name)
```

### 4. Análise Estatística

```python
from scipy import stats

def statistical_analysis(results, name1, name2):
    """
    Análise estatística completa
    """
    analysis = {}

    metrics = results[name1][0].keys()

    for metric in metrics:
        if metric == 'Stats':
            continue

        values1 = [r[metric] for r in results[name1]]
        values2 = [r[metric] for r in results[name2]]

        # Teste de normalidade
        _, p_norm1 = stats.shapiro(values1)
        _, p_norm2 = stats.shapiro(values2)

        # Escolher teste apropriado
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Dados normais - usar t-test
            statistic, p_value = stats.ttest_ind(values1, values2)
            test_used = 't-test'
        else:
            # Dados não-normais - usar Mann-Whitney U
            statistic, p_value = stats.mannwhitneyu(values1, values2)
            test_used = 'Mann-Whitney U'

        # Effect size (Cohen's d)
        mean1, std1 = np.mean(values1), np.std(values1)
        mean2, std2 = np.mean(values2), np.std(values2)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        analysis[metric] = {
            f'{name1}_mean': mean1,
            f'{name1}_std': std1,
            f'{name2}_mean': mean2,
            f'{name2}_std': std2,
            'test': test_used,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': interpret_effect_size(cohens_d),
            'winner': determine_winner(metric, mean1, mean2, p_value)
        }

    return analysis
```

### 5. Interpretação dos Resultados

```python
def interpret_effect_size(d):
    """
    Interpreta tamanho do efeito (Cohen's d)
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def determine_winner(metric, mean1, mean2, p_value):
    """
    Determina vencedor baseado na métrica
    """
    if p_value >= 0.05:
        return "No significant difference"

    # Métricas onde menor é melhor
    minimize_metrics = ['IGD', 'IGD+', 'GD', 'Spacing', 'Spread']

    # Métricas onde maior é melhor
    maximize_metrics = ['Hypervolume', 'Diversity', 'N_nondominated']

    if metric in minimize_metrics:
        return "Algorithm 1" if mean1 < mean2 else "Algorithm 2"
    elif metric in maximize_metrics:
        return "Algorithm 1" if mean1 > mean2 else "Algorithm 2"
    else:
        return "Context-dependent"
```

### 6. Visualização de Resultados

```python
import matplotlib.pyplot as plt

def plot_metrics_comparison(analysis, save_path=None):
    """
    Visualiza comparação de métricas
    """
    metrics = list(analysis.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, (n_metrics + 1) // 2,
                             figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        data = analysis[metric]
        ax = axes[idx]

        # Box plot
        algo1_values = data['algo1_values']
        algo2_values = data['algo2_values']

        ax.boxplot([algo1_values, algo2_values],
                   labels=['NSGA-II', 'MOEA/D'])
        ax.set_title(f'{metric}\np={data["p_value"]:.4f}')
        ax.set_ylabel('Value')

        # Marcar significância
        if data['significant']:
            ax.text(1.5, ax.get_ylim()[1] * 0.9, '*',
                   fontsize=20, ha='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
```

### 7. Relatório Automático

```python
def generate_evaluation_report(analysis, output_file="evaluation_report.md"):
    """
    Gera relatório em Markdown
    """
    report = ["# Multi-Objective Algorithm Evaluation Report\n"]
    report.append(f"Generated: {datetime.now()}\n")

    # Resumo executivo
    report.append("## Executive Summary\n")
    winners = {}
    for metric, data in analysis.items():
        winner = data.get('winner', 'N/A')
        winners[winner] = winners.get(winner, 0) + 1

    report.append(f"Overall performance comparison:\n")
    for winner, count in winners.items():
        report.append(f"- {winner}: {count} metrics\n")

    # Detalhes por métrica
    report.append("\n## Detailed Metrics Analysis\n")
    for metric, data in analysis.items():
        report.append(f"\n### {metric}\n")
        report.append(f"- Algorithm 1: {data['algo1_mean']:.4f} ± "
                     f"{data['algo1_std']:.4f}\n")
        report.append(f"- Algorithm 2: {data['algo2_mean']:.4f} ± "
                     f"{data['algo2_std']:.4f}\n")
        report.append(f"- Statistical test: {data['test']}\n")
        report.append(f"- p-value: {data['p_value']:.4f}\n")
        report.append(f"- Effect size: {data['effect_size']}\n")
        report.append(f"- **Winner**: {data['winner']}\n")

    # Recomendações
    report.append("\n## Recommendations\n")
    report.append(generate_recommendations(analysis))

    # Salvar relatório
    with open(output_file, 'w') as f:
        f.writelines(report)

    return "\n".join(report)
```

## Checklist de Avaliação

### ✅ Pré-processamento
- [ ] Normalizar objetivos
- [ ] Remover soluções duplicadas
- [ ] Filtrar soluções inválidas
- [ ] Verificar constraints

### ✅ Métricas de Convergência
- [ ] IGD ou IGD+
- [ ] Generational Distance (GD)
- [ ] Epsilon Indicator

### ✅ Métricas de Diversidade
- [ ] Spacing
- [ ] Spread
- [ ] Maximum Spread

### ✅ Métricas Combinadas
- [ ] Hypervolume
- [ ] R2 Indicator

### ✅ Análise Estatística
- [ ] Múltiplas execuções (≥30)
- [ ] Teste de hipótese apropriado
- [ ] Cálculo de effect size
- [ ] Intervalos de confiança

### ✅ Visualização
- [ ] Frente de Pareto
- [ ] Box plots das métricas
- [ ] Evolução temporal
- [ ] Parallel coordinates

## Exemplo Completo para PyCommend

```python
def evaluate_pycommend_algorithms():
    """
    Avaliação completa NSGA-II vs MOEA/D para PyCommend
    """
    # 1. Executar algoritmos
    package = "numpy"
    n_runs = 30

    nsga2_results = []
    moead_results = []

    for run in range(n_runs):
        # NSGA-II
        nsga2 = NSGA2(package, pop_size=50, max_gen=20)
        nsga2_sols = nsga2.run()
        nsga2_results.append(extract_objectives(nsga2_sols))

        # MOEA/D
        moead = MOEAD(package, pop_size=50, max_gen=20)
        moead_sols = moead.run()
        moead_results.append(extract_objectives(moead_sols))

    # 2. Comparar
    analysis = compare_algorithms(
        nsga2_results, moead_results,
        "NSGA-II", "MOEA/D", n_runs
    )

    # 3. Gerar relatório
    report = generate_evaluation_report(analysis)

    # 4. Visualizar
    plot_metrics_comparison(analysis, "comparison.png")

    return analysis, report
```

## Interpretação Final

### Critérios de Decisão

1. **Convergência mais importante**: Use IGD/IGD+, GD
2. **Diversidade mais importante**: Use Spread, Spacing
3. **Balance geral**: Use Hypervolume
4. **Muitos objetivos (>3)**: Prefira IGD+ sobre HV

### Thresholds de Qualidade

| Métrica | Excelente | Bom | Aceitável | Ruim |
|---------|-----------|-----|-----------|------|
| IGD | <0.01 | 0.01-0.05 | 0.05-0.1 | >0.1 |
| Spacing | <0.05 | 0.05-0.1 | 0.1-0.2 | >0.2 |
| Spread | <0.2 | 0.2-0.4 | 0.4-0.6 | >0.6 |
| HV | >0.9 | 0.7-0.9 | 0.5-0.7 | <0.5 |

## Referências
- Deb et al. (2002): "A Fast and Elitist MOEA: NSGA-II"
- Zitzler et al. (2003): "Performance Assessment Framework"
- Riquelme et al. (2015): "Performance Metrics Survey"
- Audet et al. (2021): "Performance indicators in multiobjective optimization"