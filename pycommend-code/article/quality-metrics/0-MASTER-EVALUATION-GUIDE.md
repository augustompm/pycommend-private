# Guia Mestre - Avaliação de Qualidade em Otimização Multi-Objetivo

## Sumário Executivo

Este documento consolida o conhecimento sobre métricas de qualidade para avaliar algoritmos de otimização multi-objetivo, com foco específico no projeto PyCommend.

## 1. Framework de Avaliação

### Categorias de Métricas

```
┌─────────────────────────────────────────┐
│          MÉTRICAS DE QUALIDADE          │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐        ┌──────────┐      │
│  │CONVERGÊNCIA│      │DIVERSIDADE│     │
│  └──────────┘        └──────────┘      │
│       ↓                   ↓             │
│   • IGD/IGD+          • Spacing        │
│   • GD                • Spread         │
│   • Epsilon           • Coverage       │
│                                         │
│  ┌──────────┐        ┌──────────┐      │
│  │  VOLUME  │        │CARDINALIDADE│   │
│  └──────────┘        └──────────┘      │
│       ↓                   ↓             │
│   • Hypervolume      • N solutions     │
│   • R2/R3            • N nondominated  │
│                                         │
└─────────────────────────────────────────┘
```

## 2. Métricas Essenciais - Top 5

### 2.1 Hypervolume (HV) ⭐⭐⭐⭐⭐
- **O que mede**: Volume dominado no espaço objetivo
- **Vantagem**: Único com monotonia estrita de Pareto
- **Desvantagem**: Computacionalmente caro (NP-hard)
- **Quando usar**: Avaliação geral de qualidade
- **Valor ideal**: Próximo a 1 (após normalização)

### 2.2 IGD+ ⭐⭐⭐⭐⭐
- **O que mede**: Distância invertida ao frente de Pareto
- **Vantagem**: Pareto-compliant, rápido de calcular
- **Desvantagem**: Requer conjunto de referência
- **Quando usar**: Comparação com referência conhecida
- **Valor ideal**: Próximo a 0

### 2.3 Spacing ⭐⭐⭐⭐
- **O que mede**: Uniformidade da distribuição
- **Vantagem**: Simples e intuitivo
- **Desvantagem**: Ignora extremos
- **Quando usar**: Avaliar distribuição uniforme
- **Valor ideal**: Próximo a 0

### 2.4 Spread (Δ) ⭐⭐⭐⭐
- **O que mede**: Distribuição e extensão
- **Vantagem**: Considera extremos
- **Desvantagem**: Sensível a outliers
- **Quando usar**: Avaliar cobertura completa
- **Valor ideal**: Próximo a 0

### 2.5 GD (Generational Distance) ⭐⭐⭐
- **O que mede**: Proximidade ao Pareto ótimo
- **Vantagem**: Foco em convergência
- **Desvantagem**: Ignora diversidade
- **Quando usar**: Medir apenas convergência
- **Valor ideal**: Próximo a 0

## 3. Protocolo de Avaliação para PyCommend

### Fase 1: Preparação
```python
# 1. Normalizar objetivos [0,1]
normalized = (objectives - ideal) / (nadir - ideal)

# 2. Filtrar dominados
pareto_front = filter_dominated_solutions(normalized)

# 3. Definir referências
reference_point = [1.1, 1.1, 1.1]  # Para HV
reference_set = generate_uniform_points(1000)  # Para IGD
```

### Fase 2: Cálculo de Métricas
```python
metrics = {
    # Convergência
    'igd_plus': calculate_igd_plus(pareto_front, reference_set),
    'gd': calculate_gd(pareto_front, reference_set),

    # Diversidade
    'spacing': calculate_spacing(pareto_front),
    'spread': calculate_spread(pareto_front),

    # Volume
    'hypervolume': calculate_hv(pareto_front, reference_point),

    # Estatísticas
    'n_solutions': len(pareto_front),
    'coverage_ratio': len(pareto_front) / len(solutions)
}
```

### Fase 3: Comparação Estatística
```python
# Executar 30+ vezes
n_runs = 30
results_nsga2 = [run_nsga2() for _ in range(n_runs)]
results_moead = [run_moead() for _ in range(n_runs)]

# Teste estatístico
from scipy.stats import mannwhitneyu
statistic, p_value = mannwhitneyu(results_nsga2, results_moead)

# Significância
significant = p_value < 0.05
```

## 4. Tabela de Decisão Rápida

| Situação | Métricas Prioritárias | Justificativa |
|----------|----------------------|---------------|
| Comparação geral | HV + IGD+ | Capturam convergência e diversidade |
| Foco em convergência | IGD+ + GD | Medem proximidade ao ótimo |
| Foco em diversidade | Spacing + Spread | Avaliam distribuição |
| Muitos objetivos (>5) | IGD+ + Spacing | HV muito caro computacionalmente |
| Benchmark competitivo | HV + IGD + Spread | Padrão em competições |
| Análise rápida | IGD+ + Spacing | Computação eficiente |

## 5. Interpretação de Valores

### Escala de Qualidade
```
HYPERVOLUME (normalizado 0-1):
├─ Excelente: > 0.90
├─ Muito Bom: 0.80 - 0.90
├─ Bom:       0.70 - 0.80
├─ Regular:   0.50 - 0.70
└─ Ruim:      < 0.50

IGD/IGD+ (menor é melhor):
├─ Excelente: < 0.01
├─ Muito Bom: 0.01 - 0.03
├─ Bom:       0.03 - 0.05
├─ Regular:   0.05 - 0.10
└─ Ruim:      > 0.10

SPACING (menor é melhor):
├─ Excelente: < 0.05
├─ Muito Bom: 0.05 - 0.10
├─ Bom:       0.10 - 0.15
├─ Regular:   0.15 - 0.20
└─ Ruim:      > 0.20

SPREAD (menor é melhor):
├─ Excelente: < 0.20
├─ Muito Bom: 0.20 - 0.35
├─ Bom:       0.35 - 0.50
├─ Regular:   0.50 - 0.65
└─ Ruim:      > 0.65
```

## 6. Implementação para PyCommend

### Classe de Avaliação Completa
```python
class PyCommendEvaluator:
    """
    Avaliador de qualidade para algoritmos do PyCommend
    """

    def __init__(self):
        self.metrics = {}
        self.reference_set = None

    def evaluate_algorithm(self, algorithm, package_name, n_runs=30):
        """
        Avalia algoritmo com múltiplas execuções
        """
        results = []

        for run in range(n_runs):
            # Executar algoritmo
            solutions = algorithm.run()

            # Extrair objetivos (F1, F2, F3)
            objectives = self.extract_objectives(solutions)

            # Normalizar
            objectives_norm = self.normalize(objectives)

            # Calcular métricas
            metrics = self.calculate_all_metrics(objectives_norm)
            results.append(metrics)

        # Estatísticas
        return self.aggregate_results(results)

    def calculate_all_metrics(self, objectives):
        """
        Calcula todas as métricas relevantes
        """
        return {
            'hv': self.hypervolume(objectives),
            'igd_plus': self.igd_plus(objectives),
            'spacing': self.spacing(objectives),
            'spread': self.spread(objectives),
            'n_solutions': len(objectives)
        }

    def compare_algorithms(self, algo1, algo2, package_name):
        """
        Comparação head-to-head
        """
        # Avaliar ambos
        results1 = self.evaluate_algorithm(algo1, package_name)
        results2 = self.evaluate_algorithm(algo2, package_name)

        # Comparar
        comparison = {}
        for metric in results1.keys():
            comparison[metric] = {
                'algo1': results1[metric]['mean'],
                'algo2': results2[metric]['mean'],
                'p_value': self.statistical_test(
                    results1[metric]['values'],
                    results2[metric]['values']
                ),
                'winner': self.determine_winner(metric, results1, results2)
            }

        return comparison

    def generate_report(self, comparison):
        """
        Gera relatório interpretável
        """
        report = []
        report.append("# Relatório de Avaliação\n")

        # Contagem de vitórias
        wins = {'algo1': 0, 'algo2': 0, 'tie': 0}
        for metric, data in comparison.items():
            wins[data['winner']] += 1

        # Vencedor geral
        if wins['algo1'] > wins['algo2']:
            overall = "Algoritmo 1 é superior"
        elif wins['algo2'] > wins['algo1']:
            overall = "Algoritmo 2 é superior"
        else:
            overall = "Algoritmos são equivalentes"

        report.append(f"## Conclusão: {overall}\n")
        report.append(f"- Vitórias Algo1: {wins['algo1']}")
        report.append(f"- Vitórias Algo2: {wins['algo2']}")
        report.append(f"- Empates: {wins['tie']}")

        return "\n".join(report)
```

## 7. Melhores Práticas

### DO's ✅
1. **Sempre normalize** os objetivos antes de calcular métricas
2. **Use múltiplas métricas** - nenhuma é suficiente sozinha
3. **Execute múltiplas vezes** (mínimo 30) para significância estatística
4. **Documente configurações** (população, gerações, sementes)
5. **Use conjunto de referência denso** para IGD (>1000 pontos)
6. **Reporte intervalos de confiança** além de médias
7. **Visualize os resultados** (box plots, parallel coordinates)

### DON'Ts ❌
1. **Não use apenas IGD** (não é Pareto-compliant)
2. **Não compare valores absolutos** entre problemas diferentes
3. **Não ignore testes estatísticos** ao comparar algoritmos
4. **Não use HV para >5 objetivos** sem aproximação
5. **Não esqueça de filtrar dominados** antes de calcular métricas
6. **Não use métricas únicas** para decisões importantes

## 8. Casos Especiais

### Para PyCommend (3 objetivos)
```python
recommended_metrics = {
    'primary': ['hypervolume', 'igd_plus'],
    'secondary': ['spacing', 'spread'],
    'auxiliary': ['n_nondominated', 'coverage']
}
```

### Para Muitos Objetivos (>5)
```python
many_objective_metrics = {
    'primary': ['igd_plus'],  # HV muito caro
    'secondary': ['spacing', 'maximum_spread'],
    'auxiliary': ['purity', 'delta_p']
}
```

## 9. Ferramentas e Bibliotecas

### Python
- **pymoo**: Framework completo com todas as métricas
- **jMetalPy**: Portado do Java, muito completo
- **DEAP**: Inclui HV e outras métricas básicas
- **pygmo**: Otimização e métricas eficientes

### Implementação Mínima
```python
# Instalação
pip install pymoo numpy scipy

# Uso básico
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

hv = HV(ref_point=np.array([1.1, 1.1, 1.1]))
hypervolume = hv(solutions)

igd = IGD(pareto_front)
igd_value = igd(solutions)
```

## 10. Conclusão e Recomendações

### Para PyCommend Especificamente:

1. **Métricas Primárias**:
   - Hypervolume (qualidade geral)
   - IGD+ (convergência ao ótimo)

2. **Métricas Secundárias**:
   - Spacing (uniformidade das recomendações)
   - Spread (cobertura de diferentes tipos)

3. **Protocolo de Avaliação**:
   - 30 execuções independentes
   - Teste Mann-Whitney U
   - Reportar média ± desvio padrão
   - Visualizar com box plots

4. **Critério de Decisão**:
   - Se HV(A) - HV(B) > 0.05 → A é significativamente melhor
   - Se p-value < 0.05 → Diferença estatisticamente significativa
   - Considerar múltiplas métricas antes de concluir

### Próximos Passos
1. Implementar classe `PyCommendEvaluator`
2. Executar avaliação completa NSGA-II vs MOEA/D
3. Gerar relatório com todas as métricas
4. Publicar resultados com análise estatística

## Referências Fundamentais
- Deb (2001): Multi-Objective Optimization using EAs
- Zitzler et al. (2003): Performance Assessment Framework
- Coello et al. (2007): Evolutionary Algorithms for MOO
- Audet et al. (2021): Performance Indicators in MOO (63 métricas)
- Li & Yao (2019): Many-Objective Optimization