# Métricas de Distribuição e Espalhamento - Guia Completo

## Spacing Metric (SP)

### Definição
Mede a uniformidade da distribuição das soluções no espaço objetivo.

### Fórmula
```
SP = √(1/(|A|-1) * Σ(d_i - d̄)²)
```
Onde:
- d_i = distância mínima da solução i para outras soluções
- d̄ = média de todas as distâncias d_i
- |A| = número de soluções

### Características
- **Menor valor = melhor distribuição**
- **SP = 0**: Distribuição perfeitamente uniforme
- **Não considera extremos**: Foca apenas em uniformidade interna

### Implementação
```python
def calculate_spacing(solutions):
    """
    Calcula métrica de espaçamento (Spacing)
    """
    n = len(solutions)
    if n < 2:
        return 0

    distances = []
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(solutions[i] - solutions[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)

    d_mean = np.mean(distances)
    spacing = np.sqrt(np.sum((distances - d_mean) ** 2) / (n - 1))
    return spacing
```

## Spread Metric (Δ)

### Definição
Avalia tanto a distribuição quanto a extensão das soluções, incluindo distâncias aos extremos.

### Fórmula
```
Δ = (df + dl + Σ|d_i - d̄|) / (df + dl + (|A|-1) × d̄)
```
Onde:
- df = distância ao extremo frontal
- dl = distância ao extremo lateral
- d_i = distância entre soluções consecutivas
- d̄ = média das distâncias

### Características
- **Considera extremos**: Penaliza falta de cobertura
- **Valor entre 0 e 1**: 0 = distribuição ideal
- **Mais completo que Spacing**: Avalia distribuição E extensão

### Implementação
```python
def calculate_spread(solutions, extreme_points=None):
    """
    Calcula métrica de espalhamento (Spread/Delta)
    """
    n = len(solutions)
    if n < 3:
        return 1.0  # Pior caso

    # Ordenar soluções para calcular distâncias consecutivas
    sorted_sols = solutions[np.argsort(solutions[:, 0])]

    # Distâncias consecutivas
    distances = []
    for i in range(1, n):
        dist = np.linalg.norm(sorted_sols[i] - sorted_sols[i-1])
        distances.append(dist)

    d_mean = np.mean(distances)

    # Distâncias aos extremos (se fornecidos)
    if extreme_points is not None:
        df = np.linalg.norm(sorted_sols[0] - extreme_points[0])
        dl = np.linalg.norm(sorted_sols[-1] - extreme_points[1])
    else:
        df = dl = d_mean  # Estimativa

    # Calcular spread
    numerator = df + dl + np.sum(np.abs(distances - d_mean))
    denominator = df + dl + (n - 1) * d_mean

    spread = numerator / denominator
    return spread
```

## Coverage Metric (C)

### Definição
Mede a proporção do espaço objetivo coberto pelas soluções.

### Fórmula
```
Coverage = Volume(União dos hipercubos) / Volume(espaço total)
```

### Implementação
```python
def calculate_coverage(solutions, bounds):
    """
    Calcula cobertura do espaço objetivo
    """
    n_objectives = solutions.shape[1]

    # Criar grade para avaliar cobertura
    grid_points = 100  # Resolução da grade
    covered = 0
    total = grid_points ** n_objectives

    # Método simplificado para 2-3 objetivos
    ranges = []
    for i in range(n_objectives):
        ranges.append(np.linspace(bounds[i][0], bounds[i][1],
                                 grid_points))

    # Verificar cobertura (simplificado)
    for point in np.ndindex(*[grid_points] * n_objectives):
        test_point = np.array([ranges[i][point[i]]
                               for i in range(n_objectives)])

        # Verificar se alguma solução domina este ponto
        for sol in solutions:
            if np.all(sol <= test_point):
                covered += 1
                break

    return covered / total
```

## Diversity Metric (DM)

### Definição
Mede a diversidade das soluções considerando distâncias em todos os objetivos.

### Implementação
```python
def calculate_diversity(solutions):
    """
    Calcula diversidade usando distância média aos vizinhos
    """
    n = len(solutions)
    if n < 2:
        return 0

    # Calcular matriz de distâncias
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(solutions[i] - solutions[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Diversidade = média das k menores distâncias
    k = min(5, n - 1)  # k vizinhos mais próximos
    diversity = 0

    for i in range(n):
        distances = dist_matrix[i, :]
        distances[i] = float('inf')  # Excluir self
        k_nearest = np.sort(distances)[:k]
        diversity += np.mean(k_nearest)

    return diversity / n
```

## Maximum Spread (MS)

### Definição
Mede a extensão máxima em cada objetivo.

### Fórmula
```
MS = √(Σ (max(f_i) - min(f_i))²)
```

### Implementação
```python
def calculate_maximum_spread(solutions):
    """
    Calcula extensão máxima das soluções
    """
    n_objectives = solutions.shape[1]
    spread = 0

    for i in range(n_objectives):
        obj_range = np.max(solutions[:, i]) - np.min(solutions[:, i])
        spread += obj_range ** 2

    return np.sqrt(spread)
```

## Uniformity Level (UL)

### Definição
Avalia quão uniformemente as soluções estão distribuídas usando entropia.

### Implementação
```python
def calculate_uniformity_level(solutions, n_bins=10):
    """
    Calcula nível de uniformidade usando entropia
    """
    n_objectives = solutions.shape[1]

    # Criar histograma multidimensional
    ranges = []
    for i in range(n_objectives):
        ranges.append((np.min(solutions[:, i]),
                      np.max(solutions[:, i])))

    # Contar soluções em cada bin
    hist, _ = np.histogramdd(solutions, bins=n_bins)

    # Calcular entropia
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]  # Remover zeros
    prob = hist_flat / np.sum(hist_flat)

    entropy = -np.sum(prob * np.log(prob))
    max_entropy = np.log(len(hist_flat))

    uniformity = entropy / max_entropy if max_entropy > 0 else 0
    return uniformity
```

## Comparação das Métricas

| Métrica | Foco Principal | Complexidade | Valor Ideal |
|---------|---------------|--------------|-------------|
| Spacing | Uniformidade | O(n²) | 0 |
| Spread | Distribuição + Extensão | O(n log n) | 0 |
| Coverage | Cobertura do espaço | O(n × g^d) | 1 |
| Diversity | Distância média | O(n²) | Alto |
| Max Spread | Extensão | O(n × d) | Alto |
| Uniformity | Entropia | O(n + b^d) | 1 |

## Uso Combinado em PyCommend

### Suite Completa de Avaliação
```python
def evaluate_distribution_quality(solutions, reference_set=None):
    """
    Avalia qualidade da distribuição usando múltiplas métricas
    """
    # Normalizar soluções
    solutions_norm = normalize_solutions(solutions)

    metrics = {
        'spacing': calculate_spacing(solutions_norm),
        'spread': calculate_spread(solutions_norm),
        'diversity': calculate_diversity(solutions_norm),
        'max_spread': calculate_maximum_spread(solutions_norm),
        'uniformity': calculate_uniformity_level(solutions_norm)
    }

    # Interpretação
    quality_score = 0
    if metrics['spacing'] < 0.1:
        quality_score += 25
    if metrics['spread'] < 0.3:
        quality_score += 25
    if metrics['diversity'] > 0.5:
        quality_score += 25
    if metrics['uniformity'] > 0.7:
        quality_score += 25

    metrics['overall_quality'] = quality_score
    metrics['interpretation'] = (
        'Excellent' if quality_score >= 75 else
        'Good' if quality_score >= 50 else
        'Fair' if quality_score >= 25 else
        'Poor'
    )

    return metrics
```

## Diretrizes para PyCommend

### Seleção de Métricas
Para avaliar NSGA-II vs MOEA/D em PyCommend:

1. **Métricas Essenciais**:
   - Spacing: Uniformidade das recomendações
   - Spread: Cobertura de diferentes tipos de bibliotecas
   - Diversity: Variedade nas recomendações

2. **Métricas Complementares**:
   - Coverage: Para verificar exploração do espaço
   - Uniformity: Para distribuição balanceada

### Exemplo de Avaliação
```python
def compare_algorithms_distribution(nsga2_solutions, moead_solutions):
    """
    Compara distribuição de NSGA-II vs MOEA/D
    """
    results = {}

    for name, sols in [('NSGA-II', nsga2_solutions),
                       ('MOEA/D', moead_solutions)]:
        results[name] = {
            'spacing': calculate_spacing(sols),
            'spread': calculate_spread(sols),
            'diversity': calculate_diversity(sols)
        }

    # Determinar vencedor para cada métrica
    winners = {}
    for metric in ['spacing', 'spread']:  # Menor é melhor
        winners[metric] = 'NSGA-II' if results['NSGA-II'][metric] <
                                       results['MOEA/D'][metric]
                         else 'MOEA/D'

    # Para diversidade, maior é melhor
    winners['diversity'] = 'NSGA-II' if results['NSGA-II']['diversity'] >
                                        results['MOEA/D']['diversity']
                           else 'MOEA/D'

    return results, winners
```

## Referências
- Schott (1995): Proposta original da métrica Spacing
- Deb (2001): Proposta da métrica Spread
- Zitzler et al. (2003): Framework de métricas de qualidade
- Li & Yao (2019): Survey sobre métricas de diversidade