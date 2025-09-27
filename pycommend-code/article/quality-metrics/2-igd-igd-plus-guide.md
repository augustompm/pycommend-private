# IGD e IGD+ - Guia Completo

## IGD (Inverted Generational Distance)

### Definição
IGD mede a distância média dos pontos do verdadeiro frente de Pareto para o conjunto de soluções mais próximo obtido pelo algoritmo.

### Fórmula
```
IGD(P*, A) = (1/|P*|) * Σ min d(v, a) para todo v ∈ P*, a ∈ A
```
Onde:
- P* = Verdadeiro frente de Pareto (ou referência)
- A = Conjunto de soluções aproximado
- d = Distância Euclidiana

### Características
- **Requer frente de Pareto conhecido**: Precisa de P* como referência
- **Não é Pareto-compliant**: Pode avaliar conjunto dominado como melhor
- **Rápido de calcular**: O(|P*| × |A|)
- **Sensível à distribuição de P***: Pontos de referência afetam resultado

### Implementação
```python
def calculate_igd(pareto_front, solutions):
    """
    Calcula IGD entre frente de Pareto e soluções
    """
    total_distance = 0
    for pf_point in pareto_front:
        min_distance = np.min([
            np.linalg.norm(pf_point - sol)
            for sol in solutions
        ])
        total_distance += min_distance

    return total_distance / len(pareto_front)
```

## IGD+ (IGD Plus)

### Motivação
IGD+ foi proposto para resolver o problema de não-conformidade com Pareto do IGD original.

### Diferença Principal
Usa distância modificada que considera apenas componentes dominados:

```
d+(z, a) = √(Σ max(z_i - a_i, 0)²)
```

Para minimização, conta apenas quando o ponto de referência z é pior que a solução a.

### Fórmula
```
IGD+(P*, A) = (1/|P*|) * Σ min d+(v, a) para todo v ∈ P*, a ∈ A
```

### Vantagens sobre IGD
- **Weakly Pareto-compliant**: Respeita dominância fraca
- **Mais similar ao HV**: Correlação maior com Hypervolume
- **Melhor para muitos objetivos**: Mais robusto em alta dimensão

### Implementação
```python
def calculate_igd_plus(pareto_front, solutions):
    """
    Calcula IGD+ (versão Pareto-compliant)
    """
    total_distance = 0
    for pf_point in pareto_front:
        distances = []
        for sol in solutions:
            # Distância modificada (apenas componentes dominados)
            diff = np.maximum(pf_point - sol, 0)
            dist = np.linalg.norm(diff)
            distances.append(dist)
        total_distance += np.min(distances)

    return total_distance / len(pareto_front)
```

## Comparação IGD vs IGD+

| Aspecto | IGD | IGD+ |
|---------|-----|------|
| Pareto-compliance | Não | Sim (fraco) |
| Velocidade | Rápida | Rápida |
| Correlação com HV | Média | Alta |
| Uso em competições | Comum | Crescente |
| Muitos objetivos | Problemático | Melhor |

## Interpretação de Valores

### IGD/IGD+
- **Valores menores são melhores** (medida de distância)
- **IGD = 0**: Perfeita aproximação do frente de Pareto
- **IGD < 0.01**: Excelente convergência
- **IGD = 0.01-0.1**: Boa convergência
- **IGD > 0.1**: Convergência precisa melhorar

### Normalização
Sempre normalize antes de calcular:
```python
def normalize_for_igd(solutions, ideal, nadir):
    """
    Normaliza objetivos para cálculo justo de IGD
    """
    return (solutions - ideal) / (nadir - ideal)
```

## Gerando Pontos de Referência

### Método Das-Dennis
Para gerar pontos uniformemente distribuídos:
```python
def generate_reference_points(n_objectives, n_partitions):
    """
    Gera pontos de referência uniformes (Das-Dennis)
    """
    from itertools import combinations_with_replacement

    def generate_recursive(n_obj, n_part, current=[]):
        if n_obj == 1:
            current.append(n_part)
            return [current]

        points = []
        for i in range(n_part + 1):
            points.extend(
                generate_recursive(n_obj - 1, n_part - i,
                                 current + [i])
            )
        return points

    ref_points = generate_recursive(n_objectives, n_partitions)
    return np.array(ref_points) / n_partitions
```

## Uso Prático em PyCommend

### Avaliar Qualidade
1. Gerar conjunto de referência (100-500 pontos)
2. Normalizar objetivos [0, 1]
3. Calcular IGD e IGD+
4. Comparar com baseline

### Exemplo de Código
```python
def evaluate_algorithm_quality(solutions, reference_set):
    """
    Avalia qualidade usando IGD e IGD+
    """
    # Normalizar
    ideal = np.min(reference_set, axis=0)
    nadir = np.max(reference_set, axis=0)

    solutions_norm = normalize_for_igd(solutions, ideal, nadir)
    reference_norm = normalize_for_igd(reference_set, ideal, nadir)

    # Calcular métricas
    igd = calculate_igd(reference_norm, solutions_norm)
    igd_plus = calculate_igd_plus(reference_norm, solutions_norm)

    return {
        'IGD': igd,
        'IGD+': igd_plus,
        'Quality': 'Excellent' if igd < 0.01 else
                   'Good' if igd < 0.1 else 'Poor'
    }
```

## Limitações e Cuidados

### IGD
- Pode dar resultados enganosos com conjuntos dominados
- Sensível à distribuição dos pontos de referência
- Não deve ser usado sozinho para avaliar qualidade

### IGD+
- Ainda requer conhecimento do frente de Pareto
- Weakly Pareto-compliant (não strictly)
- Pode não capturar toda diversidade

## Recomendações
1. **Use IGD+ ao invés de IGD** quando possível
2. **Combine com outras métricas** (HV, Spread)
3. **Normalize sempre** antes de calcular
4. **Use conjunto de referência denso** (>100 pontos)
5. **Reporte ambos IGD e IGD+** para comparação

## Referências
- Coello et al. (2004): Proposta original do IGD
- Ishibuchi et al. (2015): Proposta do IGD+
- EMO 2019: "Comparison of Hypervolume, IGD and IGD+"