# MOVNS v5 - Decomposition-Guided VNS

## Insight Central
**Vizinhança = Direção de Decomposição**

Ao invés de vizinhanças genéricas (flip bits), cada vizinhança representa uma direção específica no espaço objetivo, como no MOEA/D.

## Análise do Problema PyCommend

### Espaço de Objetivos
```
LU (Linked Usage): [-10000, 0]     → Maximizar co-ocorrência
SS (Semantic Sim): [-1, 0]         → Maximizar similaridade
RSS (Set Size):    [2, 15]         → Minimizar tamanho
```

### Trade-offs Naturais
1. **LU vs RSS**: Mais pacotes = mais co-ocorrências, mas maior tamanho
2. **SS vs RSS**: Pacotes similares tendem a formar grupos maiores
3. **LU vs SS**: Co-ocorrência nem sempre implica similaridade semântica

## Conceito: Decomposition-Based Neighborhoods

### Princípio
Cada vizinhança é uma **função de scalarização** que guia a busca local em uma direção específica.

### Implementação

#### Vizinhança N1: Weighted Sum Direction
```python
def N1_weighted_direction(solution, weight):
    """Move na direção do vetor peso"""
    current_obj = evaluate(solution)

    # Tentar adicionar pacote que maximiza weighted sum
    best_candidate = None
    best_score = weighted_sum(current_obj, weight)

    for package in candidates:
        new_sol = add_package(solution, package)
        new_obj = evaluate(new_sol)
        score = weighted_sum(new_obj, weight)

        if score < best_score:  # Minimização
            best_score = score
            best_candidate = package

    return best_candidate
```

#### Vizinhança N2: Tchebycheff Direction
```python
def N2_tchebycheff_direction(solution, weight, z_star):
    """Move minimizando distância Tchebycheff"""
    current_obj = evaluate(solution)

    # Identificar objetivo mais distante do ideal
    distances = weight * abs(current_obj - z_star)
    weakest = argmax(distances)

    # Melhorar objetivo mais fraco
    if weakest == 0:  # LU
        return add_best_cooccurrence(solution)
    elif weakest == 1:  # SS
        return add_best_semantic(solution)
    else:  # RSS
        return remove_weakest_package(solution)
```

#### Vizinhança N3: PBI Direction
```python
def N3_pbi_direction(solution, weight, z_star):
    """Penalty-based Boundary Intersection"""
    # Balanceia convergência (d1) e diversidade (d2)
    current_obj = evaluate(solution)

    # Calcular componentes
    d1 = projection_distance(current_obj, weight, z_star)
    d2 = perpendicular_distance(current_obj, weight, z_star)

    # Mover para reduzir penalty
    if d1 > threshold:
        return move_towards_ideal(solution, z_star)
    else:
        return move_perpendicular(solution, weight)
```

## Integração VNS + Decomposition

### MOVNS v5 Estrutura

```python
class MOVNS_V5:
    def __init__(self):
        # Gerar vetores de peso como MOEA/D
        self.weight_vectors = generate_uniform_weights(n_vectors=50)

        # Ponto ideal (melhor valor para cada objetivo)
        self.z_star = np.array([0, 0, 2])  # Ideal normalizado

        # Archive principal
        self.archive = []

        # Criar vizinhanças baseadas em decomposição
        self.neighborhoods = self.create_decomposition_neighborhoods()

    def create_decomposition_neighborhoods(self):
        """Cria vizinhanças, cada uma com direção específica"""
        neighborhoods = []

        # Vizinhanças para diferentes regiões do Pareto
        for weight in self.weight_vectors[:10]:  # Top 10 direções
            neighborhoods.append(
                lambda sol, w=weight: self.weighted_sum_move(sol, w)
            )

        # Vizinhanças Tchebycheff para equilíbrio
        for weight in self.weight_vectors[10:20]:
            neighborhoods.append(
                lambda sol, w=weight: self.tchebycheff_move(sol, w)
            )

        # Vizinhanças extremas para cada objetivo
        neighborhoods.append(lambda sol: self.extreme_lu_move(sol))
        neighborhoods.append(lambda sol: self.extreme_ss_move(sol))
        neighborhoods.append(lambda sol: self.extreme_rss_move(sol))

        return neighborhoods

    def run(self):
        """VNS loop com decomposição"""
        for iteration in range(max_iterations):
            solution = select_from_archive()

            # Escolher vetor peso adaptivamente
            weight = self.select_weight_adaptive(solution)

            # VNS com vizinhanças direcionadas
            k = 0
            while k < len(self.neighborhoods):
                # Shaking na direção do peso
                x_prime = self.directional_shaking(solution, weight, k)

                # Busca local com decomposição
                x_local = self.decomposition_local_search(x_prime, weight)

                # Aceitar se melhor na direção
                if self.is_better_in_direction(x_local, solution, weight):
                    solution = x_local
                    k = 0  # Reiniciar
                else:
                    k += 1  # Próxima vizinhança

            # Atualizar archive
            self.update_archive(solution)
```

## Vantagens da Abordagem Híbrida

### 1. Direção Clara de Busca
- Cada vizinhança tem objetivo específico
- Não é busca aleatória, é busca direcionada
- Similar ao MOEA/D mas com intensificação VNS

### 2. Cobertura do Pareto
- Múltiplas direções = múltiplas regiões do Pareto
- Archive diverso naturalmente
- Não precisa crowding distance

### 3. Eficiência
- Menos avaliações que MOBI/P completo
- Mais focado que shaking aleatório
- Aproveita estrutura do problema

## Implementação Específica para PyCommend

### Vizinhanças Direcionadas

#### N1: LU-Heavy (weight=[0.7, 0.2, 0.1])
```python
def n1_lu_heavy(solution):
    """Maximiza Linked Usage"""
    # Adicionar pacote com máxima co-ocorrência
    candidates = top_cooccurrence_packages(10)
    best = max(candidates, key=lambda p: cooccurrence_gain(solution, p))
    return add_package(solution, best)
```

#### N2: SS-Heavy (weight=[0.2, 0.7, 0.1])
```python
def n2_ss_heavy(solution):
    """Maximiza Semantic Similarity"""
    # Adicionar pacote mais similar ao centroid
    centroid = compute_centroid(solution)
    candidates = top_similar_to_centroid(centroid, 10)
    return add_package(solution, candidates[0])
```

#### N3: RSS-Heavy (weight=[0.1, 0.1, 0.8])
```python
def n3_rss_heavy(solution):
    """Minimiza tamanho mantendo qualidade"""
    # Remover pacote com menor contribuição
    contributions = compute_contributions(solution)
    weakest = min(contributions, key=lambda x: x[1])
    return remove_package(solution, weakest[0])
```

#### N4: Balanced (weight=[0.33, 0.33, 0.34])
```python
def n4_balanced(solution):
    """Busca equilíbrio entre objetivos"""
    current_obj = evaluate(solution)
    norm_obj = normalize(current_obj)

    # Melhorar objetivo mais fraco
    weakest_idx = argmin(norm_obj)

    if weakest_idx == 0:
        return n1_lu_heavy(solution)
    elif weakest_idx == 1:
        return n2_ss_heavy(solution)
    else:
        return n3_rss_heavy(solution)
```

#### N5: Tchebycheff-Guided
```python
def n5_tchebycheff(solution, weight):
    """Move usando decomposição Tchebycheff"""
    current_obj = evaluate(solution)
    z_star = self.ideal_point

    # Calcular distância Tchebycheff para cada candidato
    candidates = []
    for package in available_packages(solution):
        new_sol = toggle_package(solution, package)
        new_obj = evaluate(new_sol)
        distance = max(weight * abs(new_obj - z_star))
        candidates.append((new_sol, distance))

    # Retornar solução com menor distância
    return min(candidates, key=lambda x: x[1])[0]
```

## Parâmetros Calibrados

### Base no MOVNS v2 (que funcionava)
```python
archive_size = 100          # Aumentado de 50
n_weight_vectors = 30       # Como MOEA/D
n_neighborhoods = 10        # Direções principais
max_iterations = 50         # Mais iterações
mobi_p_neighbors = 20       # Reduzido, mais focado
```

### Adaptação Dinâmica
```python
# Selecionar peso baseado em gaps no archive
def select_weight_adaptive(archive):
    # Identificar região menos explorada
    coverage = compute_pareto_coverage(archive)
    sparse_region = find_sparse_region(coverage)

    # Retornar peso que explora região sparse
    return weight_for_region(sparse_region)
```

## Resultados Esperados

### Performance Target
- **MOVNS v5**: HV > 0.25 (superar MOEA/D)
- **Tempo**: < 60 segundos
- **Soluções**: ~100 no archive

### Vantagens sobre MOEA/D
1. **Intensificação VNS**: Melhor busca local
2. **Adaptive**: Foca em regiões promissoras
3. **Menos avaliações**: Mais eficiente

### Vantagens sobre MOVNS v2
1. **Direção clara**: Não é busca cega
2. **Cobertura garantida**: Múltiplas direções
3. **Teoria sólida**: Baseado em decomposição

## Conclusão

MOVNS v5 combina:
- **Decomposição do MOEA/D**: Direções claras de busca
- **VNS do MOVNS**: Intensificação efetiva
- **Simplicidade**: Menos componentes que v3/v4

É o "melhor dos dois mundos" - a clareza da decomposição com a força da busca local VNS.