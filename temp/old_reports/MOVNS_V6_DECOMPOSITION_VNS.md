# MOVNS v6 - VNS com Operador de Decomposição Inteligente

## Conceito Central: Manter Identidade VNS com Benefício da Decomposição

O objetivo é criar um MOVNS que:
1. **Mantenha a estrutura VNS** (shaking, busca local, mudança de vizinhança)
2. **Incorpore decomposição como UMA vizinhança** (não como framework principal)
3. **Seja computacionalmente eficiente** (evitar o timeout da v5)

## Análise: Por que v5 Falhou?

### Problema da v5
- **6 vizinhanças com decomposição** = overhead massivo
- **Cada vizinhança recalculava decomposição** = O(n²)
- **Tentou ser MOEA/D disfarçado** = perdeu identidade VNS

### Solução para v6
- **Apenas 1 vizinhança usa decomposição** = overhead controlado
- **Pré-computar weight vectors** = cálculo único
- **Manter outras vizinhanças simples** = identidade VNS preservada

## Design MOVNS v6: Decomposição como Vizinhança

### Estrutura de Vizinhanças

```python
class MOVNS_V6:
    def __init__(self):
        # 4 vizinhanças: 3 tradicionais + 1 decomposição
        self.neighborhoods = [
            self.n1_objective_focused,      # VNS tradicional
            self.n2_size_adjustment,        # VNS tradicional
            self.n3_semantic_coherence,     # VNS tradicional
            self.n4_decomposition_guided    # NOVA: com decomposição
        ]
        
        # Pré-computar componentes de decomposição
        self.weight_vectors = self.generate_sparse_weights(10)  # Apenas 10 direções
        self.reference_point = np.array([0, 0, 1])  # Ponto ideal normalizado
```

### Vizinhança de Decomposição (N4)

```python
def n4_decomposition_guided(self, solution):
    """
    Vizinhança que usa decomposição para guiar movimento.
    Diferencial: Seleciona direção baseada em gaps no arquivo.
    """
    # 1. Identificar região menos explorada no arquivo
    sparse_weight = self.find_sparse_region_weight()

    # 2. Gerar vizinho que melhora nessa direção
    indices = np.where(solution == 1)[0]

    # 3. Movimento direcionado (não exaustivo)
    if sparse_weight[0] > 0.5:  # Foco em LU
        # Adicionar pacote com alta co-ocorrência
        candidates = self.cooccur_candidates[:10]
        best = self.select_by_decomposition(solution, candidates, sparse_weight)

    elif sparse_weight[1] > 0.5:  # Foco em SS
        # Adicionar pacote semanticamente similar
        candidates = self.semantic_candidates[:10]
        best = self.select_by_decomposition(solution, candidates, sparse_weight)

    else:  # Foco em RSS (minimizar)
        # Remover pacote menos importante
        if len(indices) > 2:
            contributions = self.calculate_weighted_contributions(solution, sparse_weight)
            worst_idx = indices[np.argmin(contributions)]
            new_solution = solution.copy()
            new_solution[worst_idx] = 0
            return new_solution

    # Aplicar movimento selecionado
    new_solution = solution.copy()
    new_solution[best] = 1
    return new_solution

def select_by_decomposition(self, solution, candidates, weight):
    """
    Seleciona melhor candidato usando decomposição Tchebycheff.
    Rápido: testa apenas 10 candidatos, não todos os 9997.
    """
    current_obj = self.evaluate_objectives(solution)
    current_norm = self.normalize_objectives(current_obj)
    current_score = np.max(weight * np.abs(current_norm - self.reference_point))

    best_candidate = candidates[0]
    best_score = current_score

    for candidate in candidates:
        if solution[candidate] == 1:  # Já selecionado
            continue

        # Testar adição do candidato
        test_solution = solution.copy()
        test_solution[candidate] = 1

        test_obj = self.evaluate_objectives(test_solution)
        test_norm = self.normalize_objectives(test_obj)
        test_score = np.max(weight * np.abs(test_norm - self.reference_point))

        if test_score < best_score:  # Minimizar distância Tchebycheff
            best_score = test_score
            best_candidate = candidate

    return best_candidate
```

### Identificação de Regiões Esparsas

```python
def find_sparse_region_weight(self):
    """
    Encontra direção (weight vector) menos explorada no arquivo.
    Simples e rápido: divide espaço em setores.
    """
    if len(self.archive) < 10:
        # Arquivo pequeno: usar weight balanceado
        return np.array([0.33, 0.33, 0.34])

    # Extrair objetivos normalizados do arquivo
    archive_objectives = []
    for sol in self.archive:
        obj = self.evaluate_objectives(sol['chromosome'])
        norm_obj = self.normalize_objectives(obj)
        archive_objectives.append(norm_obj)

    archive_objectives = np.array(archive_objectives)

    # Contar soluções por setor (simplificado)
    sector_counts = np.zeros(len(self.weight_vectors))

    for obj in archive_objectives:
        # Encontrar weight vector mais próximo
        distances = []
        for w in self.weight_vectors:
            # Distância angular (cosseno)
            cos_sim = np.dot(obj, w) / (np.linalg.norm(obj) * np.linalg.norm(w) + 1e-8)
            distances.append(1 - cos_sim)

        closest_weight_idx = np.argmin(distances)
        sector_counts[closest_weight_idx] += 1

    # Retornar weight do setor menos populado
    sparse_sector = np.argmin(sector_counts)
    return self.weight_vectors[sparse_sector]
```

## Integração com VNS Principal

### Loop VNS Modificado

```python
def run(self):
    """VNS loop com vizinhança de decomposição"""
    for iteration in range(self.max_iterations):
        solution = self.select_from_archive()

        k = 0
        no_improvement = 0

        while k < 4 and no_improvement < 5:
            # Shaking
            if k < 3:
                # Vizinhanças tradicionais: shaking normal
                x_prime = self.shake(solution, k)
            else:
                # Vizinhança de decomposição: sem shaking adicional
                # (a própria vizinhança já faz movimento direcionado)
                x_prime = solution.copy()

            # Aplica vizinhança
            x_neighbor = self.neighborhoods[k](x_prime)

            # Busca local (MOBI/P simplificado)
            if k < 3:
                # Para vizinhanças tradicionais: MOBI/P normal
                x_local = self.mobi_p_search(x_neighbor, max_neighbors=10)
            else:
                # Para decomposição: busca local mais focada
                x_local = self.decomposition_local_search(x_neighbor)

            # Avalia melhoria
            if self.is_better_or_diverse(x_local, solution):
                solution = x_local
                k = 0
                no_improvement = 0
            else:
                k += 1
                no_improvement += 1

        # Atualiza arquivo
        self.update_archive(solution)
```

### Busca Local para Decomposição

```python
def decomposition_local_search(self, solution):
    """
    Busca local específica para quando usa vizinhança de decomposição.
    Mais focada e eficiente.
    """
    current = solution.copy()
    current_obj = self.evaluate_objectives(current)

    # Selecionar weight baseado na solução atual
    weight = self.find_best_weight_for_solution(current_obj)

    # Tentar melhorar em 5 iterações rápidas
    for _ in range(5):
        # Gerar 3 vizinhos direcionados
        neighbors = []

        # Vizinho 1: Adicionar
        if np.sum(current) < self.max_size:
            add_candidate = self.find_best_addition(current, weight)
            if add_candidate is not None:
                neighbor1 = current.copy()
                neighbor1[add_candidate] = 1
                neighbors.append(neighbor1)

        # Vizinho 2: Remover
        indices = np.where(current == 1)[0]
        if len(indices) > self.min_size:
            remove_candidate = self.find_worst_package(current, weight)
            if remove_candidate is not None:
                neighbor2 = current.copy()
                neighbor2[remove_candidate] = 0
                neighbors.append(neighbor2)

        # Vizinho 3: Trocar
        if len(indices) > 0 and len(indices) < self.n_packages - 1:
            swap = self.find_best_swap(current, weight)
            if swap is not None:
                neighbor3 = current.copy()
                neighbor3[swap[0]] = 0  # Remove
                neighbor3[swap[1]] = 1  # Adiciona
                neighbors.append(neighbor3)

        # Selecionar melhor vizinho usando decomposição
        best_neighbor = self.select_best_by_decomposition(neighbors, weight)

        if best_neighbor is not None:
            current = best_neighbor
        else:
            break  # Sem melhoria

    return current
```

## Vantagens da Abordagem v6

### 1. Mantém Identidade VNS
- **Estrutura clássica**: Shaking → Vizinhança → Busca Local
- **3 vizinhanças tradicionais**: Preserva caráter VNS
- **Apenas 1 vizinhança com decomposição**: Inovação controlada

### 2. Eficiência Computacional
- **Pré-computação**: Weight vectors calculados uma vez
- **Decomposição seletiva**: Apenas quando k=3
- **Candidatos limitados**: Testa 10, não 9997
- **Complexidade**: O(n) na maioria dos casos

### 3. Benefício da Decomposição
- **Direção clara**: Quando precisa explorar nova região
- **Diversificação**: Preenche gaps no arquivo Pareto
- **Convergência**: Melhora em direções específicas

### 4. Adaptação Inteligente
- **Identifica regiões esparsas**: Dinâmico
- **Seleciona weight apropriado**: Context-aware
- **Busca local focada**: Quando usa decomposição

## Parâmetros Recomendados

```python
# MOVNS v6 - Calibrado para evitar timeout
archive_size = 50           # Menor que v4/v5
max_iterations = 30         # Suficiente para convergência
n_weight_vectors = 10       # Poucos mas representativos
mobi_p_neighbors = 10       # Rápido
decomp_local_search = 5     # Iterações focadas
no_improvement_limit = 5    # Early stopping
```

## Implementação Prática

### Classe MOVNS_V6 Simplificada

```python
class MOVNS_V6(MOVNS_V2):
    """
    MOVNS v6: VNS com vizinhança de decomposição.
    Herda de v2 (que funciona) e adiciona decomposição.
    """

    def __init__(self, main_package, archive_size=50, max_iterations=30):
        super().__init__(main_package, archive_size, max_iterations)

        # Adicionar componentes de decomposição
        self.n_weights = 10
        self.weight_vectors = self.generate_uniform_weights(self.n_weights)
        self.z_star = np.array([1.0, 1.0, 0.0])  # Ideal normalizado

        # Substituir vizinhanças
        self.neighborhoods = [
            self.n1_objective_guided,       # Da v2
            self.n2_size_adjustment,        # Da v2
            self.n3_semantic_coherence,     # Da v2
            self.n4_decomposition_guided    # NOVA
        ]

    def generate_uniform_weights(self, n):
        """Gera n weight vectors uniformes"""
        weights = []
        for i in range(n):
            w = np.random.dirichlet(np.ones(3))
            weights.append(w)
        return np.array(weights)

    def n4_decomposition_guided(self, solution):
        """Vizinhança com decomposição"""
        # Implementação como descrito acima
        sparse_weight = self.find_sparse_region_weight()
        # ... resto da implementação

    def run(self):
        """Override do run para incluir lógica de decomposição"""
        # Como descrito acima
        # ...
```

## Comparação Esperada

| Algoritmo | HV Esperado | Tempo | Característica |
|-----------|------------|-------|----------------|
| MOEA/D | 0.24-0.26 | 30s | Melhor diversidade |
| MOVNS v2 | 0.16-0.20 | 60s | VNS puro |
| MOVNS v6 | 0.20-0.23 | 45s | VNS + Decomposição |

## Por que v6 Pode Funcionar?

1. **Não é híbrido completo**: Mantém estrutura VNS
2. **Decomposição como ferramenta**: Não como framework
3. **Overhead controlado**: Apenas 1 de 4 vizinhanças
4. **Pré-computação**: Evita recálculos
5. **Foco em gaps**: Explora onde precisa

## Diferenças Chave v5 vs v6

| Aspecto | v5 (Falhou) | v6 (Proposto) |
|---------|------------|---------------|
| Vizinhanças com decomp | 6 | 1 |
| Weight vectors | 30 | 10 |
| Recálculo decomp | Sempre | Pré-computado |
| Identidade | Híbrido confuso | VNS claro |
| Complexidade | O(n²) | O(n) majoritariamente |

## Conclusão

MOVNS v6 representa o equilíbrio ideal:
- **Preserva identidade VNS** para o paper
- **Incorpora benefício da decomposição** sem overhead
- **Computacionalmente viável** (<1 minuto)
- **Melhoria esperada** sobre v2 (~15-20%)

Não vai superar MOEA/D (estruturalmente impossível), mas pode chegar a **85-90% da performance**, o que é respeitável e publicavével para um paper VNS.