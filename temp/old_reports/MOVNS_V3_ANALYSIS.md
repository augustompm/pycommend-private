# MOVNS v3 - Análise para Superar MOEA/D

## Análise do MOEA/D Normalized

### Pontos Fortes do MOEA/D:
1. **Decomposição Tchebycheff**: Explora eficientemente múltiplas direções
2. **Archive Externo**: Mantém 100 soluções (vs 50 do MOVNS)
3. **Injection do Archive**: A cada 5 gerações, injeta boas soluções
4. **Neighborhood Limitado**: Atualiza apenas vizinhos próximos (eficiência)

### Fraquezas do MOVNS v2:
1. **Vizinhanças Genéricas**: Flips aleatórios sem considerar objetivos
2. **Archive Menor**: 50 soluções limitam diversidade
3. **Sem Aprendizado**: Não usa informações do archive efetivamente
4. **Intensificação Fraca**: MOBI/P não explora direções específicas

## Melhorias Propostas para MOVNS v3

### 1. Vizinhanças Orientadas por Objetivos

#### N1: Objective-Guided Single Change
- Se LU baixo: adicionar pacote com maior co-ocorrência
- Se SS baixo: adicionar pacote semanticamente similar
- Se RSS alto: remover pacote com menor contribuição

#### N2: Weight-Directed Search
- Usar pesos adaptativos como MOEA/D
- Explorar direções não dominadas
- Focar em gaps no archive

#### N3: Archive-Informed Swap
- Trocar pacotes baseado em soluções do archive
- Aprender padrões de sucesso
- Cross-over implícito com archive

#### N4: Decomposition-Inspired Move
- Simular decomposição sem perder VNS
- Explorar múltiplas direções simultaneamente
- Híbrido VNS-Decomposition

### 2. Melhorias Estruturais

#### Archive Management
- Aumentar para 100 soluções
- Injection periódico como MOEA/D
- Clustering para identificar gaps

#### Adaptive Intensity
- Shaking intensity baseado em convergência
- Mais exploração quando estagnado
- Menos quando melhorando

#### Learning from Success
- Rastrear quais movimentos geram melhorias
- Bias futuras decisões
- Memória de curto prazo

## Implementação Proposta

### Vizinhança N1 - Objective-Guided
```python
def n1_objective_guided(solution):
    objectives = self.evaluate_objectives(solution)
    norm_obj = self.normalize_objectives(objectives)

    # Identificar objetivo mais fraco
    weakest = np.argmin(norm_obj[:2])  # LU ou SS

    if weakest == 0:  # LU fraco
        # Adicionar pacote com alta co-ocorrência
        candidates = self.cooccur_candidates[:10]
    else:  # SS fraco
        # Adicionar pacote semanticamente similar
        candidates = self.semantic_candidates[:10]

    # Fazer mudança direcionada
    ...
```

### Vizinhança N2 - Weight-Directed
```python
def n2_weight_directed(solution):
    # Gerar peso aleatório como MOEA/D
    weight = np.random.dirichlet(np.ones(3))

    # Encontrar direção de melhoria
    current_obj = self.evaluate_objectives(solution)

    # Buscar na direção do peso
    ...
```

### Vizinhança N3 - Archive-Informed
```python
def n3_archive_informed(solution):
    # Selecionar solução promissora do archive
    archive_sol = self.select_from_archive()

    # Identificar diferenças úteis
    diff = archive_sol ^ solution

    # Aplicar subset das diferenças
    ...
```

### Vizinhança N4 - Hybrid Decomposition
```python
def n4_hybrid_decomposition(solution):
    # Simular múltiplas direções
    directions = self.generate_weight_vectors(5)

    # Explorar cada direção
    candidates = []
    for weight in directions:
        # Fazer movimento na direção
        ...

    # Retornar melhor candidato
    ...
```

## Resultados Esperados

Com essas melhorias, MOVNS v3 deve:
1. **Convergir mais rápido** com vizinhanças direcionadas
2. **Maior diversidade** com archive de 100 soluções
3. **Melhor exploração** com híbrido VNS-Decomposition
4. **Superar MOEA/D** combinando intensificação VNS com exploração dirigida

## Próximos Passos

1. Implementar MOVNS v3 com vizinhanças adaptadas
2. Testar contra MOEA/D Normalized
3. Ajustar parâmetros baseado em resultados
4. Documentar melhorias alcançadas