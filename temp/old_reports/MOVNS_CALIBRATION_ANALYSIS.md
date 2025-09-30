# MOVNS Calibration Analysis - Superando MOEA/D

## Análise Crítica: Por que MOEA/D Ganha?

### Pontos Fortes do MOEA/D (HV ~0.24-0.26)
1. **População = 50, Archive = 100**: Total 150 soluções mantidas
2. **Decomposição Tchebycheff**: Explora uniformemente o espaço
3. **Update Limitado**: Máximo 20% dos vizinhos atualizados (eficiência)
4. **Archive Injection**: A cada 5 gerações, injeta boas soluções
5. **Normalização Completa**: Todos os cálculos normalizados

### Problemas do MOVNS Atual
1. **Archive Pequeno**: Apenas 50 soluções (vs 150 do MOEA/D)
2. **Intensificação Excessiva**: VNS foca demais localmente
3. **Shaking Ineficiente**: Perturbações aleatórias sem direção
4. **MOBI/P Limitado**: Testa poucos vizinhos (10-15)
5. **Sem Memória**: Não aprende com iterações anteriores

## Literatura VNS Relevante

### Hansen & Mladenović (2001) - VNS Principles
- **Princípio 1**: Mínimo local em uma vizinhança não é necessariamente mínimo em outra
- **Princípio 2**: Mínimo global é mínimo local em todas as vizinhanças
- **Aplicação**: Precisamos vizinhanças REALMENTE diferentes, não apenas variações

### Dahite et al. (2022) - MOVNS Original
- **Archive Size**: Recomenda 100-200 soluções
- **Shaking Intensity**: Adaptativo baseado em estagnação
- **MOBI/P**: Testa TODOS os vizinhos possíveis (não subset)

### Duarte et al. (2015) - VNS for Multi-Objective
- **Pareto Dominance**: Usar arquivo auxiliar para direções promissoras
- **Adaptive Neighborhoods**: Ajustar tamanho baseado em sucesso
- **Path Relinking**: Conectar soluções do arquivo

## Estratégia de Calibração

### 1. Hiperparâmetros Críticos

#### Archive Size
- **Atual**: 50
- **Testar**: [100, 150, 200]
- **Justificativa**: Dahite et al. (2022) usa 100+

#### MOBI/P Neighbors
- **Atual**: 10-15
- **Testar**: [30, 50, ALL]
- **Justificativa**: Exploração completa da vizinhança

#### Shaking Intensity
- **Atual**: Fixo (1-3)
- **Testar**: Adaptativo (1-10) baseado em estagnação
- **Justificativa**: Hansen & Mladenović (2001)

#### Neighborhoods
- **Atual**: 4 neighborhoods genéricos
- **Novo**: 6 neighborhoods especializados
- **Justificativa**: Princípio VNS - vizinhanças diversas

### 2. Melhorias Estruturais

#### Memory-Based VNS
- Rastrear movimentos bem-sucedidos
- Bias futuras decisões
- Similar ao Q-Learning

#### Decomposition-Guided Shaking
- Usar vetores de peso como MOEA/D
- Shaking direcionado, não aleatório
- Híbrido VNS-Decomposition

#### Archive Management
- Primary Archive: 100 soluções (Pareto)
- Secondary Archive: 50 soluções (promissoras)
- Total: 150 soluções como MOEA/D

#### Adaptive Parameters
```python
if no_improvement > 5:
    shaking_intensity *= 1.5
    mobi_p_neighbors *= 2
else:
    shaking_intensity *= 0.9
    mobi_p_neighbors *= 0.9
```

## Implementação Proposta - MOVNS v4

### Configuração Base
```python
MOVNS_V4:
  archive_size: 150
  secondary_archive: 50
  max_iterations: 100
  mobi_p_neighbors: 50
  shaking_base: 2
  shaking_max: 10
  n_neighborhoods: 6
  adaptive: True
  memory_size: 100
```

### Neighborhoods Especializados

1. **N1_Greedy_LU**: Adiciona pacote com máxima co-ocorrência
2. **N2_Greedy_SS**: Adiciona pacote mais similar semanticamente
3. **N3_Size_Reduction**: Remove pacotes de baixa contribuição
4. **N4_Swap_Quality**: Troca por melhor qualidade
5. **N5_Path_Relinking**: Move em direção a solução do arquivo
6. **N6_Decomposition_Move**: Move em direção específica (peso)

### Pseudocódigo Melhorado
```
function MOVNS_V4(problem, params):
    archive = initialize_diverse(150)
    secondary = []
    memory = []
    no_improvement = 0

    for iteration in 1..max_iterations:
        solution = select_from_archive(archive)

        # Adaptive shaking
        if no_improvement > 5:
            intensity = min(10, base_intensity * 1.5^(no_improvement/5))
        else:
            intensity = base_intensity

        # Try all neighborhoods
        for k in 1..n_neighborhoods:
            x_prime = shake(solution, N[k], intensity)

            # Enhanced MOBI/P with more neighbors
            neighbors = mobi_p_enhanced(x_prime, 50)

            # Update both archives
            improved = update_archives(neighbors)

            if improved:
                memory.append((k, improvement_degree))
                k = bias_neighborhood_selection(memory)
                no_improvement = 0
            else:
                k = k + 1
                no_improvement += 1

        # Archive injection (like MOEA/D)
        if iteration % 5 == 0:
            inject_from_secondary(archive, secondary)

    return archive
```

## Experimentos Propostos

### Fase 1: Grid Search Básico
- Archive: [100, 150, 200]
- MOBI/P: [30, 50, 70]
- Iterations: [50, 75, 100]
- Total: 27 configurações

### Fase 2: Fine Tuning
- Melhor configuração da Fase 1
- Ajuste fino de:
  - Shaking adaptativo
  - Memory size
  - Injection frequency

### Fase 3: Validação
- 10 runs com melhor configuração
- Comparação estatística com MOEA/D
- Teste em múltiplos packages

## Métricas de Sucesso

1. **Hypervolume**: MOVNS v4 > MOEA/D (>0.26)
2. **Convergência**: Mais rápida que MOEA/D
3. **Diversidade**: Comparable ao MOEA/D
4. **Robustez**: Baixa variância entre runs

## Conclusão

Com calibração adequada baseada em literatura e exploração sistemática de hiperparâmetros, MOVNS pode superar MOEA/D através de:

1. **Archive maior** (150 vs 50 atual)
2. **MOBI/P expandido** (50 neighbors vs 10)
3. **Shaking adaptativo** (1-10 vs 1-3)
4. **Neighborhoods especializados** (6 vs 4)
5. **Memória e aprendizado** (novo)
6. **Decomposition híbrida** (novo)

Tempo estimado: 2-5 minutos por run (aceitável para calibração)