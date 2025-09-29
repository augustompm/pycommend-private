# Requisitos de Gráficos e Estatísticas para Artigos MOEA (2023-2024)

## Baseado em Pesquisa de Estado da Arte

### 1. GRÁFICOS ESSENCIAIS

#### 1.1 Convergência de Hypervolume
**Padrão atual (2023-2024)**:
- **Linha principal**: Média de 30+ runs independentes
- **Intervalo de confiança**: Sombreado com 95% CI ou ±1 desvio padrão
- **Eixo X**: Gerações/Iterações (mínimo 30-50)
- **Eixo Y**: Hypervolume normalizado [0,1]
- **Múltiplos algoritmos**: 3-5 algoritmos no mesmo gráfico
- **Cores distintas**: Paleta colorblind-friendly

#### 1.2 Box Plots Comparativos
**Elementos necessários**:
- Box plots em gerações específicas (10, 20, 30, 40, 50)
- Comparação lado a lado dos algoritmos
- Outliers claramente marcados
- Mediana e média indicadas

#### 1.3 Evolução da Frente de Pareto
**Visualizações requeridas**:
- Scatter plots 2D/3D mostrando evolução
- Gerações: inicial (1), intermediárias (10, 20, 30), final (50)
- Frente de referência se disponível
- Cores/símbolos diferentes por algoritmo

#### 1.4 Violin Plots para Métricas Finais
**Métricas a incluir**:
- Hypervolume (HV)
- IGD+ (Inverted Generational Distance Plus)
- Spacing
- Spread/Diversity
- Runtime

### 2. MÉTRICAS ESTATÍSTICAS OBRIGATÓRIAS

#### 2.1 Testes de Significância
**Wilcoxon Signed-Rank Test** (padrão para MOEAs):
- Comparação pareada entre algoritmos
- p-value < 0.05 para significância
- Correção de Bonferroni para múltiplas comparações

#### 2.2 Estatísticas Descritivas
Para cada métrica, reportar:
- **Média** (mean)
- **Desvio padrão** (std)
- **Mediana** (median)
- **Quartis** (Q1, Q3)
- **Mínimo e Máximo**

#### 2.3 Taxa de Melhoria
```
Improvement Rate = (Final_HV - Initial_HV) / Initial_HV × 100%
```

### 3. TABELAS REQUERIDAS

#### 3.1 Tabela de Performance Comparativa
```latex
\begin{table}
Algoritmo | HV (mean±std) | IGD+ (mean±std) | Spacing | Time(s)
MOVNS     | 0.5616±0.032  | 0.142±0.021     | 0.023   | 85.3
MOEA/D    | 0.4355±0.041  | 0.187±0.031     | 0.031   | 92.1
NSGA-II   | 0.2340±0.055  | 0.234±0.042     | 0.045   | 78.5
\end{table}
```

#### 3.2 Tabela de Significância Estatística
```latex
\begin{table}
Comparison      | HV p-value | IGD+ p-value | Significant?
MOVNS vs MOEA/D | 0.0023    | 0.0012       | Yes
MOVNS vs NSGA-II| 0.0001    | 0.0001       | Yes
MOEA/D vs NSGA-II| 0.0156    | 0.0234       | Yes
\end{table}
```

### 4. CONFIGURAÇÕES EXPERIMENTAIS

#### 4.1 Parâmetros Mínimos
- **Runs independentes**: 30 (mínimo), 50 (recomendado)
- **Gerações/Iterações**: 50 (mínimo), 100 (recomendado)
- **População**: 100 indivíduos (padrão)
- **Sementes aleatórias**: Documentar todas

#### 4.2 Configuração de Hardware
Reportar:
- CPU modelo e frequência
- RAM disponível
- Sistema operacional
- Linguagem e versão

### 5. GRÁFICOS ADICIONAIS (OPCIONAIS MAS VALORIZADOS)

#### 5.1 Parallel Coordinates Plot
- Para soluções finais da Frente de Pareto
- Mostra trade-offs entre objetivos

#### 5.2 Heatmap de Correlação
- Correlação entre objetivos
- Útil para análise de conflito

#### 5.3 Radar/Spider Charts
- Comparação multi-critério dos algoritmos
- Normalizar todas métricas para [0,1]

### 6. EXEMPLO DE CÓDIGO PARA TRACKING

```python
class MetricsTracker:
    def __init__(self):
        self.history = {
            'generation': [],
            'hypervolume': [],
            'igd_plus': [],
            'spacing': [],
            'spread': [],
            'n_solutions': [],
            'cpu_time': []
        }

    def update(self, gen, solutions, reference_set=None):
        # Calculate all metrics
        hv = self.calculate_hypervolume(solutions)
        igd = self.calculate_igd_plus(solutions, reference_set)
        spacing = self.calculate_spacing(solutions)
        spread = self.calculate_spread(solutions)

        # Store in history
        self.history['generation'].append(gen)
        self.history['hypervolume'].append(hv)
        self.history['igd_plus'].append(igd)
        self.history['spacing'].append(spacing)
        self.history['spread'].append(spread)
        self.history['n_solutions'].append(len(solutions))
        self.history['cpu_time'].append(time.time())
```

### 7. CHECKLIST PARA PUBLICAÇÃO

- [ ] 30+ runs independentes
- [ ] 50+ gerações/iterações
- [ ] Gráfico de convergência HV com CI
- [ ] Box plots comparativos
- [ ] Teste Wilcoxon com p-values
- [ ] Tabela LaTeX com mean±std
- [ ] Evolução da Frente de Pareto
- [ ] Tempo computacional reportado
- [ ] Código reproduzível disponível
- [ ] Seeds aleatórias documentadas

### 8. REFERÊNCIAS DE PAPERS MODELO (2023-2024)

1. **Nature Scientific Reports (2024)**: "A many-objective evolutionary algorithm based on three states"
2. **IEEE TEVC (2023)**: Papers sobre MOEA/D melhorado
3. **MDPI Mathematics (2023)**: "NSGA-II/SDR-OLS with Opposition-Based Learning"
4. **Swarm and Evolutionary Computation (2024)**: Benchmarks com 30-50 runs

### 9. FERRAMENTAS RECOMENDADAS

- **Python**: matplotlib, seaborn, plotly
- **R**: ggplot2, plotly
- **LaTeX**: pgfplots, tikz
- **Métricas**: pymoo, jMetal, MOEA Framework

---
*Nota: Estes requisitos representam o padrão mínimo aceito em conferências e journals de alto impacto em 2023-2024*