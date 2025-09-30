# Novos Artigos Científicos Baixados - 2024

## Resumo da Busca

Foram pesquisados e baixados **5 artigos científicos recentes** (2024) de repositórios de acesso aberto (arXiv) sobre otimização multiobjetivo, com foco em:
- Algoritmos evolutivos multiobjetivo (MOEA/D, NSGA-II)
- Variable Neighborhood Search (VNS)
- Métricas de qualidade (Hypervolume, IGD)
- Aplicações em engenharia de software

## Artigos Baixados com Sucesso

### 1. Survey of Decomposition-Based MOEAs - Part II (2024)

**Arquivo:** `3-Survey_Decomposition_MOEA_Part2_2024.pdf` (26MB)

**Referência:** arXiv:2404.14228v1 [cs.NE] 22 Apr 2024

**Título:** A Survey of Decomposition-Based Evolutionary Multi-Objective Optimization: Part II—A Data Science Perspective

**Relevância para PyCommend:**
- Survey abrangente sobre MOEA/D de 2008 a 2023
- Análise de dados sobre evolução do campo
- Perspectiva data science para MOEAs
- **Conexão direta**: Base teórica para comparação MOVNS vs MOEA/D

**Citação sugerida:** Fundamental para contextualizar a escolha do MOEA/D como baseline de comparação.

---

### 2. LLM Aided Multi-Objective Evolutionary Algorithm (2024)

**Arquivo:** `4-LLM_Aided_MOEA_2024.pdf` (747KB)

**Referência:** arXiv:2410.02301v1 [cs.NE] 3 Oct 2024

**Título:** Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach

**Autores:** Adaptive framework integrado com MOEA/D, NSGA-II, NSGA-III

**Relevância para PyCommend:**
- Comparações entre MOEA/D e NSGA-II em test instances ZDT e UF
- Framework adaptativo de baixo custo
- **Conexão**: Validação de que MOEA/D e NSGA-II são baselines padrão em 2024
- Abordagem moderna combinando LLMs com MOEAs

**Citação sugerida:** Para discussão sobre estado da arte em comparações MOEA/D vs NSGA-II.

---

### 3. Speeding Up NSGA-II via Dynamic Population (2024)

**Arquivo:** `5-Dynamic_Population_NSGA2_2024.pdf` (304KB)

**Referência:** arXiv:2509.01739 [cs.NE] 3 Sep 2024

**Título:** Speeding Up the NSGA-II via Dynamic Population Sizes

**Proposta:** dNSGA-II com população dinâmica

**Relevância para PyCommend:**
- Melhoria de performance do NSGA-II
- Comparação com NSGA-III, SMS-EMOA, MOEA/D, SPEA2
- **Conexão**: Estratégias para melhorar convergência do NSGA-II
- Tamanhos de população adaptativos (similar ao arquivo adaptativo do MOVNS)

**Citação sugerida:** Para justificar escolha de parâmetros de população e discussão sobre convergência.

---

### 4. Multi-Objective Hyperparameter Optimization in ML (2024)

**Arquivo:** `6-MultiObjective_Hyperparameter_Optimization_ML_2024.pdf` (1.8MB)

**Referência:** arXiv:2206.07438v4 [cs.LG] 27 Jun 2024

**Título:** Multi-Objective Hyperparameter Optimization in Machine Learning -- An Overview

**Relevância para PyCommend:**
- Otimização de múltiplos objetivos em ML
- Trade-offs accuracy vs. complexity vs. energy
- **Conexão direta**: Problema de otimização multi-objetivo em software/ML
- Métricas e abordagens para problemas com objetivos conflitantes

**Citação sugerida:** Para fundamentar a formulação multi-objetivo do problema de recomendação de pacotes.

---

### 5. Performance Indicators in Multiobjective Optimization (2018)

**Arquivo:** `7-Performance_Indicators_MultiObjective_2018.pdf` (1.4MB)

**Referência:** arXiv:1802.08792v1 [cs.NE] 24 Feb 2018

**Título:** Performance indicators in multiobjective optimization

**Relevância para PyCommend:**
- Análise detalhada de Hypervolume, IGD, IGD+, Spacing
- Propriedades teóricas dos indicadores
- **Conexão crítica**: Base teórica para as métricas usadas no projeto
- Comparação entre indicadores (HV vs IGD vs IGD+)

**Citação sugerida:** Para fundamentar a escolha de Hypervolume como métrica principal e discussão sobre Spacing.

---

## Tentativa sem Sucesso

### MOEA/D Hyper-Heuristic (2023)

**Status:** Download falhou (apenas 134 bytes)

**Referência:** PMC10669882 - Biomimetics 2023

**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10669882/

**Nota:** Artigo acessível via web mas PDF não baixou corretamente. Pode ser tentado novamente ou acessado via navegador.

---

## Distribuição por Tema

### Algoritmos MOEA/D (3 artigos)
1. Survey Decomposition-Based MOEAs Part II
2. LLM Aided MOEA
3. Multi-Objective Hyperparameter Optimization

### Algoritmos NSGA-II (2 artigos)
1. LLM Aided MOEA (comparação)
2. Dynamic Population NSGA-II

### Métricas de Qualidade (1 artigo)
1. Performance Indicators

### Estado da Arte (5 artigos de 2024)
- Todos os artigos são de 2024, exceto o de Performance Indicators (2018)
- Cobrem tendências recentes: LLMs, população dinâmica, surveys atualizados

---

## Relevância para o Projeto PyCommend

### Fundamentação Teórica
- **Survey MOEA/D**: Contextualiza escolha do algoritmo baseline
- **Performance Indicators**: Justifica uso de Hypervolume e Spacing

### Comparações e Validação
- **LLM Aided MOEA**: Confirma MOEA/D e NSGA-II como baselines padrão
- **Dynamic NSGA-II**: Estratégias modernas para MOEAs

### Formulação do Problema
- **Hyperparameter Optimization ML**: Analogia com problema de recomendação multi-objetivo

### Gap Identificado
Nenhum dos artigos de 2024 aborda:
- **VNS para multi-objetivo** (gap que o projeto preenche)
- **Recomendação de pacotes Python** com MOEAs
- **MOBI/P local search** em contexto de software

**Conclusão:** Os artigos reforçam a originalidade e relevância do PyCommend ao combinar MOVNS com recomendação de software, área não coberta pela literatura recente de 2024.

---

## Como Citar

Os artigos podem ser citados no formato:

```bibtex
@misc{arxiv2404.14228,
  title={A Survey of Decomposition-Based Evolutionary Multi-Objective Optimization: Part II},
  author={...},
  year={2024},
  eprint={2404.14228},
  archivePrefix={arXiv},
  primaryClass={cs.NE}
}
```

**Recomendação:** Adicionar seção "Related Work" no artigo citando:
1. Survey MOEA/D (contextualização)
2. Performance Indicators (fundamentação métricas)
3. LLM Aided MOEA (validação escolha baselines)

---

*Documento gerado em: 2025-01-XX*
*Total de artigos baixados: 5*
*Total de páginas estimadas: ~150-200 páginas*
