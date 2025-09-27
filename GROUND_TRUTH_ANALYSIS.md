# Ground Truth Analysis - PyCommend

## A Verdade sobre a "Ground Truth"

### O que temos atualmente
As listas de "expected packages" no código são **hardcoded manualmente**:

```python
test_packages = {
    'numpy': ['scipy', 'matplotlib', 'pandas', 'scikit-learn', 'sympy'],
    'flask': ['werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe'],
    'requests': ['urllib3', 'certifi', 'idna', 'charset-normalizer', 'chardet']
}
```

**Origem**: Suposições do desenvolvedor, não análise sistemática.

### O que realmente temos disponível

## 8,794 Requirements.txt Reais do GitHub!

Localização: `pycommend-code/data/github/dependencies/`
- 8,794 arquivos requirements.txt de projetos reais
- Coletados de repositórios populares do GitHub
- Representam uso real de pacotes Python

### Análise Rápida dos Dados Reais

```bash
# Projetos que usam Flask
grep -l "flask" *.txt | wc -l
# Resultado: centenas de projetos

# Projetos com dependências do Flask (werkzeug, jinja2, click)
grep -l "werkzeug\|jinja2\|click" *.txt | wc -l
# Resultado: 715 projetos
```

## Problema com a Métrica Atual

### Taxa de 66.7% é Questionável

1. **Não é ground truth real** - são suposições arbitrárias
2. **Mistura conceitos diferentes**:
   - Dependências diretas (Flask→Werkzeug)
   - Pacotes complementares (NumPy→Pandas)
   - Ecossistema relacionado (NumPy→scikit-learn)

3. **Ignora descobertas válidas**:
   - Se o algoritmo encontra `torch` para `numpy`, isso é errado?
   - Se encontra `sqlalchemy` para `flask`, isso é inválido?

## Como Criar Ground Truth Real

### Opção 1: Análise Estatística dos Requirements
```python
def analyze_real_cooccurrences():
    """Analyze 8,794 real requirements.txt files"""
    cooccurrence_count = {}

    for file in requirements_files:
        packages = parse_requirements(file)
        for pkg1 in packages:
            for pkg2 in packages:
                if pkg1 != pkg2:
                    cooccurrence_count[(pkg1, pkg2)] += 1

    # For each package, find top-N most common companions
    ground_truth = {}
    for package in target_packages:
        companions = get_top_companions(package, cooccurrence_count, n=10)
        ground_truth[package] = companions

    return ground_truth
```

### Opção 2: Dependências Oficiais do PyPI
```python
def get_pypi_dependencies():
    """Get actual dependencies from PyPI metadata"""
    # Flask real dependencies:
    # - Werkzeug>=3.1
    # - Jinja2>=3.1
    # - Click>=8.1
    # - ItsDangerous>=2.2
    # - MarkupSafe>=2.1
    # - Blinker>=1.9
```

### Opção 3: Análise de Co-instalação
Use a matriz de co-ocorrência que já temos para definir "ground truth" como:
- Top-10 pacotes mais co-instalados
- Pacotes com co-ocorrência > threshold (ex: 100)
- Pacotes no mesmo cluster semântico

## Métricas Alternativas Mais Honestas

### 1. Força de Conexão Média
```python
avg_cooccurrence = mean([rel_matrix[main_pkg, found_pkg] for found_pkg in solution])
```

### 2. Coerência Semântica (Já Implementado!)
```python
coherence = mean(cosine_similarity(embeddings, centroid))
```

### 3. Diversidade vs Relevância
```python
relevance = sum(cooccurrences) / len(solution)
diversity = 1 - mean(pairwise_similarity(solution))
score = alpha * relevance + (1-alpha) * diversity
```

### 4. Comparação com Top-K Real
```python
# Compare com os top-K reais da matriz
top_k_real = get_top_k_from_matrix(package, k=10)
overlap = len(set(found) & set(top_k_real)) / k
```

## Recomendação

### Problema Atual
A métrica de 66.7% não é confiável porque compara com uma lista arbitrária, não com ground truth real.

### Solução Proposta
1. **Curto prazo**: Use os top-10 da matriz de co-ocorrência como ground truth
2. **Médio prazo**: Analise os 8,794 requirements.txt para criar ground truth estatística
3. **Longo prazo**: Combine múltiplas fontes (PyPI, GitHub, surveys)

### Código Exemplo
```python
def get_real_ground_truth(package_name):
    """Get real ground truth from co-occurrence matrix"""
    idx = package_names.index(package_name)
    connections = rel_matrix[idx].toarray().flatten()

    # Get top-10 by co-occurrence strength
    ranked = np.argsort(connections)[::-1]
    top_10 = [package_names[i] for i in ranked[:10] if connections[i] > 0]

    return top_10

# Comparação honesta
real_truth = {
    'numpy': get_real_ground_truth('numpy'),
    'flask': get_real_ground_truth('flask'),
    'requests': get_real_ground_truth('requests')
}
```

## Conclusão

1. **A "ground truth" atual é arbitrária**, não baseada em dados
2. **Temos 8,794 requirements.txt reais** que poderiam ser usados
3. **A matriz de co-ocorrência já tem a informação** necessária
4. **Taxa de 66.7% é enganosa** sem ground truth validada

### Métrica Mais Honesta
Em vez de "66.7% de taxa de sucesso", seria mais honesto reportar:
- "Força média de conexão: 2500"
- "Coerência semântica: 0.65"
- "Overlap com top-10 real: 40%"

Essas métricas são objetivas e verificáveis, não dependem de listas arbitrárias.