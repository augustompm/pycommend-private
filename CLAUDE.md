# PyCommend - Memória do Projeto (Atualizada 2025-01-27)

## Contexto do Projeto
Sistema de recomendação de pacotes Python usando algoritmos multi-objetivo (NSGA-II e MOEA/D) com **inicialização inteligente** baseada em pesquisa 2023-2024.

## Estrutura Principal
- **pycommend-code/**: Código dos algoritmos de otimização
- **pycommend-collect/**: Scripts de coleta de dados do GitHub
- **article/**: Documentação técnica e bibliografia verificada
- **temp/**: Scripts de análise, debug e conteúdo antigo

## Regras do Projeto (rules.json)
- Sem comentários inline no código (apenas docstrings)
- Seguir padrões Python estabelecidos
- Documentação técnica na pasta article/
- Sem emojis no código

## 🎯 SOLUÇÃO IMPLEMENTADA: Weighted Probability Initialization

### Problema Resolvido
- **Inicialização aleatória em 10k pacotes**: Taxa de sucesso < 4%
- **Causa**: Probabilidade 0.01% de escolher pacotes relevantes aleatoriamente

### Solução Validada
**Método: Weighted Probability Initialization**
- **Performance**: 74.3% taxa de acerto (18.6x melhor que aleatório)
- **Força de conexão**: 2488.4 avg (62.5x melhor que aleatório)
- **Implementação**: 20 linhas de código
- **Baseado em**: Zhang et al. (2023) e Sharma & Trivedi (2020)

### Código da Solução
```python
def weighted_probability_initialization(rel_matrix, main_idx, pop_size=100, k=100):
    connections = rel_matrix[main_idx].toarray().flatten()
    ranked = np.argsort(connections)[::-1]
    valid = ranked[connections[ranked] > 0][:100]  # Top-100
    weights = connections[valid] / connections[valid].sum()

    population = []
    for _ in range(pop_size):
        size = random.randint(3, 15)
        individual = np.random.choice(valid, size, replace=False, p=weights)
        population.append(individual)
    return population
```

## Algoritmos Atualizados

### NSGA-II (src/optimizer/nsga2.py)
- ✅ Precisa integrar Weighted Probability Initialization
- 3 objetivos de otimização
- Threshold de 3.0-4.0 para conexões fortes

### MOEA/D (src/optimizer/moead.py)
- ✅ Precisa integrar Weighted Probability Initialization
- Baseado em Zhang & Li (2007) IEEE
- 3 métodos de decomposição: Tchebycheff, Weighted Sum, PBI

## Dados

### Matriz de Relacionamento
- **Arquivo**: data/package_relationships_10k.pkl
- **Estrutura**: {'matrix': sparse_matrix, 'package_names': list}
- **Tamanho**: 9997 x 9997 pacotes
- **Formato**: scipy.sparse.csr_matrix
- **Esparsidade**: 98.36%

### Validação com Dados Reais
Testes realizados com numpy, pandas, flask, requests, scikit-learn:
- **NumPy**: Encontrou scipy, matplotlib, pandas (85.7% dos esperados)
- **Flask**: Encontrou werkzeug, jinja2, click (100% dos esperados)
- **Pandas**: Encontrou numpy, scipy, matplotlib (85.7% dos esperados)

## Arquivos Importantes Atuais

### Documentação Verificada (article/)
- `constructive.md` - Documentação completa do Weighted Probability
- `audit-sources.md` - Auditoria das fontes científicas
- `verified-sources.md` - Fontes confirmadas como reais
- `test-results-summary.md` - Resultados dos testes com dados reais

### Implementações (temp/)
- `test_initialization_methods.py` - Testes unitários completos
- `simple_best_method.py` - Implementação limpa do método vencedor

### Análises Anteriores (mover para temp/old/)
- Arquivos de análise antiga da matriz
- Relatórios de comparação anteriores
- Scripts de debug antigos

## Próximos Passos Imediatos

1. **Integrar Weighted Probability em NSGA-II**
   ```python
   # Em pycommend-code/src/optimizer/nsga2.py
   from temp.simple_best_method import weighted_probability_initialization
   ```

2. **Integrar Weighted Probability em MOEA/D**
   ```python
   # Em pycommend-code/src/optimizer/moead.py
   from temp.simple_best_method import weighted_probability_initialization
   ```

3. **Testar com diferentes pacotes**
   ```bash
   cd /e/pycommend/pycommend-code
   python -m src.optimizer.nsga2 --package numpy
   python -m src.optimizer.moead --package numpy
   ```

## Comandos Úteis
```bash
# Testar métodos de inicialização
cd /e/pycommend/temp
python test_initialization_methods.py

# Rodar NSGA-II (após integração)
cd /e/pycommend/pycommend-code
python -m src.optimizer.nsga2 --package numpy

# Rodar MOEA/D (após integração)
python -m src.optimizer.moead --package numpy
```

## Status Atual (2025-01-27)
- ✅ Problema identificado e resolvido
- ✅ Solução validada com dados reais (74.3% sucesso)
- ✅ Fontes científicas auditadas e verificadas
- ✅ Implementação pronta e testada
- ⏳ Aguardando integração no NSGA-II e MOEA/D
- ⏳ Aguardando testes pós-integração

## Referências Científicas Verificadas
1. **Zhang et al. (2023)**: NSGA-II/SDR-OLS, Mathematics MDPI, vol. 11(8)
   - Link: https://www.mdpi.com/2227-7390/11/8/1911
2. **Sharma & Trivedi (2020)**: LHS-NSGA-III, Int. J. Construction Management
   - Link: https://www.tandfonline.com/doi/abs/10.1080/15623599.2020.1843769

## Métricas de Sucesso
- **Antes**: 4% taxa de acerto, 39.8 força média
- **Depois**: 74.3% taxa de acerto, 2488.4 força média
- **Melhoria**: 18.6x em descoberta, 62.5x em qualidade