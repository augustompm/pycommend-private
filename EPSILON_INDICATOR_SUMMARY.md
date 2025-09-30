# Epsilon-Indicator (ε-indicator) Summary

## O que é o ε-indicator?

O **ε-indicator** é uma métrica importante em otimização multi-objetivo que mede a **qualidade de convergência** de uma frente de Pareto em relação a uma frente de referência.

## Definição

O ε-indicator calcula o **menor valor ε** pelo qual uma aproximação A precisa ser "transladada" no espaço objetivo para **dominar fracamente** uma frente de referência R.

### Versão Aditiva (Implementada)
```
ε+(A,R) = max{r∈R} min{a∈A} max{i} (a_i - r_i)
```

- **A**: Conjunto aproximado (algoritmo testado)
- **R**: Conjunto de referência (frente Pareto ideal ou outro algoritmo)
- **Menor é melhor**: ε próximo de 0 indica melhor convergência

## Propriedades

1. **Weakly Pareto-compliant**: Respeita dominância fraca
2. **Unário ou binário**: Pode comparar com referência ideal ou entre algoritmos
3. **Sensível à convergência**: Detecta pequenas diferenças em proximidade à frente ótima

## Resultados na Apresentação

Na apresentação (main.tex), você reporta:
- **NSGA-II**: ε = 0.087
- **MOVNS**: ε = 0.062 (28.7% melhor)

## Resultados Atuais (v16)

Com 20 iterações:
- **MOVNS**: ε = 1.2833
- **MOEA/D**: ε = 0.0407

### Por que a diferença?

1. **Tamanho do arquivo**:
   - MOVNS: 23 soluções
   - MOEA/D: 100 soluções
   - Mais soluções = melhor cobertura = menor ε

2. **Trade-off HV vs ε**:
   - MOVNS otimiza para HV (volume dominado)
   - ε-indicator foca em proximidade/convergência

3. **Configuração diferente**:
   - Apresentação: MOVNS vs NSGA-II
   - Teste atual: MOVNS vs MOEA/D

## Interpretação

- **ε < 0.1**: Excelente convergência
- **ε < 0.5**: Boa convergência
- **ε > 1.0**: Convergência pobre

## Como Melhorar o ε-indicator

1. **Aumentar soluções no arquivo**: Melhor cobertura da frente
2. **Focar em convergência**: Busca local mais agressiva
3. **Balancear objetivos**: Evitar soluções extremas

## Código de Uso

```python
from evaluation.quality_metrics import QualityMetrics

qm = QualityMetrics()

# Calcular epsilon-indicator
epsilon = qm.epsilon_indicator(objectives_A, reference_set)
print(f"Epsilon-indicator: {epsilon:.4f}")
```

## Conclusão

O ε-indicator é complementar ao Hypervolume:
- **HV**: Mede volume dominado (convergência + diversidade)
- **ε-indicator**: Mede proximidade à frente ótima (convergência pura)

Para vencer em ambos, um algoritmo precisa:
1. Boa convergência (baixo ε)
2. Boa distribuição (bom HV)
3. Número adequado de soluções

---
*Documento criado em 2024-12-30*
*Projeto PyCommend v16*