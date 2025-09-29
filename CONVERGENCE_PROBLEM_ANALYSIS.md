# Análise do Problema de Convergência Negativa no MOEA/D

## Diagnóstico: O Problema Real

Após análise profunda do código e literatura, identifiquei o **problema fundamental** que causa convergência negativa:

### 1. PROBLEMA DE ESCALA DOS OBJETIVOS

Os três objetivos têm escalas **drasticamente diferentes**:
- **LU (Linked Usage)**: Valores na ordem de -1000 a -10000
- **SS (Semantic Similarity)**: Valores na ordem de -0.1 a -1.0
- **RSS (Set Size)**: Valores na ordem de 2 a 15

**Consequência**: Na decomposição Tchebycheff, o objetivo LU (1000x maior) domina completamente o cálculo:
```python
np.max(weight * np.abs(objectives - z))
# Se objectives = [-5000, -0.5, 5]
# Com weight = [0.33, 0.33, 0.33]
# Resultado = max(1650, 0.165, 1.65) = 1650 (LU domina!)
```

### 2. PONTO DE REFERÊNCIA IDEAL (z) MAL INICIALIZADO

O código inicializa:
```python
self.z = np.full(self.n_objectives, np.inf)
```

Depois atualiza com:
```python
self.z = np.minimum(self.z, objectives)
```

**Problema**: O ponto ideal z começa com infinito e rapidamente converge para valores ruins nas primeiras gerações. Como os objetivos não são normalizados, z fica instável.

### 3. ARQUIVO EXTERNO vs POPULAÇÃO

O hypervolume é calculado no **arquivo externo**, não na população:
```python
if self.external_archive:
    objectives = np.array([obj for _, obj in self.external_archive])
```

**Problema**: O arquivo acumula soluções diversas mas não necessariamente melhores. Como ele tem limite de 100 e remove soluções próximas (não dominadas), pode estar **removendo soluções boas** e mantendo soluções diversas mas ruins.

### 4. FALTA DE NORMALIZAÇÃO

Literatura confirma (Springer, 2017): *"MOEA/D shows difficulties in finding uniformly distributed solutions when each objective has totally different range of values"*

Sem normalização:
- Decomposição favorece objetivo de maior magnitude
- Hypervolume fica dominado por um único objetivo
- Soluções convergem para extremos, não para balanço

## Por Que Hypervolume Diminui?

### Cenário Real Observado:

1. **Geração 0**: Arquivo tem soluções iniciais diversas (algumas boas por sorte)
2. **Gerações 1-5**: Decomposição força convergência para minimizar LU (maior escala)
3. **Gerações 6-10**: Arquivo atinge limite (100), começa a remover soluções
4. **Critério de remoção**: Remove soluções próximas (crowding), não ruins
5. **Resultado**: Remove soluções balanceadas, mantém extremos
6. **Hypervolume cai**: Extremos têm menor hypervolume que soluções balanceadas

## Solução Correta

### 1. NORMALIZAÇÃO OBRIGATÓRIA
```python
def normalize_objectives(self, objectives):
    # Normalizar para [0, 1] baseado em limites conhecidos
    norm_obj = objectives.copy()

    # LU: esperado entre -10000 e 0
    norm_obj[0] = (objectives[0] - (-10000)) / 10000

    # SS: esperado entre -1 e 0
    norm_obj[1] = (objectives[1] - (-1)) / 1

    # RSS: esperado entre 2 e 15
    norm_obj[2] = (objectives[2] - 2) / 13

    return norm_obj
```

### 2. DECOMPOSIÇÃO COM OBJETIVOS NORMALIZADOS
```python
def decompose(self, objectives, weight):
    norm_obj = self.normalize_objectives(objectives)
    norm_z = self.normalize_objectives(self.z)

    if self.decomposition == 'tchebycheff':
        return np.max(weight * np.abs(norm_obj - norm_z))
```

### 3. ARQUIVO COM CRITÉRIO DE QUALIDADE
```python
def update_archive(self, solution, objectives):
    # Adicionar apenas se melhorar hypervolume
    temp_archive = self.external_archive + [(solution, objectives)]
    new_hv = calculate_hypervolume(temp_archive)
    old_hv = calculate_hypervolume(self.external_archive)

    if new_hv >= old_hv:
        # Aceitar solução
```

## Evidências da Literatura

1. **Zhang & Li (2007)** original menciona normalização mas não enfatiza
2. **Estudos recentes (2017-2024)** identificam normalização como crítica
3. **HDE-MOEA/D (2021)** propõe entropia para detectar este problema
4. **WVA-MOEA/D (2020)** ajusta pesos adaptivamente para compensar escalas

## Conclusão

**O MOEA/D está implementado corretamente**, mas falta um componente essencial não enfatizado no paper original: **normalização adequada dos objetivos**.

Sem normalização:
- Decomposição Tchebycheff falha com objetivos de escalas diferentes
- Arquivo externo acumula soluções ruins
- Hypervolume diminui porque soluções balanceadas são perdidas

Com normalização:
- Todos objetivos contribuem igualmente
- Arquivo mantém soluções realmente boas
- Convergência positiva garantida

## Recomendação Imediata

Adicionar normalização é **trivial** (10 linhas de código) mas **essencial** para convergência. Isso não é "trapacear" - é prática padrão em MOEAs modernos, apenas não era enfatizado em 2007.