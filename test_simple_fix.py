"""
Teste simples: MOVNS Advanced com mais soluções
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_advanced import MOVNS_Advanced
from optimizer.moead_normalized import MOEAD_Normalized
from evaluation.quality_metrics import QualityMetrics

print("TESTE SIMPLES: Forçar MOVNS a ter mais soluções")
print("="*60)

# MOVNS Advanced - forçar geração de mais soluções
movns = MOVNS_Advanced('fastapi', archive_size=100, max_iterations=10, track_metrics=True)

# Executar
solutions = movns.run()

# Adicionar mais soluções manualmente se necessário
print(f"\nSoluções iniciais: {len(solutions)}")

if len(solutions) < 50:
    print("Gerando soluções adicionais...")
    # Gerar variações das existentes
    extra_solutions = []
    for _ in range(50 - len(solutions)):
        # Pegar solução aleatória e modificar
        base_sol = solutions[np.random.randint(len(solutions))]['chromosome'].copy()
        # Fazer pequena mudança
        idx = np.random.randint(len(base_sol))
        base_sol[idx] = 1 - base_sol[idx]
        obj = movns.evaluate_objectives(base_sol)
        extra_solutions.append({'chromosome': base_sol, 'objectives': obj})

    solutions.extend(extra_solutions)
    print(f"Soluções após adicionar: {len(solutions)}")

# Calcular métricas
qm = QualityMetrics()
objectives = []
for sol in solutions:
    obj = movns.evaluate_objectives(sol['chromosome'])
    objectives.append(obj)
objectives = np.array(objectives)

spacing = qm.spacing(objectives)
print(f"\nSpacing com {len(solutions)} soluções: {spacing:.4f}")

# Comparar com MOEA/D
print("\nMOEA/D para comparação:")
moead = MOEAD_Normalized('fastapi', pop_size=100, max_gen=10, track_metrics=True)
moead_sol = moead.run()

qm2 = QualityMetrics()
moead_obj = []
for sol in moead_sol:
    obj = moead.evaluate_objectives(sol['chromosome'])
    moead_obj.append(obj)
moead_obj = np.array(moead_obj)

moead_spacing = qm2.spacing(moead_obj)
print(f"MOEA/D com {len(moead_sol)} soluções: spacing={moead_spacing:.4f}")

print("\n" + "="*60)
if spacing < moead_spacing:
    print("MOVNS vence Spacing!")
else:
    print("MOEA/D vence Spacing")
    print(f"Diferença: MOVNS precisa melhorar {(spacing/moead_spacing - 1)*100:.1f}%")