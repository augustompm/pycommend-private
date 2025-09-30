"""
Identificar o gargalo no MOVNS Robust
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'pycommend-code/src'))
os.chdir(os.path.join(os.path.dirname(__file__), 'pycommend-code'))

from optimizer.movns_robust import MOVNS_Robust

print("Teste de gargalo - MOVNS Robust")
print("="*60)

# Criar instância
print("\n1. Criando instância...")
start = time.time()
movns = MOVNS_Robust('fastapi', archive_size=100, max_iterations=2, track_metrics=True)
print(f"   Tempo: {time.time() - start:.1f}s")

# Testar inicialização
print("\n2. Testando aggressive_initialization...")
start = time.time()
movns.aggressive_initialization()
print(f"   Tempo: {time.time() - start:.1f}s")
print(f"   Archive size: {len(movns.archive)}")

# Testar uma iteração de busca local
print("\n3. Testando local_search_intensive...")
if len(movns.archive) > 0:
    test_solution = movns.archive[0]['chromosome'].copy()
    start = time.time()
    improved = movns.local_search_intensive(test_solution)
    print(f"   Tempo: {time.time() - start:.1f}s")

# Testar geração de solução híbrida
print("\n4. Testando generate_hybrid_solution...")
start = time.time()
for i in range(10):
    sol = movns.generate_hybrid_solution()
print(f"   Tempo para 10 soluções: {time.time() - start:.1f}s")

print("\nAnálise completa")