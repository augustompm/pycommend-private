# File path: nsga2_recommender.py

import numpy as np
import pickle
import random
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
import time
import os

class NSGA2:
    def __init__(self, package_name, pop_size=100, max_gen=100):
        """
        Inicializa o algoritmo NSGA-II para recomendação de pacotes
        
        Args:
            package_name: Nome do pacote principal para o qual queremos recomendações
            pop_size: Tamanho da população
            max_gen: Número máximo de gerações
        """
        self.package_name = package_name.lower()
        self.pop_size = pop_size
        self.max_gen = max_gen
        
        # Carregar dados das matrizes
        self.load_data()
        
        # Verificar se o pacote principal existe nos dados
        if self.package_name not in self.pkg_to_idx:
            raise ValueError(f"Pacote '{package_name}' não encontrado nos dados.")
        
        # Determinar o tamanho dos cromossomos
        self.n_packages = len(self.package_names)
        
        # Probabilidades de operadores genéticos
        self.crossover_prob = 0.9
        self.mutation_prob = 1.0 / self.n_packages
        
        # Parâmetros de distribuição para SBX e Mutação polinomial
        self.eta_c = 20  # Índice de distribuição para crossover
        self.eta_m = 100  # Índice de distribuição para mutação
        
        # Tamanho mínimo e máximo para as soluções
        self.min_size = 3
        self.max_size = 15
        
        # Threshold para considerar uma conexão forte
        self.colink_threshold = self.calculate_colink_threshold()
        
        print(f"NSGA-II inicializado para recomendações ao pacote '{package_name}'")
        print(f"Número total de pacotes na base: {self.n_packages}")
        print(f"Tamanho mínimo de solução: {self.min_size}, máximo: {self.max_size}")
        print(f"Threshold para colink forte: {self.colink_threshold}")
        
    def calculate_colink_threshold(self):
        """Calcula um threshold para considerar um colink como forte"""
        # Amostra aleatória de valores para determinar o threshold
        sample_size = 10000
        sample_values = []
        
        # Coletar valores de amostra da matriz
        rows = np.random.randint(0, self.rel_matrix.shape[0], sample_size)
        cols = np.random.randint(0, self.rel_matrix.shape[1], sample_size)
        
        for i in range(sample_size):
            val = self.rel_matrix[rows[i], cols[i]]
            if val > 0:  # Considera apenas valores positivos
                sample_values.append(val)
        
        if not sample_values:
            return 1.0  # Valor padrão se não houver amostras positivas
            
        # Retorna o 75º percentil dos valores positivos
        return np.percentile(sample_values, 75)
        
    def load_data(self):
        """Carrega os dados das matrizes de relacionamento e similaridade"""
        # Carregar matriz de relacionamentos (F1)
        with open('data/package_relationships_10k.pkl', 'rb') as f:
            rel_data = pickle.load(f)

        self.rel_matrix = rel_data['matrix']
        self.package_names = rel_data['package_names']
        self.pkg_to_idx = {name.lower(): i for i, name in enumerate(self.package_names)}

        # Carregar matriz de similaridade semântica (F2)
        with open('data/package_similarity_matrix_10k.pkl', 'rb') as f:
            sim_data = pickle.load(f)

        self.sim_matrix = sim_data['similarity_matrix']
        
        print("Dados carregados com sucesso!")
        
    def fast_non_dominated_sort(self, P):
        """
        Implementa o algoritmo de ordenação não-dominada rápida descrito no artigo NSGA-II
        
        Args:
            P: População a ser ordenada
            
        Returns:
            F: Lista de fronteiras não-dominadas
        """
        S = [[] for _ in range(len(P))]  # Conjunto de soluções dominadas
        n = [0] * len(P)  # Contador de dominação
        F = [[]]  # Fronteiras
        
        # Para cada p em P
        for p_idx in range(len(P)):
            S[p_idx] = []
            n[p_idx] = 0
            
            # Para cada q em P
            for q_idx in range(len(P)):
                if p_idx == q_idx:
                    continue
                    
                # Se p domina q
                if self.dominates(P[p_idx]['objectives'], P[q_idx]['objectives']):
                    S[p_idx].append(q_idx)  # Adiciona q ao conjunto de soluções dominadas por p
                # Se q domina p
                elif self.dominates(P[q_idx]['objectives'], P[p_idx]['objectives']):
                    n[p_idx] += 1  # Incrementa o contador de dominação de p
            
            # Se p não é dominado por ninguém
            if n[p_idx] == 0:
                P[p_idx]['rank'] = 0  # p pertence à primeira fronteira
                F[0].append(p_idx)
        
        i = 0
        while F[i]:
            Q = []  # Usado para armazenar os membros da próxima fronteira
            
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:  # q não é dominado por ninguém na fronteira atual
                        P[q]['rank'] = i + 1
                        Q.append(q)
            
            i += 1
            F.append(Q)
            
        # Remove a última fronteira vazia
        F.pop()
        return F
    
    def dominates(self, obj1, obj2):
        """
        Verifica se obj1 domina obj2
        
        Args:
            obj1: Lista de valores de objetivos da solução 1
            obj2: Lista de valores de objetivos da solução 2
            
        Returns:
            True se obj1 domina obj2, False caso contrário
        """
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # Para minimização
                return False
            elif obj1[i] < obj2[i]:  # Para minimização
                better_in_any = True
                
        return better_in_any
    
    def crowding_distance_assignment(self, I, P):
        """
        Calcula a distância de crowding para um conjunto de soluções
        
        Args:
            I: Índices das soluções na população P
            P: População
        """
        l = len(I)
        if l == 0:
            return
        
        # Inicializa a distância para todos os indivíduos
        for i in I:
            P[i]['crowding_distance'] = 0
        
        # Para cada objetivo
        for m in range(3):  # Assumindo 3 objetivos: F1, F2, F3
            # Ordena a população pelo objetivo m
            sorted_indices = sorted(I, key=lambda i: P[i]['objectives'][m])
            
            # Atribui valores infinitos aos extremos
            P[sorted_indices[0]]['crowding_distance'] = float('inf')
            P[sorted_indices[-1]]['crowding_distance'] = float('inf')
            
            # Calcula a distância para os demais indivíduos
            f_max = P[sorted_indices[-1]]['objectives'][m]
            f_min = P[sorted_indices[0]]['objectives'][m]
            
            if f_max == f_min:
                continue
                
            # Para os indivíduos do meio
            for i in range(1, l-1):
                P[sorted_indices[i]]['crowding_distance'] += (
                    P[sorted_indices[i+1]]['objectives'][m] - 
                    P[sorted_indices[i-1]]['objectives'][m]
                ) / (f_max - f_min)
    
    def crowded_comparison_operator(self, i, j, P):
        """
        Implementa o operador de comparação de crowd
        
        Args:
            i, j: Índices dos indivíduos a serem comparados
            P: População
            
        Returns:
            True se i é preferido sobre j, False caso contrário
        """
        if P[i]['rank'] < P[j]['rank']:
            return True  # i tem rank melhor (menor) que j
        elif P[i]['rank'] == P[j]['rank'] and P[i]['crowding_distance'] > P[j]['crowding_distance']:
            return True  # Mesmo rank, mas i tem maior diversidade
        return False
    
    def tournament_selection(self, P):
        """
        Implementa a seleção por torneio usando o operador de comparação de crowd
        
        Args:
            P: População
            
        Returns:
            Índice do indivíduo selecionado
        """
        i = random.randint(0, len(P) - 1)
        j = random.randint(0, len(P) - 1)
        
        if self.crowded_comparison_operator(i, j, P):
            return i
        else:
            return j
    
    def create_individual(self):
        """
        Cria um indivíduo aleatório (vetor binário) excluindo o pacote principal
        e com tamanho entre min_size e max_size
        
        Returns:
            Dicionário representando o indivíduo
        """
        # Cria um vetor binário vazio
        chromosome = np.zeros(self.n_packages, dtype=np.int8)
        
        # Número de pacotes a selecionar (entre min_size e max_size)
        num_selected = random.randint(self.min_size, self.max_size)
        
        # Seleciona índices aleatoriamente (excluindo o pacote principal)
        indices = random.sample([i for i in range(self.n_packages) 
                                if i != self.pkg_to_idx[self.package_name]], num_selected)
        
        chromosome[indices] = 1
        
        # Avalia os objetivos
        objectives = self.evaluate_objectives(chromosome)
        
        return {
            'chromosome': chromosome,
            'objectives': objectives,
            'rank': None,
            'crowding_distance': 0
        }
    
    def evaluate_objectives(self, chromosome):
        """
        Avalia os objetivos para um cromossomo
        
        Args:
            chromosome: Vetor binário representando os pacotes selecionados
            
        Returns:
            Lista de valores dos objetivos [f1, f2, f3]
        """
        # Índices dos pacotes selecionados (incluindo o pacote principal)
        package_idx = self.pkg_to_idx[self.package_name]
        selected_indices = [i for i in range(self.n_packages) if chromosome[i] == 1]
        all_indices = selected_indices + [package_idx]
        
        # Tamanho da solução (F3)
        size = len(selected_indices)
        
        # Se solução vazia ou muito pequena, penalizar todos os objetivos
        if size < self.min_size:
            return [float('inf'), float('inf'), float('inf')]
        
        # F1: Linked usage modificado (minimizar o negativo)
        # Aqui fazemos a modificação para favorecer diversidade de conexões
        total_linked_usage = 0.0
        count = 0
        strong_connections = set()
        
        for i in range(len(all_indices)):
            for j in range(i+1, len(all_indices)):
                link_value = self.rel_matrix[all_indices[i], all_indices[j]]
                total_linked_usage += link_value
                count += 1
                
                # Conta pacotes com conexões fortes
                if link_value > self.colink_threshold:
                    strong_connections.add(all_indices[i])
                    strong_connections.add(all_indices[j])
        
        # Média básica de linked usage
        avg_linked_usage = total_linked_usage / count if count > 0 else 0.0
        
        # Ajusta F1 para considerar diversidade de conexões
        # Remove o pacote principal da contagem
        if package_idx in strong_connections:
            strong_connections.remove(package_idx)
        
        # Proporção de pacotes com conexões fortes
        proportion_strong = len(strong_connections) / size if size > 0 else 0
        
        # F1 modificado - ponderado pela proporção de pacotes com conexões fortes
        # Multiplicador logarítmico para suavizar o efeito
        if proportion_strong > 0:
            f1 = -avg_linked_usage * (1 + np.log1p(proportion_strong))
        else:
            f1 = -avg_linked_usage
            
        # F2: Similaridade semântica (minimizar o negativo)
        total_sim = 0.0
        count = 0
        
        for idx in selected_indices:
            total_sim += self.sim_matrix[package_idx, idx]
            count += 1
        
        f2 = -total_sim / count if count > 0 else 0.0
        
        # F3: Tamanho do conjunto (minimizar)
        f3 = size
        
        return [f1, f2, f3]
    
    def repair_chromosome(self, chromosome):
        """
        Repara um cromossomo para garantir que atenda às restrições de tamanho
        
        Args:
            chromosome: Vetor binário a ser reparado
            
        Returns:
            Vetor binário reparado
        """
        repaired = chromosome.copy()
        selected_indices = np.where(repaired == 1)[0]
        size = len(selected_indices)
        
        # Se o tamanho for menor que o mínimo, adicionar pacotes aleatórios
        if size < self.min_size:
            non_selected = [i for i in range(self.n_packages) 
                           if repaired[i] == 0 and i != self.pkg_to_idx[self.package_name]]
            
            # Quantos pacotes precisamos adicionar
            to_add = self.min_size - size
            
            if to_add > 0 and non_selected:
                # Adiciona pacotes aleatórios até atingir o mínimo
                add_indices = random.sample(non_selected, min(to_add, len(non_selected)))
                repaired[add_indices] = 1
        
        # Se o tamanho for maior que o máximo, remover pacotes aleatórios
        elif size > self.max_size:
            # Quantos pacotes precisamos remover
            to_remove = size - self.max_size
            
            if to_remove > 0:
                # Remove pacotes aleatórios até atingir o máximo
                remove_indices = random.sample(list(selected_indices), to_remove)
                repaired[remove_indices] = 0
        
        return repaired
    
    def sbx_crossover(self, parent1, parent2):
        """
        Implementa o operador de crossover SBX adaptado para vetores binários
        
        Args:
            parent1, parent2: Cromossomos dos pais
            
        Returns:
            Dois cromossomos filhos
        """
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # Para cada gene
        for i in range(len(parent1)):
            # Aplicar crossover uniforme com probabilidade de troca
            if random.random() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        # Repara os filhos para garantir tamanho mínimo e máximo
        child1 = self.repair_chromosome(child1)
        child2 = self.repair_chromosome(child2)
        
        return child1, child2
    
    def polynomial_mutation(self, chromosome):
        """
        Implementa o operador de mutação polinomial adaptado para vetores binários
        
        Args:
            chromosome: Cromossomo a ser mutado
            
        Returns:
            Cromossomo mutado
        """
        mutated = chromosome.copy()
        
        # Para cada gene
        for i in range(len(chromosome)):
            # Pula o pacote principal
            if i == self.pkg_to_idx[self.package_name]:
                continue
                
            # Aplica mutação com probabilidade mutation_prob
            if random.random() < self.mutation_prob:
                mutated[i] = 1 - mutated[i]  # Inverte o bit
        
        # Repara o cromossomo mutado
        mutated = self.repair_chromosome(mutated)
        
        return mutated
    
    def make_new_pop(self, P):
        """
        Cria uma nova população através de seleção, crossover e mutação
        
        Args:
            P: População atual
            
        Returns:
            Nova população
        """
        Q = []
        
        while len(Q) < len(P):
            # Seleção
            parent1_idx = self.tournament_selection(P)
            parent2_idx = self.tournament_selection(P)
            
            # Crossover
            child1_chrom, child2_chrom = self.sbx_crossover(
                P[parent1_idx]['chromosome'],
                P[parent2_idx]['chromosome']
            )
            
            # Mutação
            child1_chrom = self.polynomial_mutation(child1_chrom)
            child2_chrom = self.polynomial_mutation(child2_chrom)
            
            # Avaliação
            child1 = {
                'chromosome': child1_chrom,
                'objectives': self.evaluate_objectives(child1_chrom),
                'rank': None,
                'crowding_distance': 0
            }
            
            child2 = {
                'chromosome': child2_chrom,
                'objectives': self.evaluate_objectives(child2_chrom),
                'rank': None,
                'crowding_distance': 0
            }
            
            Q.append(child1)
            
            # Adicionar o segundo filho apenas se ainda não atingiu o tamanho da população
            if len(Q) < len(P):
                Q.append(child2)
        
        return Q
    
    def run(self):
        """
        Executa o algoritmo NSGA-II
        
        Returns:
            Lista de soluções não-dominadas
        """
        start_time = time.time()
        
        # Inicializa a população
        P = [self.create_individual() for _ in range(self.pop_size)]
        
        # Classifica a população inicial
        F = self.fast_non_dominated_sort(P)
        
        # Calcula a distância de crowding
        for front in F:
            self.crowding_distance_assignment(front, P)
        
        # Cria a primeira geração de descendentes
        Q = self.make_new_pop(P)
        
        # Loop principal do NSGA-II
        for gen in range(self.max_gen):
            # Combina as populações P e Q
            R = P + Q
            
            # Classifica a população combinada
            F = self.fast_non_dominated_sort(R)
            
            # Constrói a nova população P
            P_next = []
            i = 0
            
            while len(P_next) + len(F[i]) <= self.pop_size:
                # Calcula a distância de crowding para a fronteira atual
                self.crowding_distance_assignment(F[i], R)
                
                # Adiciona a fronteira atual a P_next
                P_next.extend([R[j] for j in F[i]])
                i += 1
            
            # Se P_next ainda não está completo
            if len(P_next) < self.pop_size:
                # Calcula a distância de crowding para a fronteira atual
                self.crowding_distance_assignment(F[i], R)
                
                # Ordena a fronteira atual pela distância de crowding
                sorted_front = sorted(F[i], 
                                     key=lambda j: R[j]['crowding_distance'], 
                                     reverse=True)
                
                # Adiciona as soluções necessárias para completar P_next
                P_next.extend([R[j] for j in sorted_front[:self.pop_size - len(P_next)]])
            
            # Atualiza P
            P = P_next
            
            # Cria a nova geração de descendentes
            Q = self.make_new_pop(P)
            
            # Exibe progresso a cada 10 gerações
            if gen % 10 == 0 or gen == self.max_gen - 1:
                elapsed_time = time.time() - start_time
                print(f"Geração {gen} de {self.max_gen} concluída ({elapsed_time:.2f}s)")
                
                # Exibe métricas sobre a primeira fronteira
                first_front_size = len(F[0])
                first_front_avg_f1 = np.mean([-R[j]['objectives'][0] for j in F[0]])
                first_front_avg_f2 = np.mean([-R[j]['objectives'][1] for j in F[0]])
                first_front_avg_f3 = np.mean([R[j]['objectives'][2] for j in F[0]])
                
                print(f"  Primeira fronteira: {first_front_size} soluções")
                print(f"  F1 (linked usage) média: {first_front_avg_f1:.4f}")
                print(f"  F2 (similaridade) média: {first_front_avg_f2:.4f}")
                print(f"  F3 (tamanho) média: {first_front_avg_f3:.2f}")
        
        # Obtém a fronteira final não-dominada
        F = self.fast_non_dominated_sort(P)
        final_front = [P[i] for i in F[0]]
        
        # Retorna a fronteira final
        elapsed_time = time.time() - start_time
        print(f"NSGA-II concluído em {elapsed_time:.2f}s")
        print(f"Fronteira final: {len(final_front)} soluções")
        
        return final_front
    
    def format_results(self, final_front):
        """
        Formata os resultados em um DataFrame para salvar em CSV,
        removendo duplicatas e ordenando por F3, F2, F1
        
        Args:
            final_front: Lista de soluções na fronteira final
            
        Returns:
            DataFrame com os resultados
        """
        results = []
        
        # Conjunto para rastrear combinações únicas de pacotes
        unique_package_sets = {}
        
        for i, solution in enumerate(final_front):
            # Converte os valores de objetivos (invertendo F1 e F2 para maximização)
            f1 = -solution['objectives'][0]  # Linked usage (maximizar)
            f2 = -solution['objectives'][1]  # Similaridade semântica (maximizar)
            f3 = solution['objectives'][2]   # Tamanho do conjunto (minimizar)
            
            # Obtém os pacotes recomendados
            selected_indices = np.where(solution['chromosome'] == 1)[0]
            recommended_packages = sorted([self.package_names[i] for i in selected_indices])
            
            # Cria uma chave única baseada nos pacotes recomendados
            package_key = ','.join(recommended_packages)
            
            # Verifica se esta combinação de pacotes já foi vista
            if package_key not in unique_package_sets:
                unique_package_sets[package_key] = {
                    'solution_id': i + 1,
                    'linked_usage': f1,
                    'semantic_similarity': f2,
                    'size': f3,
                    'recommended_packages': ','.join(recommended_packages)
                }
        
        # Converte o dicionário de soluções únicas em uma lista
        results = list(unique_package_sets.values())
        
        # Cria um DataFrame
        df = pd.DataFrame(results)
        
        # Ordena por F3 (crescente), F2 (decrescente), F1 (decrescente)
        df = df.sort_values(by=['size', 'semantic_similarity', 'linked_usage'], 
                           ascending=[True, False, False])
        
        # Renumera os IDs de solução após a ordenação
        df['solution_id'] = range(1, len(df) + 1)
        
        print(f"Encontradas {len(df)} soluções únicas após filtrar duplicatas")
        
        return df
    
    def save_results(self, df, output_dir='results'):
        """
        Salva os resultados em um arquivo CSV
        
        Args:
            df: DataFrame com os resultados
            output_dir: Diretório de saída
        """
        # Cria o diretório se não existir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Nome do arquivo baseado no pacote principal
        filename = f"{self.package_name}_recommendations.csv"
        file_path = os.path.join(output_dir, filename)
        
        # Salva o DataFrame como CSV
        df.to_csv(file_path, index=False)
        print(f"Resultados salvos em {file_path}")

def main():
    """Função principal para executar o NSGA-II para recomendação de pacotes"""
    parser = argparse.ArgumentParser(description='NSGA-II para recomendação de pacotes Python')
    
    parser.add_argument('package', type=str, help='Nome do pacote principal')
    parser.add_argument('--pop_size', type=int, default=100, help='Tamanho da população')
    parser.add_argument('--generations', type=int, default=100, help='Número de gerações')
    parser.add_argument('--output_dir', type=str, default='results', help='Diretório de saída')
    
    args = parser.parse_args()
    
    try:
        # Inicializa e executa o NSGA-II
        nsga2 = NSGA2(args.package, args.pop_size, args.generations)
        final_front = nsga2.run()
        
        # Formata e salva os resultados
        results_df = nsga2.format_results(final_front)
        nsga2.save_results(results_df, args.output_dir)
        
    except Exception as e:
        print(f"Erro: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()