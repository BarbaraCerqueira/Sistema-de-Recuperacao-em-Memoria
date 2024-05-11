"""
Módulo: avaliacao.py
Data: 24/04/2024
Autor: Bárbara Cerqueira
Descrição: Este módulo implementa uma classe cuja função é avaliar o sistema de recuperação em memória baseado
           no modelo vetorial. Ela irá ler os arquivo sde resultados gerado pelo modelo e analisar os resultados,
           gerando uma série de medidas e diagramas em formato CSV e gráfico, os quais estão listados a seguir:
            1. Gráfico de 11 pontos de precisão e recall
            2. F1
            3. Precision@5
            4. Precision@10
            5. Histograma de R-Precision (comparativo)
            6. MAP
            7. MRR
            8. Discounted Cumulative Gain (médio)
            9. Normalized Discounted Cumulative Gain
"""

from nltk.metrics import precision, recall
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
import logging as log
import numpy as np
import pandas as pd
import csv
import os


class Evaluator:
    def __init__(self, results_file, expected_file, output_dir = '', identifier = None):
        self.results_file = results_file
        self.expected_file = expected_file
        self.output_dir = output_dir
        self.identifier = identifier  # Identificador da coleção de dados encontrados, será usada na terminação nos nomes dos arquivos gerados
        self.results = defaultdict(list)  # Para cada query, guarda uma lista de resultados (cada resultado é um dicionário com ranking, documento, pontuação de tf-idf)
        self.expected = defaultdict(list)  # Para cada query, guarda uma lista de resultados esperados (cada resultado é um dicionário com id do documento e número de votos)
        self.limit = 10  # Limite de rank para métricas em que tal decisão seja importante

    def __read_data(self):
        """
        Read results and expected results from the provided CSV files.
        """
        try:
            log.info("Reading obtained results...")
            with open(self.results_file, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                for row in reader:
                    query = int(row[0])
                    result = eval(row[1])
                    # Resultado está no formato [rank, doc_id, similarity_score]
                    self.results[query].append({
                        'rank': result[0],
                        'doc': result[1],
                        'score': result[2]
                    })

            log.info("Reading expected results...")
            with open(self.expected_file, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                header = next(reader)  # Pular o header
                for row in reader:
                    query = int(row[0])
                    doc_id = row[1]
                    votes = int(row[2])
                    self.expected[query].append({
                        'doc': doc_id,
                        'votes': votes
                    })

        except FileNotFoundError:
            log.error(f"Results file not found: {self.results_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def __plot_11_point_precision_recall_curve(self):
        """
        Plot the 11 point precision recall curve and save it as PDF file.
        """
        try:
            log.info("Plotting 11-point precision recall curve...")
            recall_levels = np.linspace(0, 1, num=11)  # 11 níveis de recall igualmente espaçados

            # Calculando a precisão por nivel de recall para cada query
            interpolated_precision_values = defaultdict(list)
            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query]]
                total_relevant_docs = len(relevant_docs)
                relevant_retrieved = 0
                precision_values = []
                recall_values = []

                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant_docs:
                        relevant_retrieved += 1
                    precision_values.append(relevant_retrieved / (i + 1))
                    recall_values.append(relevant_retrieved / total_relevant_docs)

                # Calculo da precisão interpolada para cada nível
                interpolated_precision = []
                for recall_level in recall_levels:
                    max_precision = max([precision_values[i] for i, recall_value in enumerate(recall_values) if recall_value >= recall_level], default=0)
                    interpolated_precision.append(max_precision)

                interpolated_precision_values[query] = interpolated_precision  # Lista ordenada com a precisão em cada recall level, de 0.0 a 1.0

            # Calculo da precisão média por nível de recall considerando todas as queries
            average_precisions = []
            for i, level in enumerate(recall_levels):
                precisions_at_level_i = [interpolated_precision_values[query][i] for query in self.expected]
                average_precisions.append(sum(precisions_at_level_i) / len(precisions_at_level_i))

            # Plota a curva de precisão-recall
            plt.plot(recall_levels, average_precisions, marker='o', label=self.identifier)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('11-Point Interpolated Precision-Recall Curve')
            plt.grid(True)
            plt.tight_layout()

            # Salva o gráfico em um PDF
            output_file_path = os.path.join(self.output_dir, f'11pontos-{self.identifier}.pdf')
            plt.savefig(output_file_path)
            plt.close()
            log.info(f"Graph was saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __f1_score(self):
        """
        Calculate the average F1 score for all queries using the top 10 retrieved and 
        expected documents for each query.
        """
        try:
            log.info("Calculating F1 Score...")
            f1_scores = []

            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= 10]  # Top 10 docs obtidos

                relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]

                # Calcular precisão e recall
                precision = len(relevant_retrieved) / len(retrieved_docs) if len(retrieved_docs) != 0 else 0
                recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) != 0 else 0

                # Calculo de F1
                if (precision + recall) == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

                f1_scores.append(f1)

            # Média dos F1 scores
            average_f1 = sum(f1_scores) / len(f1_scores) if len(f1_scores) != 0 else 0

            # Salva o F1 score médio em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'f1-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['F1_Score'])
                writer.writerow([average_f1])
                log.info(f"F1 score saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __precision_at_k(self, k):
        """
        Calculates precision for the top k retrieved documents for each query.
        """
        try:
            log.info(f"Calculating Precision@{k}...")
            precision_values = []

            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= k]  # Pegar os top k documentos recuperados
                relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))

                precision = relevant_retrieved / k if k != 0 else 0
                precision_values.append(precision)

            # Media da Precisão de todas as queries
            avg_precision = sum(precision_values) / len(precision_values)

            # Salva o score de Precision em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'precision@{k}-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'Precision@{k}'])
                writer.writerow([avg_precision])
                log.info(f"Precision@{k} saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __plot_r_precision_histogram(self):
        """
        Plot the R-Precision histogram and save it as PDF file.
        """
        try:
            log.info("Plotting R-Precision Histogram...")
            queries = list(self.expected.keys())
            r_precisions = []

            for query in queries:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= self.limit]  # Considera apenas os top 10 docs
                relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
                r_precision = len(relevant_retrieved) / len(relevant_docs)
                r_precisions.append(r_precision)
            
            # Plotando o histograma
            plt.bar(queries, r_precisions)
            plt.xlabel('Query')
            plt.ylabel('R-Precision')
            plt.title('R-Precision Histogram')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Salva o histograma em um PDF
            output_filename = os.path.join(self.output_dir, f'histograma-r-precision-{self.identifier}.pdf')
            plt.savefig(output_filename)
            plt.close()
            log.info(f"R-Precision Histogram saved in path: {output_filename}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __map(self):
        """
        Calculate Mean Average Precision for all queries.
        """
        try:
            log.info("Calculating Mean Average Precision (MAP)...")
            avg_precisions = []

            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= self.limit]  # Considera apenas os top 10 docs

                precision_values = []
                relevant_retrieved = 0

                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant_docs:
                        relevant_retrieved += 1
                    precision_values.append(relevant_retrieved / (i + 1))

                # Precisao média para a query
                avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
                avg_precisions.append(avg_precision)

            # Média das médias de precisão de todas as queries
            map_score = sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0.0

            # Salva o score de MAP em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'map-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'MAP'])
                writer.writerow([map_score])
                log.info(f"Mean Average Precision saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __mrr(self):
        """
        Calculate Mean Reciprocal Rank for all queries.
        """
        try:
            log.info("Calculating Mean Reciprocal Rank (MRR)...")
            reciprocal_ranks = []

            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query]]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= self.limit]  # Considera apenas os top 10 docs
                reciprocal_rank = 0.0  # default, permanecerá se nenhum doc relevante for encontrado

                # Obter o inverso da posição do primeiro documento relevante recuperado
                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant_docs:
                        reciprocal_rank = 1 / (i + 1)     
                        break

                reciprocal_ranks.append(reciprocal_rank)

            # Média dos reciprocal ranks de todas as queries
            mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

            # Salva o score de MRR em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'mrr-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'MRR'])
                writer.writerow([mrr_score])
                log.info(f"Mean Reciprocal Rank saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __dcg(self):
        """
        Calculate the average Discounted Cumulative Gain.
        """
        try:
            log.info("Calculating Discounted Cumulative Gain (DCG)...")
            dcg_values = []

            for query in self.expected:
                relevant_docs_votes = {doc['doc']: doc['votes'] for doc in self.expected[query]}
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= self.limit]  # Considera apenas os top 10 docs
                dcg = 0.0

                # Acumula os ganhos de cada documento recuperado usando o numero de votos, descontado de acordo com seu ranking obtido
                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant_docs_votes:
                        raw_gain = int(relevant_docs_votes[doc])
                    else:
                        raw_gain = 0
                    dcg += raw_gain / np.log2(i+1) if i != 0 else raw_gain

                dcg_values.append(dcg)

            # Média dos valores de dcg de todas as queries
            dcg_score = sum(dcg_values) / len(dcg_values) if dcg_values else 0.0

            # Salva o score de DCG em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'dcg-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'DCG'])
                writer.writerow([dcg_score])
                log.info(f"Discounted Cumulative Gain saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def __normalized_dcg(self):
        """
        Calculate the average Normalized Discounted Cumulative Gain.
        """
        try:
            log.info("Calculating Normalized Discounted Cumulative Gain (DCG)...")
            normal_dcg_values = []

            for query in self.expected:
                relevant_docs_votes = {doc['doc']: doc['votes'] for doc in self.expected[query]}
                sorted_expected = sorted(self.expected[query], key=itemgetter('votes'), reverse=True)

                # Garantir o mesmo tamanho para o vetor de docs obtido e o ótimo, para que haja consistência
                max_limit = min(self.limit, len(self.expected[query]))

                optimal_retrieved_order = [doc['doc'] for doc in sorted_expected][:max_limit]
                retrieved_docs = [result['doc'] for result in self.results[query] if result['rank'] <= max_limit]
                
                optimal_dcg = 0.0
                dcg = 0.0

                # Calculando o DCG real
                for i, doc in enumerate(retrieved_docs):
                    if doc in relevant_docs_votes:
                        raw_gain = int(relevant_docs_votes[doc])
                    else:
                        raw_gain = 0
                    dcg += raw_gain / np.log2(i+1) if i != 0 else raw_gain

                # Calculando o DCG ótimo
                for i, doc in enumerate(optimal_retrieved_order):
                    if doc in relevant_docs_votes:
                        raw_gain = int(relevant_docs_votes[doc])
                    else:
                        raw_gain = 0
                    optimal_dcg += raw_gain / np.log2(i+1) if i != 0 else raw_gain

                # DCG normalizado para a consulta
                normalized_dcg = dcg / optimal_dcg
                normal_dcg_values.append(normalized_dcg)

            # Média dos valores de dcg normalizado de todas as queries
            normal_dcg_score = sum(normal_dcg_values) / len(normal_dcg_values) if normal_dcg_values else 0.0

            # Salva o score de DCG normalizado em um arquivo CSV
            output_file_path = os.path.join(self.output_dir, f'normalized-dcg-{self.identifier}.csv')
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f'Normalized DCG'])
                writer.writerow([normal_dcg_score])
                log.info(f"Normalized DCG saved in path: {output_file_path}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def run(self):
        # Redefinindo ou inicializando os atributos para garantir um estado limpo
        self.results = defaultdict(list)
        self.expected = defaultdict(list)

        log.info("Evaluator started.")
        os.makedirs(self.output_dir, exist_ok=True)  # Cria o diretório de saída se não existir

        self.__read_data()
        self.__plot_11_point_precision_recall_curve()
        self.__f1_score()
        self.__precision_at_k(k=5)
        self.__precision_at_k(k=10)
        self.__plot_r_precision_histogram()
        self.__map()
        self.__mrr()
        self.__dcg()
        self.__normalized_dcg()
        log.info("Evaluator completed.")
