"""
Módulo: avaliacao.py
Data: 24/04/2024
Autor: Bárbara Cerqueira
Descrição: Este módulo implementa uma classe cuja função é avaliar o sistema de recuperação em memória baseado
           no modelo vetorial. Ela irá ler o arquivo de resultados gerado pelo modelo e comparar com os resultados
           esperados, gerando uma série de medidas e diagramas em formato CSV e gráfico, os quais estão listados a seguir:
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
                    query = row[0]
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
                    query = row[0]
                    doc_id = row[1]
                    votes = int(row[2])
                    self.expected[query].append({
                        'doc': doc_id,
                        'votes': votes
                    })

                # Ordenar os documentos pelo número de votos (do maior para o menor)
                for query in self.expected:
                    self.expected[query] = sorted(self.expected[query], key=itemgetter('votes'), reverse=True)

        except FileNotFoundError:
            log.error(f"Results file not found: {self.results_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def __plot_11_point_precision_recall_curve(self):
        """
        Plot the 11 point precision recall curve and save it to CSV and PDF (graph) files.
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
        expected documents for each query and save the result in a CSV file.
        """
        try:
            log.info("Calculating F1 Score...")
            f1_scores = []

            for query in self.expected:
                relevant_docs = [doc['doc'] for doc in self.expected[query][:10]]  # Top 10 docs relevantes (mais votos)
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
                retrieved_docs = [result['doc'] for result in self.results[query]][:k]  # Pegar os top k documentos recuperados
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
        Plot the R-Precision histogram and save it to PDF.
        """
        try:
            log.info("Plotting R-Precision Histogram...")
            r_precisions = []
            for query in self.expected:
                relevant_docs = self.expected[query]  # R = relevant_docs
                retrieved_docs = self.results[query][:len(relevant_docs)]
                relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
                r_precision = len(relevant_retrieved) / len(relevant_docs)
                r_precisions.append(r_precision)
            
            # Desenhando o histograma
            plt.hist(np.arange(0, len(r_precisions), 1), bins=r_precisions, edgecolor='black', alpha=0.7)
            plt.xlabel('Query ID')
            plt.ylabel('R-Precision')
            plt.title('R-Precision Histogram')
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.grid(True)
            plt.tight_layout()

            # Salva o histograma em um PDF
            output_filename = os.path.join(self.output_dir, f'histograma-r-precision-{self.identifier}.pdf')
            plt.savefig(output_filename)
            plt.close()
            log.info(f"R-Precision Histogram saved in path: {output_filename}")

        except OSError as e:
            log.error(f"Failed to find output path or save file: {e}")

    def run(self):
        # Redefinindo ou inicializando os atributos para garantir um estado limpo
        self.results = defaultdict(list)
        self.expected = defaultdict(list)

        log.info("Evaluator started.")
        # Cria o diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)

        self.__read_data()
        self.__plot_11_point_precision_recall_curve()
        self.__f1_score()
        self.__precision_at_k(k=5)
        self.__precision_at_k(k=10)
        self.__plot_r_precision_histogram()
        log.info("Evaluator completed.")
