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

from collections import defaultdict
import logging as log
from operator import itemgetter
import numpy as np
import csv
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, results_file, expected_file, suffix = None):
        self.results_file = results_file
        self.expected_file = expected_file
        self.suffix = suffix  # Terminação nos nomes dos arquivos gerados e também identificador para algumas métricas
        self.results = defaultdict(list)  # Para cada query, guarda uma lista de resultados (cada resultado é um dicionário com ranking, documento, pontuação de tf-idf)
        self.expected = defaultdict(list)  # Para cada query, guarda uma lista de resultados esperados (cada resultado é um dicionário com id do documento e número de votos)

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
        Plot the 11 point precision recall curve for the data available. 
        """
        precisions = []
        recalls = []

        for query, query_results in self.results.items():
            expected_docs = set([res['doc'] for res in self.expected[query]])
            relevant_docs = 0
            total_retrieved_docs = 0
            relevant_and_retrieved_docs = 0

            for i, result in enumerate(query_results):
                total_retrieved_docs += 1
                if result['doc'] in expected_docs:
                    relevant_docs += 1
                    relevant_and_retrieved_docs += 1

                precision = relevant_and_retrieved_docs / total_retrieved_docs
                recall = relevant_and_retrieved_docs / len(expected_docs)

                # Se o recall atual for um dos 11 pontos pré-definidos, adicionamos precisão e recall à lista
                if round(recall, 1) == len(precisions) / 10:
                    precisions.append(precision)
                    recalls.append(recall)

        # Plotando o gráfico
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o')
        plt.title('11-point Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def run(self):
        log.info("Evaluator started.")
        self.__read_data()
        self.__plot_11_point_precision_recall_curve()
        log.info("Evaluator completed.")