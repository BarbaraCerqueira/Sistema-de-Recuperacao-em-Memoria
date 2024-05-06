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
    def __init__(self, results_file, expected_file, output_dir = '', suffix = None):
        self.results_file = results_file
        self.expected_file = expected_file
        self.output_dir = output_dir
        self.suffix = suffix  # Terminação nos nomes dos arquivos gerados
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

            # Plot the 11-point interpolated precision-recall curve
            plt.plot(recall_levels, average_precisions, marker='o')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('11-Point Interpolated Precision-Recall Curve')
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            output_filename = os.path.join(self.output_dir, f'11pontos-{self.suffix}.pdf')
            plt.savefig(output_filename)
            plt.close()
            log.info(f"Graph was saved in path: {output_filename}")

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
        log.info("Evaluator completed.")