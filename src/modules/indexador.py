"""
Módulo: indexador.py
Descrição: Este módulo implementa um indexador, responsável por criar um modelo vetorial a partir de uma
           lista invertida simples usando a métrica tf-idf (Term Frequency-Inverse Document Frequency).
           O modelo vetorial é gerado na forma de um arquivo CSV.
Autor: Bárbara Cerqueira
Data: 08/04/2024
"""

import logging as log
import math
import csv
from collections import defaultdict

class Indexer:
    def __init__(self, config_file):
        self.config_file = config_file
        self.input_file = None
        self.output_file = None
        self.document_terms = defaultdict(dict)  # Guarda o número de vezes que um termo aparece em cada documento
        self.document_frequency = defaultdict(int)  # Guarda o número de documentos em que cada termo é encontrado
        self.tf_idf_scores = defaultdict(dict)
        self.total_documents = 0

    def __read_configuration(self):
        """
        Read the configuration file to get input and output file names.
        """
        log.info("Reading configuration file...")
        try:
            with open(self.config_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("LEIA="):
                        self.input_file = line.strip().split("=")[1]
                    elif line.startswith("ESCREVA="):
                        self.output_file = line.strip().split("=")[1]
        except FileNotFoundError:
            log.error(f"Configuration file not found: {self.config_file}")
            raise
        except OSError as e:
            log.error(f"Error reading configuration file: {e}")
            raise
    
    def __read_data(self):
        """
        Read inverted index from CSV and store data.
        """
        log.info("Reading inverted index from CSV file...")
        try:
            count = 0
            document_set = set()  # Guarda os ids de todos os documentos distintos, evitando repetição

            with open(self.input_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=';')
                for row in csv_reader:
                    count += 1
                    term = row[0]
                    document_ids = eval(row[1])  # Lista de IDs dos documentos onde o termo é encontrado

                    # Calculo do total de documentos distintos em que o termo aparece
                    self.document_frequency[term] = len(set(document_ids))

                    # Calculo da frequência do termo em cada documento
                    for doc_id in document_ids:
                        document_set.add(doc_id)
                        # Incrementa frequência do termo no documento
                        if term not in self.document_terms[doc_id]:
                            self.document_terms[doc_id][term] = 1
                        else:
                            self.document_terms[doc_id][term] += 1

            self.total_documents = len(document_set)

            log.info("Finished reading inverted index. Total rows read: %d", count)

        except FileNotFoundError:
            log.error(f"Inverted Index file not found: {self.input_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def __process_data(self):
        """
        Process retrieved data to create the term-document matrix and calculate tf-idf scores.
        """
        log.info("Processing term/document data...")
        for doc_id, terms in self.document_terms.items():
            max_term_freq = max(terms.values())

            for term, freq in terms.items():
                tf = freq / float(max_term_freq)
                idf = math.log(self.total_documents / self.document_frequency[term])
                self.tf_idf_scores[doc_id][term] = tf * idf

        log.info("Finished processing term/document data.")

    def __write_data(self):
        """
        Write the TF-IDF scores to the output file.
        """
        log.info("Writing tf-idf scores to file...")
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                for doc_id, terms_scores in self.tf_idf_scores.items():
                    for term, score in terms_scores.items():
                        writer.writerow([doc_id, term, score])
            log.info("Finished writing tf-idf scores.")

        except OSError as e:
            log.error(f"Failed to write output file: {e}")
            raise

    def run(self):
        """
        Main method to run the indexer.
        """
        # Redefinindo ou inicializando os atributos para garantir um estado limpo
        self.document_terms = defaultdict(dict)
        self.document_frequency = defaultdict(int)
        self.tf_idf_scores = defaultdict(dict)
        self.total_documents = 0

        log.info("Indexer started.")
        self.__read_configuration()
        self.__read_data()
        self.__process_data()
        self.__write_data()
        log.info("Indexer completed.")