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
import re
from collections import defaultdict

class Indexer:
    def __init__(self, config_file):
        self.config_file = config_file
        self.input_file = None
        self.output_file = None
        self.document_terms = defaultdict(dict)
        self.term_document_frequency = defaultdict(int)
        self.total_documents = 0

    def read_configuration(self):
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
        Read inverted index from CSV.
        """
        log.info("Reading inverted index from CSV file...")
        try:
            count = 0
            with open(self.input_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=';')
                for row in csv_reader:
                    count += 1
                    term = row[0]
                    document_ids = eval(row[1])  # Convert string representation of list to actual list
                    for doc_id in document_ids:
                        self.total_documents += 1
                        if term not in self.document_terms[doc_id]:
                            self.document_terms[doc_id][term] = 1
                        else:
                            self.document_terms[doc_id][term] += 1
                        self.term_document_frequency[term] += 1

            log.info("Finished reading inverted index. Total rows read: %d", count)

        except FileNotFoundError:
            log.error(f"Inverted Index file not found: {self.input_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def process_documents(self):
        """
        Process documents to create the term-document matrix and calculate tf-idf scores.
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=';')
                for row in csv_reader:
                    term = row[0]
                    document_ids = eval(row[1])  # Convert string representation of list to actual list
                    for doc_id in document_ids:
                        self.total_documents += 1
                        if term not in self.document_terms[doc_id]:
                            self.document_terms[doc_id][term] = 1
                        else:
                            self.document_terms[doc_id][term] += 1
                        self.term_document_frequency[term] += 1
        except Exception as e:
            log.error(f"Error processing documents: {e}")
            raise

    def calculate_tf_idf(self):
        """
        Calculate the TF-IDF score for each term in each document.
        """
        tf_idf_scores = defaultdict(dict)
        for doc_id, terms in self.document_terms.items():
            doc_term_count = sum(terms.values())
            for term, count in terms.items():
                tf = count / float(doc_term_count)
                idf = math.log(self.total_documents / (1 + self.term_document_frequency[term]))
                tf_idf_scores[doc_id][term] = tf * idf
        return tf_idf_scores

    def write_data(self, tf_idf_scores):
        """
        Write the TF-IDF scores to the output file.
        """
        log.info("Writing tf-idf scores to file...")
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
                csv_writer = csv.writer(file, delimiter=';')
                for doc_id, terms_scores in tf_idf_scores.items():
                    for term, score in terms_scores.items():
                        csv_writer.writerow([doc_id, term, score])
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
        self.term_document_frequency = defaultdict(int)
        self.total_documents = 0

        log.info("Starting Indexer...")
        self.read_configuration()
        self.process_documents()
        tf_idf_scores = self.calculate_tf_idf()
        self.write_output(tf_idf_scores)
        log.info("Indexer completed.")