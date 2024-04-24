"""
Módulo: buscador.py
Descrição: Este módulo implementa um buscador, responsável por obter os resultados de um conjunto de consultas 
           a partir de um modelo vetorial, procurando os documentos mais relevantes a cada consulta. 
Autor: Bárbara Cerqueira
Data: 11/04/2024
"""

from collections import defaultdict
import logging as log
import numpy as np
import csv

class SearchEngine:
    def __init__(self, config_file):
        self.config_file = config_file
        self.model_file = None
        self.queries_file = None
        self.results_file = None
        self.unique_terms = []  # Guarda todas as palavras distintas encontradas no corpus (documentos)
        self.model = defaultdict(dict)  # Guarda a pontuação de tf-idf dos termos de cada documento
        self.queries = defaultdict(list)  # Guarda a lista de palavras contida em cada query
        self.results = defaultdict(list)  # Guarda uma lista para cada query com os documentos, seu ranking e score de similaridade, ordenados por ranking

    def __read_configuration(self):
        """
        Read the configuration file to get model file, queries file, and results file names.
        """
        log.info("Reading configuration file...")
        try:
            with open(self.config_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("MODELO="):
                        self.model_file = line.strip().split("=")[1]
                    elif line.startswith("CONSULTAS="):
                        self.queries_file = line.strip().split("=")[1]
                    elif line.startswith("RESULTADOS="):
                        self.results_file = line.strip().split("=")[1]
        except FileNotFoundError:
            log.error(f"Configuration file not found: {self.config_file}")
            raise
        except OSError as e:
            log.error(f"Error reading configuration file: {e}")
            raise

    def __read_model(self):
        """
        Read model (TF-IDF scores) from the model file.
        """
        log.info("Reading tf-idf scores from file...")
        try:
            count = 0
            with open(self.model_file, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                for row in reader:
                    count += 1
                    doc_id, term, score = row
                    self.model[doc_id][term] = float(score)

            log.info("Finished reading tf-idf scores. Total scores read: %d", count)

        except FileNotFoundError:
            log.error(f"Model file not found: {self.model_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def __read_queries(self):
        """
        Read queries from the queries file.
        """
        log.info("Reading queries from file...")
        try:
            count = 0
            with open(self.queries_file, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                next(reader)  # Pula o header
                for row in reader:
                    count += 1
                    query_id = int(row[0])
                    query_text = row[1]
                    self.queries[query_id] = query_text.split()
            
            log.info("Finished reading queries. Total queries read: %d", count)

        except FileNotFoundError:
            log.error(f"Queries file not found: {self.queries_file}")
            raise
        except OSError as e:
            log.error(f"Error opening CSV file: {e}")
            raise

    def __process_data(self):
        """
        Process queries and find results.
        """
        log.info("Processing queries and tf-idf model data...")
        self.__build_unique_terms()

        log.info("Building document and query vectors...")
        query_vectors = self.__build_query_vectors()
        document_vectors = self.__build_document_vectors()

        log.info("Calculating similarity...")
        for query_id, query_vector in query_vectors.items():
            scores = defaultdict(float)  # Guarda a pontuação de similaridade de cada documento em relação a uma query

            for doc_id, doc_vector in document_vectors.items():
                scores[doc_id] = self.__calculate_similarity(query_vector, doc_vector)

            # Ordenação dos scores dos documentos em relação a query e armazenamento
            sorted_doc_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # Lista de tuplas (doc_id, score) em ordem decrescente de score
            self.results[query_id] = sorted_doc_scores

        log.info("Finished processing, similarity search concluded.")

    def __build_unique_terms(self):
        """
        Build a list with all unique words found in the documents.
        """
        unique_terms_set = set()
        for _, doc_term_scores in self.model.items():
            for term in doc_term_scores:
                unique_terms_set.add(term)
        self.unique_terms = sorted(unique_terms_set)  # Lista de todos os termos em ordem alfabética 

    def __build_document_vectors(self):
        """
        Build a vector for each document based on unique_terms.
        """
        document_vectors = defaultdict(list)
        for doc_id, terms in self.model.items():
            vector = np.array([terms.get(term, 0) for term in self.unique_terms])  # Pesos das palavras dos docs são seu score tf-idf
            document_vectors[doc_id] = vector
        return document_vectors

    def __build_query_vectors(self):
        """
        Build a vector for each query based on unique_terms.
        """
        query_vectors = defaultdict(list)
        for query_id, terms in self.queries.items():
            vector = np.array([1 if term in terms else 0 for term in self.unique_terms])  # Pesos das palavras da query são fixos em 1
            query_vectors[query_id] = vector
        return query_vectors

    def __calculate_similarity(self, vector1, vector2):
        """
        Calculate similarity between two vectors using cosine similarity.
        """
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product != 0:       
            similarity = dot_product / norm_product
        else:
            similarity = 0
        return similarity

    def __write_data(self):
        """
        Write search results to the results file.
        """
        log.info("Writing search results to file...")
        try:
            with open(self.results_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                for query_id, results in self.results.items():
                    ranking = 1
                    for doc_id, score in results:
                        writer.writerow([query_id, [ranking, doc_id, score]])
                        ranking += 1
            log.info("Finished writing search results.")

        except OSError as e:
            log.error(f"Failed to write output file: {e}")
            raise

    def run(self):
        """
        Main method to run the search engine.
        """
        # Redefinindo ou inicializando os atributos para garantir um estado limpo
        self.unique_terms = []
        self.model = defaultdict(dict)
        self.queries = defaultdict(list)
        self.results = defaultdict(list)

        log.info("Search engine started.")
        self.__read_configuration()
        self.__read_model()
        self.__read_queries()
        self.__process_data()
        self.__write_data()
        log.info("Search engine completed.")
