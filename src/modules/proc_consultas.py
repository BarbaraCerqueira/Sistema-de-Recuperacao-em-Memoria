"""
Módulo: proc_consultas.py
Descrição: Este módulo implementa um processador de consultas, responsável por ler um arquivo XML contendo consultas 
           e seus resultados esperados, normalizar os textos das consultas e salvar os resultados em arquivos CSV.
Autor: Bárbara Cerqueira
Data: 04/04/2024
"""

import xml.etree.ElementTree as ET
from .utils.preprocessing import normalize_text
import logging as log
import csv

class QueryProcessor:
    def __init__(self, config_file):
        self.config_file = config_file
        self.queries_file = None
        self.output_queries_file = None
        self.output_expected_file = None
        self.queries = []
        self.expected = []
    
    def __read_configuration(self):
        """
        Read configuration file and extract relevant information.
        """
        log.info("Reading configuration file...")
        try:
            with open(self.config_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("LEIA="):
                        self.queries_file = line.strip().split("=")[1]
                    elif line.startswith("CONSULTAS="):
                        self.output_queries_file = line.strip().split("=")[1]
                    elif line.startswith("ESPERADOS="):
                        self.output_expected_file = line.strip().split("=")[1]
        except FileNotFoundError:
            log.error(f"Configuration file not found: {self.config_file}")
            raise
        except OSError as e:
            log.error(f"Error reading configuration file: {e}")
            raise
    
    def __read_data(self):
        """
        Read queries from XML file and store relevant information.
        """
        log.info("Reading queries...")
        try:
            tree = ET.parse(self.queries_file)
            root = tree.getroot()
            count = 0

            for query in root.findall('QUERY'):
                count += 1
                query_number = int(query.find('QueryNumber').text)
                query_text = query.find('QueryText').text
                records = query.find('Records')

                if records is not None:
                    for item in records.findall('Item'):
                        doc_number = int(item.text)
                        score_str = item.get('score')
                        # Calcula nota final do documento como a soma de todas as notas recebidas
                        doc_score = sum(map(lambda x: int(x) > 0, score_str))
                        self.expected.append((query_number, doc_number, doc_score))
                self.queries.append((query_number, query_text)) 

            log.info("Finished reading queries. Total queries read: %d", count)

        except ET.ParseError as e:
            log.error(f"Failed to parse XML file: {e}")
            raise
        except FileNotFoundError:
            log.error(f"Queries file not found: {self.queries_file}")
            raise
        except OSError as e:
            log.error(f"Error opening XML file: {e}")
            raise
    
    def __process_data(self):
        """
        Process queries to normalize text data.
        """
        log.info("Processing queries...")
        processed_queries = []
        for query_number, query_text in self.queries:
            processed_text = normalize_text(query_text)
            processed_queries.append((query_number, processed_text))
        
        self.queries = processed_queries       
        log.info("Finished processing queries.")

    def __write_data(self):
        """
        Write queries and expected results to CSV files.
        """
        log.info("Writing output files...")
        try:
            with open(self.output_queries_file, 'w', newline='', encoding='utf-8') as queries_file, \
             open(self.output_expected_file, 'w', newline='', encoding='utf-8') as expected_file:
                queries_writer = csv.writer(queries_file, delimiter=';')
                expected_writer = csv.writer(expected_file, delimiter=';')

                queries_writer.writerow(["QueryNumber", "QueryText"])
                expected_writer.writerow(["QueryNumber", "DocNumber", "DocVotes"])

                for queries_data in self.queries:
                    queries_writer.writerow(queries_data)
                for expected_data in self.expected:
                    expected_writer.writerow(expected_data)

            log.info("Finished writing output files.")

        except OSError as e:
            log.error(f"Failed to write output file: {e}")
            raise

    def run(self):
        """
        Main method to run the query processor.
        """
        log.info("Starting Query Processor...")
        self.__read_configuration()
        self.__read_data()
        self.__process_data()
        self.__write_data()
        log.info("Query Processor completed.")
