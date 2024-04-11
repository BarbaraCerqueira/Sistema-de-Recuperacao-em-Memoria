"""
Módulo: lista_invertida.py
Descrição: Este módulo implementa um gerador de listas invertidas simples, responsável por processar um conjunto de 
           documentos em formato XML e construir uma lista invertida simples, que será escrita em um arquivo CSV.
Autor: Bárbara Cerqueira
Data: 06/04/2024
"""

import xml.etree.ElementTree as ET
from .utils.preprocessing import normalize_text
import logging as log
import csv

class InvertedListGenerator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.input_files = []
        self.output_file = None
        self.documents = []  # Guarda o registro e abstract dos documentos lidos 
        self.inverted_index = {}
    
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
                        input_file = line.strip().split("=")[1]
                        self.input_files.append(input_file)
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
        Read data from XML files and store relevant information.
        """
        log.info("Reading documents...")
        try:
            count = 0
            for input_file in self.input_files:
                tree = ET.parse(input_file)
                root = tree.getroot()

                for document in root.findall('RECORD'):
                    count += 1
                    record_num = int(document.find('RECORDNUM').text)
                    if document.find('ABSTRACT') is not None:
                        abstract = document.find('ABSTRACT').text
                    elif document.find('EXTRACT') is not None:
                        abstract = document.find('EXTRACT').text
                    else:
                        abstract = ''
                        log.warning("Document Nº %d found in %s has no Abstract or Extract element.", record_num, input_file)
                    self.documents.append((record_num, abstract))

            log.info("Finished reading documents. Total documents read: %d", count)

        except ET.ParseError as e:
            log.error(f"Failed to parse XML file: {e}")
            raise
        except FileNotFoundError:
            log.error(f"One or more of the designated input files was not found.")
            raise
        except OSError as e:
            log.error(f"Error opening XML file: {e}")
            raise

    def __process_data(self):
        """
        Process data from documents and generate inverted index.
        """
        log.info("Processing documents data...")
        for record, abstract in self.documents:
            processed_abstract = normalize_text(abstract)
            self.__build_inverted_index(record, processed_abstract)
        log.info("Finished processing documents data.")

    def __write_data(self):
        """
        Write inverted index to output file.
        """
        log.info("Writing output files...")
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as output_file:
                writer = csv.writer(output_file, delimiter=';')

                for word, doc_ids in self.inverted_index.items():
                    writer.writerow([word, doc_ids])

            log.info("Finished writing output files.")

        except OSError as e:
            log.error(f"Failed to write output file: {e}")
            raise
    
    def __build_inverted_index(self, record_num, text):
        """
        Build inverted index from document text.
        """
        words = text.split()
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = []
            self.inverted_index[word].append(record_num)
    
    def run(self):
        """
        Main method to run the inverted list generator.
        """
        # Redefinindo ou inicializando os atributos para garantir um estado limpo
        self.input_files = []
        self.documents = []
        self.inverted_index = {}

        log.info("Starting Inverted List Generator...")
        self.__read_configuration()
        self.__read_data()
        self.__process_data()
        self.__write_data()
        log.info("Inverted List Generator completed.")