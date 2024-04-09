from modules.proc_consultas import QueryProcessor
from modules.lista_invertida import InvertedListGenerator
from modules.indexador import Indexer
from modules.utils import setup_nltk, setup_logging

# Preparing environment
setup_nltk
setup_logging

if __name__ == "__main__":
    query_processor = QueryProcessor(config_file="./config/pc.cfg")
    inverted_list_generator = InvertedListGenerator(config_file="./config/gli.cfg")
    indexer = Indexer(config_file="index.cfg")

    # query_processor.run()
    # inverted_list_generator.run()
    indexer.run()