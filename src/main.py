from modules.utils import setup_nltk, setup_logging  # Configurações de ambiente
from modules.proc_consultas import QueryProcessor
from modules.lista_invertida import InvertedListGenerator
from modules.indexador import Indexer
from modules.buscador import SearchEngine
import logging as log

def in_memory_retrieval(stemmer=False):
    sufix = "stemmer" if stemmer else "nostemmer"

    modules = [
        QueryProcessor(config_file=f"./config/pc_{sufix}.cfg"),
        InvertedListGenerator(config_file=f"./config/gli_{sufix}.cfg"),
        Indexer(config_file=f"./config/index_{sufix}.cfg"),
        SearchEngine(config_file=f"./config/busca_{sufix}.cfg")
    ]

    for module in modules:
        module.run()

if __name__ == "__main__":
    log.info("Running In Memory Retrieval using PORTER STEMMER...")
    in_memory_retrieval(stemmer=True)

    log.info("Running In Memory Retrieval using NO STEMMER...")
    in_memory_retrieval(stemmer=False)