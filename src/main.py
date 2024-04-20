from modules.utils import setup_nltk, setup_logging  # Configurações de ambiente
from modules.proc_consultas import QueryProcessor
from modules.lista_invertida import InvertedListGenerator
from modules.indexador import Indexer
from modules.buscador import SearchEngine

if __name__ == "__main__":
    modules = [
        QueryProcessor(config_file="./config/pc.cfg"),
        InvertedListGenerator(config_file="./config/gli.cfg"),
        Indexer(config_file="./config/index.cfg"),
        SearchEngine(config_file="./config/busca.cfg")
    ]

    for module in modules:
        module.run()