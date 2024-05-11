from modules.utils import setup_nltk, setup_logging  # Configurações de ambiente
from modules.proc_consultas import QueryProcessor
from modules.lista_invertida import InvertedListGenerator
from modules.indexador import Indexer
from modules.buscador import SearchEngine
from modules.avaliacao import Evaluator
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

def evaluate_results(stemmer=False):
    sufix = "stemmer" if stemmer else "nostemmer"

    Evaluator(results_file=f'./result/resultados_{sufix}.csv',
              expected_file='./result/esperados.csv',
              output_dir='./avalia/',
              identifier=sufix).run()

if __name__ == "__main__":
    # Execução do sistema de recuperação de informações com e sem stemmer
    log.info("Running In Memory Retrieval using PORTER STEMMER...")
    in_memory_retrieval(stemmer=True)

    log.info("Running In Memory Retrieval WITHOUT STEMMER...")
    in_memory_retrieval(stemmer=False)

    # Avaliação dos resultados do sistema de IR
    log.info("Evaluating PORTER STEMMED results...")
    evaluate_results(stemmer=True)

    log.info("Evaluating NOT STEMMED results...")
    evaluate_results(stemmer=False)
