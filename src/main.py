from modules.proc_consultas import QueryProcessor
from utils import setup_nltk, setup_logging

# Setting environment
setup_nltk
setup_logging

if __name__ == "__main__":
    query_processor = QueryProcessor(config_file="./config/pc.cfg")
    query_processor.run()