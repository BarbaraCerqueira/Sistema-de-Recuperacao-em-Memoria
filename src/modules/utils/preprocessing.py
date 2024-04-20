import re
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def normalize_text(text):
    """
    Preprocess text to remove accents, stopwords, non-alphabetic characters, apply stemming and convert it to uppercase.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Remover acentuação
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Converter para maiúsculas
    text = text.upper()
    # Remover caracteres especiais e números
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    # Remover palavras com apenas 1 letra
    text = re.sub(r'\b\w{1}\b', '', text)
    # Remover stopwords e aplicar stemming
    words = re.findall(r'\b\w+\b', text)
    filtered_stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_stemmed_words)