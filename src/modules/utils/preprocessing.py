import re
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def normalize_text(text, stemmer=False):
    """
    Preprocess text to remove accents, stopwords, non-alphabetic characters, apply stemming (optional) and convert it to uppercase.
    """
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()

    # Remover acentuação
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Converter para maiúsculas
    text = text.upper()
    # Remover caracteres especiais e números
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    # Remover palavras com apenas 1 letra
    text = re.sub(r'\b\w{1}\b', '', text)
    # Remover stopwords
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Aplicar stemming opcional
    if stemmer:
        final_words = [porter_stemmer.stem(word) for word in filtered_words]
    else:
        final_words = filtered_words
    return ' '.join(final_words)