import re
from unicodedata import normalize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def normalize_text(text):
    """
    Preprocess text to remove accents, stopwords, non-alphabetic characters, and convert it to uppercase.
    """
    # Remover acentuação
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Converter para maiúsculas
    text = text.upper()
    # Remover caracteres especiais e números
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    # Remover stopwords
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)