import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def cleaner(raw_text):
    # Asegurar que los recursos necesarios estén disponibles
    try:
        stopwords.words('spanish')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        WordNetLemmatizer()
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Recurso adicional para lematización multilingüe

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Limpieza del texto
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]', '', raw_text)
    # Remove digits
    clean_text = re.sub(r'\d+', '', clean_text)
    # Remove stopwords
    clean_text = ' '.join([word for word in clean_text.split() if word.lower() not in stopwords.words('spanish')])
    # Remove HTML
    clean_text = re.sub(r'<.*?>', '', clean_text)
    # Lemmatize
    clean_text = ' '.join([lemmatizer.lemmatize(word) for word in clean_text.split()])
    # Stemming
    clean_text = ' '.join([stemmer.stem(word) for word in clean_text.split()])

    print(clean_text)
