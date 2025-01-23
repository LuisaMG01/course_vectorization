import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def cleaner(raw_text):
    try:
        stopwords.words('spanish')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        WordNetLemmatizer()
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4') 

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_text = re.sub(r'[^\w\s]', '', raw_text)
    clean_text = re.sub(r'\d+', '', clean_text)
    clean_text = ' '.join([word for word in clean_text.split() if word.lower() not in stopwords.words('spanish')])
    clean_text = re.sub(r'<.*?>', '', clean_text)
    clean_text = ' '.join([lemmatizer.lemmatize(word) for word in clean_text.split()])
    clean_text = ' '.join([stemmer.stem(word) for word in clean_text.split()])

    print(clean_text)
