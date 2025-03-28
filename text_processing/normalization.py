import nltk
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
nltk.download('wordnet')

def LowerCasing(text):
    return text.lower()

def Stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = word_tokenize(text)
    stemming = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemming)
    
def Lemmatization(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)