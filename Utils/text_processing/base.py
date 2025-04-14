import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def LowerCasing(text):
        return text.lower()

def Tokenization(text):
    return word_tokenize(text)