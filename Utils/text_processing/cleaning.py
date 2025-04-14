import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')  
 
def StopWords(text):    
    stop_word = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    clean_text = [token for token in tokens if token.isalnum() and token not in stop_word]
    return ' '.join(clean_text)