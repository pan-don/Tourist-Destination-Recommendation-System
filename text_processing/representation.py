from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

def Tfidf(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text)
    return tfidf_matrix, vectorizer

def WordEmbeddings(text, vector_size=100, window=5, min_count=1):
    tokens = word_tokenize(text)
    model = Word2Vec(sentences=tokens, vector_size=vector_size, window=window, min_count=min_count)
    return model

def CosineSimilarity(matrix):
    cosim = cosine_similarity(matrix)
    return cosim