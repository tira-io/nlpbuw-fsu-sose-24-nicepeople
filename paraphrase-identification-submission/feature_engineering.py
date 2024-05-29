import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_features(sentences1, sentences2):
    # Initialize TF-IDF Vectorizer for N-gram features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Use 1-grams, 2-grams, and 3-grams
    tfidf_matrix1 = vectorizer.fit_transform(sentences1)
    tfidf_matrix2 = vectorizer.transform(sentences2)
    
    # Combine TF-IDF features
    tfidf_features = scipy.sparse.hstack([tfidf_matrix1, tfidf_matrix2])
    
    # Calculate cosine similarity
    cosine_sim = np.array([cosine_similarity(tfidf_matrix1[i], tfidf_matrix2[i])[0][0] for i in range(tfidf_matrix1.shape[0])]).reshape(-1, 1)
    
    # Combine all features into a single matrix
    features = scipy.sparse.hstack([tfidf_features, scipy.sparse.csr_matrix(cosine_sim)])
    
    return features, vectorizer