import pandas as pd
from tira.rest_api_client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    df = text.join(labels)
    
    # Prepare the data
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    
    # Define the function for creating features
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
    
    # Create features using the defined function
    X, vectorizer = create_features(sentences1, sentences2)
    y = df["label"]
    
    # Train the logistic regression model on the full dataset
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save the model and vectorizer
    joblib.dump(model, Path(__file__).parent / "model.joblib")
    joblib.dump(vectorizer, Path(__file__).parent / "vectorizer.joblib")
