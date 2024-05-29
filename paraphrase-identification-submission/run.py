import pandas as pd
from tira.rest_api_client import Client
import joblib
from pathlib import Path
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")
    
    # Load the TF-IDF vectorizer and model
    vectorizer = joblib.load(Path(__file__).parent / "vectorizer.joblib")
    model = joblib.load(Path(__file__).parent / "model.joblib")
    
    # Prepare the data
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    
    # Vectorize the text
    X1 = vectorizer.transform(sentences1)
    X2 = vectorizer.transform(sentences2)
    
    # Ensure that the dimensions of the concatenated TF-IDF vectors match the expected input dimensions of the model
    # Here, we use scipy.sparse.vstack instead of scipy.sparse.hstack
    X_ = scipy.sparse.hstack([X1, X2])  # Use vstack to stack vertically

    # Calculate cosine similarity
    cosine_sim = np.array([cosine_similarity(X1[i], X2[i])[0][0] for i in range(X1.shape[0])]).reshape(-1, 1)
    
    # Combine all features into a single matrix
    X = scipy.sparse.hstack([X_, scipy.sparse.csr_matrix(cosine_sim)])
    
    # Make predictions
    predictions = model.predict(X)
    
    # Prepare output
    output = pd.DataFrame({'id': df.index, 'label': predictions})
    output.to_json(Path(__file__).parent / "predictions.jsonl", orient="records", lines=True)
