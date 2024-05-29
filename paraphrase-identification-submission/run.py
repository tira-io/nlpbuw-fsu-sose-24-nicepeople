import pandas as pd
from tira.rest_api_client import Client
import joblib
from pathlib import Path
from feature_engineering import create_features

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")
    
    # Load the trained model and vectorizer
    model = joblib.load(Path(__file__).parent / "logistic_regression_model.joblib")
    vectorizer = joblib.load(Path(__file__).parent / "tfidf_vectorizer.joblib")
    
    # Prepare the data
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    
    # Create features using the same vectorizer used during training
    X, _ = create_features(sentences1, sentences2)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Prepare output
    output = pd.DataFrame({'id': df.index, 'label': predictions})
    output.to_json(Path(__file__).parent / "predictions.jsonl", orient="records", lines=True)
