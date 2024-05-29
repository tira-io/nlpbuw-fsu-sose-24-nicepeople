import pandas as pd
from tira.rest_api_client import Client
import joblib
from pathlib import Path
from feature_engineering import create_features

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-test-20240515-test").set_index("id")
    
    # Prepare the data
    sentences1 = text["sentence1"].tolist()
    sentences2 = text["sentence2"].tolist()
    
    # Load the models and vectorizer
    model = joblib.load(Path(__file__).parent / "logistic_regression_model.joblib")
    vectorizer = joblib.load(Path(__file__).parent / "tfidf_vectorizer.joblib")
    
    # Create features using the updated function
    X, _ = create_features(sentences1, sentences2)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Prepare output
    output = pd.DataFrame({'id': text.index, 'label': predictions})
    output.to_json(Path(__file__).parent / "predictions.jsonl", orient="records", lines=True)
