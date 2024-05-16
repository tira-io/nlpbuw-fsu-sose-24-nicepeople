# run.py

from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Define a custom preprocessing function
def preprocess_text(text):
    preprocessed_text = []
    for sentence in text:
        if pd.isna(sentence):
            preprocessed_text.append(np.nan)
            continue
        # Split text and change chars to utf-8 decimal number
        chars = [ord(char) for char in sentence if char.isalpha()]
        preprocessed_text.append(" ".join(map(str, chars)))
    return preprocessed_text

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Load the trained model
    classifier = load(Path(__file__).parent / "model.joblib")

    # Preprocess the text
    df["text"] = preprocess_text(df["text"])

    # Make predictions
    predictions = classifier.predict(df["text"])
    df["lang"] = predictions
    df = df[["id", "lang"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
