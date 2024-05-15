from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Update the model pipeline to include feature scaling
model = make_pipeline(
    CountVectorizer(),
    StandardScaler(with_mean=False),  # Scale sparse matrix without centering
    LogisticRegression(max_iter=2000)
)

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
    tira = Client()
    # loading training data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    text_train = text_train.set_index("id")
    labels_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-train-20240429-training"
    )
    df_train = text_train.join(labels_train.set_index("id"))
    
    # Preprocess the text
    df_train["text"] = preprocess_text(df_train["text"])
    
    model.fit(df_train["text"], df_train["lang"])
    
    # saving the model
    model_path = Path(__file__).parent / "model.joblib"
    dump(model, model_path)

    print("Implementation successful")

