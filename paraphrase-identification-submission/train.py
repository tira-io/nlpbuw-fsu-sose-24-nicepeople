import pandas as pd
from tira.rest_api_client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import joblib
from pathlib import Path
from feature_engineering import create_features

if __name__ == "_main_":
    
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")

    # Prepare the data
    df = text.join(labels)
    
    # Combine sentence1 and sentence2 into one feature set
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    
    # Create features using the updated function
    X, vectorizer = create_features(sentences1, sentences2)
    y = df["label"]
    
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    mcc = matthews_corrcoef(y_val, y_pred)
    print(f"Validation MCC: {mcc}")
    
    # Save the model and vectorizer
    joblib.dump(model, Path(__file__).parent / "logistic_regression_model.joblib")
    joblib.dump(vectorizer, Path(__file__).parent / "tfidf_vectorizer.joblib")
