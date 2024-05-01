from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    # Train the model
    model = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", SVC(kernel='linear'))]
    )  # Use linear kernel for SVC
    model.fit(df["text"], df["generated"])

    # Save the model
    model_path = Path(__file__).parent / "model.joblib"
    dump(model, model_path)
