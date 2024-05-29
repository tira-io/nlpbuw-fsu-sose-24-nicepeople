import pandas as pd
from tira.rest_api_client import Client
import joblib
from pathlib import Path
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tira.third_party_integrations import get_output_directory

def main():
    try:
        tira = Client()
        df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")
        
        vectorizer_path = Path(__file__).parent / "vectorizer.joblib"
        model_path = Path(__file__).parent / "model.joblib"
        
        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError("Vectorizer or model file not found.")
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        
        sentences1 = df["sentence1"].tolist()
        sentences2 = df["sentence2"].tolist()
        
        X1 = vectorizer.transform(sentences1)
        X2 = vectorizer.transform(sentences2)
        
        X_ = scipy.sparse.hstack([X1, X2])  
        
        cosine_sim = np.array([cosine_similarity(X1[i], X2[i])[0][0] for i in range(X1.shape[0])]).reshape(-1, 1)
        
        X = scipy.sparse.hstack([X_, scipy.sparse.csr_matrix(cosine_sim)])

        predictions = model.predict(X)
        
        df["label"] = predictions
        df = df.reset_index()[["id", "label"]]  
        
        output_directory = get_output_directory(str(Path(__file__).parent))
        output_file = Path(output_directory) / "predictions.jsonl"
        df.to_json(output_file, orient="records", lines=True)
        
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
