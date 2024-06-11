from rouge_score import rouge_scorer
from pathlib import Path
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # Initialize TIRA client
    tira = Client()

    # Load predictions and targets
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction = pd.read_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training")

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for target_summary, predicted_summary in zip(targets_validation["summary"], prediction["summary"]):
        scores = scorer.score(target_summary, predicted_summary)
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)

    # Calculate average ROUGE scores
    avg_rouge_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    # Print ROUGE scores
    print("ROUGE Scores:")
    print("-------------")
    for metric, score in avg_rouge_scores.items():
        print(f"{metric.upper()}: {score}")