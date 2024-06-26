from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory


import spacy

import pandas as pd



nlp = spacy.load("en_core_web_sm")

if __name__ == "__main__":

    tira = Client()


    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )


    def predict_labels(sentence):
        # Process the sentence with spaCy NER model
        doc = nlp(sentence)
        
        # Initialize empty list for tags
        tags = []
        
        # Iterate over tokens in the processed doc
        for token in doc:
            # Use token.ent_iob_ to get IOB format (B, I, O)
            # Use token.ent_type_ to get the entity type
            if token.ent_iob_ != 'O':  # If the token is part of an entity
                to_append = f"{token.ent_iob_}-{token.ent_type_}"
                splitIOB = to_append.split("-")
                ner_tag = splitIOB[0] + "-" + splitIOB[-1].lower()

                tags.append(ner_tag)
            else:
                tags.append('O')
        
        
        return tags
    



    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(predict_labels)
    predictions = predictions[['id', 'tags']]

    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )