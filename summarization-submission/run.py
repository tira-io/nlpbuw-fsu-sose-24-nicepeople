import numpy as np
import networkx as nx
from nltk.corpus import stopwords
import string
import re
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Pre-download NLTK data to avoid runtime issues
nltk.data.path.append('/usr/local/share/nltk_data')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenize into words
    words = word_tokenize(text)
    
    # Remove stop words and lemmatize and stem
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]
    
    # Remove extra whitespace
    text = ' '.join(words)
    
    return text

def textrank_summarizer(text, top_n=3):
    sentences = sent_tokenize(text)
    cleaned_sentences = [preprocess(sentence) for sentence in sentences]
    
    # Filter out empty cleaned sentences
    non_empty_sentences = [sentence for sentence in cleaned_sentences if sentence.strip() != '']
    
    if len(non_empty_sentences) == 0:
        return "No valid content to summarize."
    
    # Build similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    tfidf_vectorizer = TfidfVectorizer().fit_transform(non_empty_sentences)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j and cleaned_sentences[i] and cleaned_sentences[j]:
                similarity_matrix[i][j] = cosine_similarity(tfidf_vectorizer[non_empty_sentences.index(cleaned_sentences[i])], 
                                                            tfidf_vectorizer[non_empty_sentences.index(cleaned_sentences[j])])
    
    # Create graph and calculate PageRank
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    
    # Rank sentences by their scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select top_n sentences for the summary
    summary = ' '.join([sentence for score, sentence in ranked_sentences[:top_n]])
    return summary

def cosine_similarity(vec1, vec2):
    return np.dot(vec1.toarray(), vec2.toarray().T)[0, 0]

def process_row(row, text_column):
    summary = textrank_summarizer(row[text_column])
    return {'id': row.name, 'summary': summary}

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    text_column = 'story' if 'story' in df.columns else df.columns[0]

    # Preprocess the documents
    df['cleaned_text'] = df[text_column].apply(preprocess)

    # Parallel processing 
    with Pool() as pool:
        rows = pool.starmap(
            process_row,
            [(row, text_column) for _, row in df.iterrows()]
        )

    # Convert processed rows back to DataFrame
    df_processed = pd.DataFrame(rows)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df_processed.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
