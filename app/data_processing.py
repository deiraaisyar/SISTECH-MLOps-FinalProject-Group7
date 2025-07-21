import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
import translators as ts
import re

def normalize(vectors):
    """
    Normalize the vectors to unit length.
    This is important for cosine similarity calculations.
    Input:
        vectors: numpy array of shape (n_samples, n_features)
    Output:
        normalized vectors of the same shape
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def load_data(input_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(input_path)

def generate_embeddings(texts: list, method: str) -> list:
    """
    Generate embeddings for a list of texts using the specified method.
    """
    if method == 'tfidf':
        model = TfidfVectorizer()
        embeddings = model.fit_transform(texts)
    elif method == 'sentence_transformers':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, verbose=True)
    else:
        raise ValueError("Unsupported method. Use 'tfidf' or 'sentence_transformers'.")
    
    return model, embeddings

def build_faiss_index(embeddings: list):
    """
    Build FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.toarray()
    index.add(embeddings)
    return index
    
def save_faiss_index(index, output_path: str):
    """
    Save the FAISS index to a file.
    """
    faiss.write_index(index, output_path)
    
def load_faiss_index(input_path: str):
    """
    Load the FAISS index from a file.
    """
    return faiss.read_index(input_path)


def translate_to_english(text: str) -> str:
    print(f"Input text: {text}")
    try:
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?]) +', text)
        translated_sentences = []
        for sentence in sentences:
            if sentence.strip():  # Ensure the sentence is not empty
                translation = ts.google(sentence, from_language='id', to_language='en')
                if translation is None:
                    print(f"Translation returned None for sentence: {sentence}. Falling back to original.")
                    translation = sentence
                translated_sentences.append(translation)
        # Join translated sentences back into a single text
        translated_text = ' '.join(translated_sentences)
        print(f"Output translation: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Update process_data to include translation
def process_data(input_path:str, output_path:str, method:str):
    df = load_data(input_path)
    
    df['text'] = df['text'].apply(translate_to_english)
        
    model, embeddings = generate_embeddings(df['text'].tolist(), method)
    print(embeddings.shape)
    if method == 'sentence_transformers':
        embeddings = normalize(embeddings)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, output_path)
    return df, model, index

if __name__ == "__main__":
    process_data("preprocessed/linkedin_jobs.csv", "app/jobs_tfidf.index", "tfidf")
    process_data("preprocessed/edx_courses.csv", "app/courses_tfidf.index", "tfidf")
    print("Data processed and FAISS index created.")
