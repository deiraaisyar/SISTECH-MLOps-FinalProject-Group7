import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
import translators as ts
import re
import json
import pickle

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

def load_csv(input_path:str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    return pd.read_csv(input_path)

def load_json(input_path:str) -> dict:
    """
    Load data from a JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    
def store_model(model, output_path: str):
    """
    Store the model to a file.
    """
    if isinstance(model, TfidfVectorizer):
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
    elif isinstance(model, SentenceTransformer):
        model.save(output_path)
    else:
        raise ValueError("Unsupported model type. Use 'tfidf' or 'sentence_transformers'.")
    
def load_model(input_path: str):
    """
    Load the model from a file.
    """
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif os.path.isdir(input_path):
        return SentenceTransformer(input_path)
    else:
        raise ValueError("Unsupported model file type. Use '.pkl' for TF-IDF or directory for Sentence Transformers.")

# Update process_data to include translation
def process_data(input_path:str, output_path:str, method:str):
    if(input_path.endswith('.csv')):
        data = load_csv(input_path)
        text_list = data['text'].tolist()
    elif(input_path.endswith('.json')):
        data = load_json(input_path)
        # for entry in data:
        #     entry['text'] = translate_to_english(entry['text'])
        text_list = [entry['text'] for entry in data]
        
    if os.path.exists(output_path):
        index = load_faiss_index(output_path)
        if method == 'tfidf':
            model = load_model(output_path.replace('.index', '.pkl'))
            print(model)
        else:
            model = load_model(output_path.replace('.index', ''))
            print(model)
        return data, model, index

    model, embeddings = generate_embeddings(text_list, method)
    if method == 'tfidf':
        store_model(model, output_path.replace('.index', '.pkl'))
    else:
        store_model(model, output_path.replace('.index', ''))
    print(embeddings.shape)
    if method == 'sentence_transformers':
        embeddings = normalize(embeddings)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, output_path)
    return data, model, index

if __name__ == "__main__":
    process_data("preprocessed/linkedin_jobs.csv", "app/jobs_tfidf.index", "tfidf")
    process_data("preprocessed/edx_courses.csv", "app/courses_tfidf.index", "tfidf")
    process_data("preprocessed/edx_courses.json", "app/courses_tfidf.index", "tfidf")
    process_data("preprocessed/linkedin_jobs.json", "app/jobs_tfidf.index", "tfidf")
    
    print("Data processed and FAISS index created.")
