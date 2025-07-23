import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os
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

def load_json(input_path: str) -> pd.DataFrame:
    """
    Load data from a JSON file and convert it to a DataFrame.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert list of dictionaries to DataFrame
    if isinstance(data, list):
        return pd.DataFrame(data)
    else:
        return pd.DataFrame([data])

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
    Load the model from a file or directory.
    - For Sentence Transformers, the model should be a directory.
    """
    if not input_path:
        raise ValueError("Model path cannot be None. Please provide a valid path.")
    
    if os.path.isdir(input_path):
        # Load Sentence Transformers model
        return SentenceTransformer(input_path)
    else:
        raise ValueError("Unsupported model file type. Use a directory for Sentence Transformers.")

def process_data(input_path: str, output_path: str, method: str, model_path: str = None):
    if input_path.endswith('.csv'):
        data = load_csv(input_path)
        text_list = data['text'].tolist()
    elif input_path.endswith('.json'):
        data = load_json(input_path)
        print(f"Loaded JSON data: {type(data)}")
        text_list = data['text'].tolist()
        
    if os.path.exists(output_path):
        index = load_faiss_index(output_path)
        model = load_model(model_path)
        print("Model and index loaded from disk.")
        return data, model, index
    
    # Use Sentence Transformers for embedding
    if method == "sentence_transformers":
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Ganti dengan model yang Anda gunakan
        print(f"Using Sentence Transformers model: {model}")
        embeddings = model.encode(text_list, show_progress_bar=True)
        embeddings = normalize(embeddings)
        if model_path:
            model.save(model_path)  # Simpan model ke direktori
    else:
        raise ValueError("Unsupported method. Use 'sentence_transformers'.")

    index = build_faiss_index(embeddings)
    save_faiss_index(index, output_path)
    print(f"Processed data saved to {output_path}")
    return data, model, index

if __name__ == "__main__":
    jobs_df, job_model, job_index = process_data(
        input_path="preprocessed/linkedin_jobs.json",
        output_path="app/models/jobs_st.index",
        method="sentence_transformers",
        model_path="app/models/st_model"
    )
    courses_df, course_model, course_index = process_data(
        input_path="preprocessed/edx_courses.json",
        output_path="app/models/courses_st.index",
        method="sentence_transformers",
        model_path="app/models/st_model"
    )
    programs_df, program_model, program_index = process_data(
        input_path="preprocessed/major_final.json",
        output_path="app/models/programs_st.index",
        method="sentence_transformers",
        model_path="app/models/st_model"
    )
    
    print("Job model and index loaded.")
    print("Course model and index loaded.")
    print("Program model:", program_model)
    print("Program index:", program_index)
    print("Programs data:", programs_df[:5])  # Tampilkan 5 baris pertama untuk debugging
