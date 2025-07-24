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

def load_json(input_path: str) -> pd.DataFrame:
    """
    Load data from a JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_embeddings(texts: list, model:SentenceTransformer) -> list:
    """
    Generate embeddings for a list of texts
    """
    embeddings = model.encode(texts, verbose=True)
    return embeddings

def build_model():
    """
    Build a Sentence Transformer model.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def build_faiss_index(embeddings: list):
    """
    Build FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
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
    if os.path.isdir(output_path):
        model.save(output_path)
    else:
        raise ValueError("Output path must be a directory for storing the model.")
    
def load_model(input_path: str):
    """
    Load the model from a file or directory.
    - For Sentence Transformers, the model should be a directory.
    """
    if os.path.isdir(input_path):
        # Load Sentence Transformers model
        return SentenceTransformer(input_path)
    else:
        raise ValueError("Unsupported model file type. Use a directory for Sentence Transformers.")

def get_model(model_path: str):
    """ 
    Get the model
    If the model does not exist, it will be built and stored.
    """
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        store_model(model, model_path)
        print(f"Model saved to {model_path}")
    return model
        
def process_data(input_path: str, output_path: str, model_path: str = None):
    data = load_json(input_path)
    print(f"Loaded JSON data: {type(data)}")
    text_list = [d['text'] for d in data]
    
    if os.path.exists(output_path):
        index = load_faiss_index(output_path)
        print(f"Index loaded from {output_path}")
    else:
        if model_path is None:
            raise ValueError("Model path must be provided if the index does not exist.")
        model = get_model(model_path)
        embeddings = generate_embeddings(text_list, model)
        embeddings = normalize(embeddings)
        index = build_faiss_index(embeddings)
        save_faiss_index(index, output_path)
        print(f"Index saved to {output_path}")
    
    return data, index

if __name__ == "__main__":
    jobs_data, job_index = process_data(
        input_path="preprocessed/linkedin_jobs.json",
        output_path="app/embeddings/jobs_st.index",
        model_path="app/models/st_model"
    )
    courses_data, course_index = process_data(
        input_path="preprocessed/edx_courses.json",
        output_path="app/embeddings/courses_st.index",
        model_path="app/models/st_model"
    )
    
    career_data, career_index = process_data(
        input_path="preprocessed/onet_careers.json",
        output_path="app/embeddings/careers_st.index",
        model_path="app/models/st_model"
    )
    # programs_df, program_model, program_index = process_data(
    #     input_path="preprocessed/major_final.json",
    #     output_path="app/models/programs_st.index",
    #     method="sentence_transformers",
    #     model_path="app/models/st_model"
    # )
    
    print("Job model and index loaded.")
    print("Course model and index loaded.")
    # print("Program model:", program_model)
    # print("Program index:", program_index)
    # print("Programs data:", programs_df[:5])  # Tampilkan 5 baris pertama untuk debugging
