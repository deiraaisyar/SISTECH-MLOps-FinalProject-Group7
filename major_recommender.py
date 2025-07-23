import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_model(model_path):
    return SentenceTransformer(model_path)

def recommend_programs(query_text, model, program_data, top_n=5):
    query_emb = np.array(model.encode([query_text]))
    query_emb = normalize(query_emb)

    corpus = [item['text'] for item in program_data]
    corpus_emb = np.array(model.encode(corpus))
    corpus_emb = normalize(corpus_emb)

    similarities = np.dot(corpus_emb, query_emb.T).flatten()

    results = []
    for idx, sim in enumerate(similarities):
        program = program_data[idx]
        rank = int(float(program.get('Rank', 999)))

        rank_score = (999 - rank) / 999 
        composite_score = 0.5 * rank_score + 0.5 * sim

        results.append({
            'program': program.get('Prodi', ''),
            'university': program.get('Universitas', ''),
            'rank': rank,
            'similarity': float(sim),
            'composite_score': float(composite_score)
        })

    sorted_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
    return sorted_results[:top_n]

if __name__ == '__main__':
    model_path = 'app/models/st_model'  
    data_path = 'preprocessed/major_final.json' 

    program_data = load_data(data_path)
    model = load_model(model_path)

    query = "medicine"
    recommendations = recommend_programs(query, model, program_data, top_n=5)

    for r in recommendations:
        print(f"{r['program']} - {r['university']} (Rank: {r['rank']}, Score: {r['composite_score']:.4f})")
