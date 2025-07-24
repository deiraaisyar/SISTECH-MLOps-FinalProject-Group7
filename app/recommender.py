import numpy as np
import requests
import base64
from dotenv import load_dotenv
import os
from app.text_preprocessing import preprocessing
from app.data_processing import normalize
import app.data_processing as dp
import requests
import os
import dotenv
from typing import Optional


def recommend_careers(data, 
                      index, 
                      model,
                      r: int, i: int, a: int, s: int, e: int, c: int, 
                      in_highschool: bool, 
                      skills: str, 
                      interests: str, 
                      top_n: int = 5, 
                      major: Optional[str] = None) -> list:
    
    
    if not in_highschool and major is None:
        raise ValueError("Parameter 'major' is required when 'in_highschool' is True")
    
    url = f"https://services.onetcenter.org/ws/mnm/interestprofiler/careers?Realistic={r}&Investigative={i}&Artistic={a}&Social={s}&Enterprising={e}&Conventional={c}"
    load_dotenv()
    headers = {
        'User-Agent': 'python-OnetWebService/1.00 (bot)',
        'Authorization': 'Basic ' + base64.standard_b64encode((os.getenv('ONET_USERNAME') + ':' + os.getenv('ONET_PASSWORD')).encode()).decode(),
        'Accept': 'application/json'
    }
    
    # Fetch career data
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"Error fetching data from O*NET: {r.status_code} - {r.text}")
    
    onet_result = r.json()
    careers = onet_result['career']
    top_career_codes = [c['code'] for c in careers[:top_n*3]]
    filtered_career_idx = [idx for idx, career in enumerate(data) if career['code'] in top_career_codes]
    
    query_text = f"{major if major else ''} {skills} {interests}"
    query_text = preprocessing(query_text)
    query_emb = model.encode([query_text])
    query_emb = normalize(query_emb)
    
    D, I = index.search(query_emb, index.ntotal)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx in filtered_career_idx:
            d = data[idx]
            results.append({
                'code': d['code'],
                'title': d['title'],
                'also_called': d['also_called'],
                'what_they_do': d['what_they_do'],
                'on_the_job': d['on_the_job'],
                'knowledges': d['knowledges'],
                'skills': d['skills'],
                'abilities': d['abilities'],
                'technologies': d['technologies'],
                'job_outlook': d['job_outlook'],
                'text': d['text'] if in_highschool else f"{d['text']} {query_text}",
                'score': float(dist)
            })
            if len(results) == top_n:
                break
    
    return results

def recommend_jobs(query_text: str, model, data, index, top_n: int = 5) -> list:
    query_emb = np.array(model.encode([query_text]))
    query_emb = normalize(query_emb)

    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        d = data[idx]
        results.append({
            'link': d['job_link'],
            'title': d['job_title'],
            'company_name': d['company_name'],
            'location': d['location'],
            'responsibilities': d['responsibilities'],
            'requirements': d['requirements'],
            'level': d['level'],
            'employment_type': d['employment_type'],
            'job_function': d['job_function'],
            'industries': d['industries'],
            'time_posted': d['time_posted'],
            'num_applicants': d['num_applicants'],
            'score': float(dist)
        })
    
    
    # sort based on level of job
    level = ['internship', 'entry level', 'associate', 'mid-senior level', 'director', 'executive']
    results = sorted(results, key=lambda x: (level.index(x['level'].lower()) if x['level'].lower() in level else len(level), x['score']))
    return results


def recommend_courses(query_text: str, model, data, index, top_n: int = 5) -> list:
    query_emb = np.array(model.encode([query_text]))
    query_emb = normalize(query_emb)

    D, I = index.search(query_emb, index.ntotal)
    courses = []
    certifications = []
    for idx, dist in zip(I[0], D[0]):
        d = data[idx]
        result = {
            'marketing_url': d['marketing_url'],
            "title": d['title'],
            'partner': d['partner'],
            'primary_description': d['primary_description'],
            'secondary_description': d['secondary_description'],
            'tertiary_description': d['tertiary_description'],
            'availability': d['availability'],
            'subject': d['subject'],
            'level': d['level'],
            'language': d['language'],
            'product': d['product'],
            'program_type': d['program_type'],
            'staff': d['staff'],
            'translation_language': d['translation_language'],
            'transcription_language': d['transcription_language'],
            'recent_enrollment_count': d['recent_enrollment_count'],
            'weeks_to_complete': d['weeks_to_complete'],
            'skill': d['skill'],
            'score': float(dist)
        }
        
        # to get top_n courses & top_n certifications
        if d['product'].lower() == 'course' and len(courses) < top_n:
            courses.append(result)
        elif d['product'].lower() == 'program' and len(certifications) < top_n:
            certifications.append(result)
        if len(courses) == top_n and len(certifications) == top_n:
            break
        
    return {'courses':courses, 'certifications':certifications}

def recommend_programs(query_text: str, model, data, index, top_n: int = 5) -> list:
    query_emb = np.array(model.encode([query_text]))
    query_emb = normalize(query_emb)

    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        d = data[idx]
        results.append({
            'university': d['Universitas'],
            'program': d['Prodi'],
            'rank': int(d['Rank']),
            'score':float(dist)
        })
     
    results = sorted(results, key=lambda x: x['rank'])
    return results


def get_job_articles(query:str, top_n:int = 3) -> list:
    """
    Fetches job articles from Google Custom Search API based on the query.

    Args:
        query (str): The search query for job trends.
        top_n (int): Number of top results to return.
        
    Returns:
        list: A list of dictionaries containing job trend information.
    """
    # Load environment variables
    dotenv.load_dotenv()
    
    query = query + " career outlook OR employment trends OR job market OR future demand OR career forecast"

    api_key = os.getenv('GOOGLE_API_KEY')
    search_id = os.getenv('SEARCH_ENGINE_ID')
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_id}&q={query}"

    result = requests.get(url)
    if result.status_code != 200:
        raise Exception(f"Error fetching data from Google Custom Search API: {result.status_code} - {result.text}")

    result = result.json()
    
    search_res = []
    for item in result.get('items', [])[:top_n]:
        title = item.get('title')
        link = item.get('link')
        snippet = item.get('snippet')
        search_res.append({
            'title': title,
            'link': link,
            'snippet': snippet
        })
    
    return search_res
