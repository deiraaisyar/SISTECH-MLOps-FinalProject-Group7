import numpy as np
import pandas as pd
import requests
import base64
from dotenv import load_dotenv
import os
from app.text_preprocessing import preprocessing
from app.data_processing import normalize
from sentence_transformers import SentenceTransformer
import requests
import os
import dotenv


def recommend_careers(r: int, i: int, a: int, s: int, e: int, c: int, top_n: int = 5) -> list:
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
    
    data = r.json()
    careers = data.get('career', [])
    
    results = []
    for career in careers[:top_n]:
        try:
            # Fetch detailed career report
            report_url = f"https://services.onetcenter.org/ws/mnm/careers/{career['code']}/report"
            report_response = requests.get(report_url, headers=headers)
            if report_response.status_code != 200:
                raise Exception(f"Error fetching career report: {report_response.status_code} - {report_response.text}")
            
            report_data = report_response.json()
            code = report_data.get('career', {}).get('code', 'N/A')
            title = report_data.get('career', {}).get('title', 'Unknown Title')
            also_called = ", ".join(report_data.get('career', {}).get('also_called', {}).get('title', []))
            what_they_do = report_data.get('career', {}).get('what_they_do', 'N/A')
            on_the_job = ", ".join(report_data.get('career', {}).get('on_the_job', {}).get('task', []))
            
            # Process knowledges
            c_knowledges = []
            for knowledge in report_data.get('knowledge', {}).get('group', []):
                c_knowledges.append(knowledge['title']['name'])
                for element in knowledge.get('element', []):
                    c_knowledges.append(element['name'])
            
            # Process skills
            c_skills = []
            for skill in report_data.get('skills', {}).get('group', []):
                c_skills.append(skill['title']['name'])
                for element in skill.get('element', []):
                    c_skills.append(element['name'])
            
            # Process abilities
            c_abilities = []
            for ability in report_data.get('abilities', {}).get('group', []):
                c_abilities.append(ability['title']['name'])
                for element in ability.get('element', []):
                    c_abilities.append(element['name'])
            
            # Process technologies
            c_technologies = []
            for tech in report_data.get('technology', {}).get('category', []):
                c_technologies.append(tech['title']['name'])
                for example in tech.get('example', []):
                    c_technologies.append(example['name'])
            
            # Job outlook
            outlook = report_data.get('job_outlook', {}).get('outlook', 'N/A')
            
            # Preprocess text for TF-IDF or embeddings
            title_preprocessed = preprocessing(title)
            also_called_preprocessed = preprocessing(also_called)
            what_they_do_preprocessed = preprocessing(what_they_do)
            on_the_job_preprocessed = preprocessing(on_the_job)
            preprocessed_text = f"{title_preprocessed} {also_called_preprocessed} {what_they_do_preprocessed} {on_the_job_preprocessed}"
            
            # Append to results
            results.append({
                'code': code,
                'title': title,
                'also_called': also_called,
                'what_they_do': what_they_do,
                'on_the_job': on_the_job,
                'knowledges': c_knowledges,
                'skills': c_skills,
                'abilities': c_abilities,
                'technologies': c_technologies,
                'outlook': outlook,
                'preprocessed_text': preprocessed_text
            })
        except Exception as e:
            print(f"Error processing career data for {career.get('code', 'N/A')}: {e}")
            continue
    
    return results

def recommend_jobs(query_text: str, model, job_data, top_n: int = 5) -> list:
    # Validate job_data type
    if isinstance(job_data, pd.DataFrame):
        job_data = job_data.to_dict(orient="records")  # Convert DataFrame to list of dict
    
    print(f"Query text: {query_text}")
    try:
        query_emb = np.array(model.encode([query_text]))
        print(f"Query embedding shape: {query_emb.shape}")
        query_emb = normalize(query_emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

    try:
        corpus = [item['job_title'] for item in job_data]
        corpus_emb = np.array(model.encode(corpus))
        corpus_emb = normalize(corpus_emb)
        print(f"Corpus embedding shape: {corpus_emb.shape}")
    except Exception as e:
        print(f"Error generating corpus embeddings: {e}")
        raise

    try:
        similarities = np.dot(corpus_emb, query_emb.T).flatten()
        print(f"Similarities: {similarities}")
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        raise

    results = []
    for idx, sim in enumerate(similarities):
        try:
            job = job_data[idx]
            results.append({
                "link": job.get('job_link', ''),
                "title": job.get('job_title', ''),
                "company_name": job.get('company_name', ''),
                "location": job.get('location', ''),
                "responsibilities": job.get('responsibilities', ''),
                "requirements": job.get('requirements', ''),
                "level": job.get('level', ''),
                "employment_type": job.get('employment_type', ''),
                "function": job.get('job_function', ''),
                "industries": job.get('industries', ''),
                "time_posted": job.get('time_posted', ''),
                "num_applicants": int(float(job['num_applicants'])) if job.get('num_applicants') not in [None, ''] else '',
                "score": float(sim)
            })
        except Exception as e:
            print(f"Error processing job data at index {idx}: {e}")
            raise

    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    print(f"Sorted results: {sorted_results}")
    return sorted_results[:top_n]


def recommend_courses(query_text: str, model, course_data, top_n: int = 5) -> list:
    # Validate course_data type
    if isinstance(course_data, pd.DataFrame):
        course_data = course_data.to_dict(orient="records")  # Convert DataFrame to list of dict
    
    print(f"Query text: {query_text}")
    try:
        query_emb = np.array(model.encode([query_text]))
        print(f"Query embedding shape: {query_emb.shape}")
        query_emb = normalize(query_emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

    try:
        corpus = [item['title'] for item in course_data]
        corpus_emb = np.array(model.encode(corpus))
        corpus_emb = normalize(corpus_emb)
        print(f"Corpus embedding shape: {corpus_emb.shape}")
    except Exception as e:
        print(f"Error generating corpus embeddings: {e}")
        raise

    try:
        similarities = np.dot(corpus_emb, query_emb.T).flatten()
        print(f"Similarities: {similarities}")
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        raise

    results = []
    for idx, sim in enumerate(similarities):
        try:
            course = course_data[idx]
            results.append({
                "title": course.get('title', ''),
                "partner": course.get('partner', ''),
                "primary_description": course.get('primary_description', ''),
                "secondary_description": course.get('secondary_description', ''),
                "tertiary_description": course.get('tertiary_description', ''),
                "availability": course.get('availability', ''),
                "subject": course.get('subject', ''),
                "level": course.get('level', ''),
                "language": course.get('language', ''),
                "product": course.get('product', ''),
                "program_type": course.get('program_type', ''),
                "staff": course.get('staff', ''),
                "translation_language": course.get('translation_language', ''),
                "transcription_language": course.get('transcription_language', ''),
                "recent_enrollment_count": int(float(course['recent_enrollment_count'])) if course.get('recent_enrollment_count') not in [None, ''] else '',
                "marketing_url": course.get('marketing_url', ''),
                "weeks_to_complete": int(float(course['weeks_to_complete'])) if course.get('weeks_to_complete') not in [None, ''] else '',
                "skill": course.get('skill', ''),
                "score": float(sim)
            })
        except Exception as e:
            print(f"Error processing course data at index {idx}: {e}")
            raise

    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    print(f"Sorted results: {sorted_results}")
    return sorted_results[:top_n]


def recommend_programs(query_text: str, model, program_data, top_n: int = 5) -> list:
    # Validate program_data type
    if isinstance(program_data, pd.DataFrame):
        program_data = program_data.to_dict(orient="records")  # Convert DataFrame to list of dict
    
    print(f"Query text: {query_text}")
    try:
        query_emb = np.array(model.encode([query_text]))
        print(f"Query embedding shape: {query_emb.shape}")
        query_emb = normalize(query_emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

    try:
        corpus = [item['text'] for item in program_data]
        corpus_emb = np.array(model.encode(corpus))
        corpus_emb = normalize(corpus_emb)
        print(f"Corpus embedding shape: {corpus_emb.shape}")
    except Exception as e:
        print(f"Error generating corpus embeddings: {e}")
        raise

    try:
        similarities = np.dot(corpus_emb, query_emb.T).flatten()
        print(f"Similarities: {similarities}")
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        raise

    results = []
    for idx, sim in enumerate(similarities):
        try:
            program = program_data[idx]
            rank = int(float(program.get('Rank', 999)))  # Default rank is 999 if not provided
            composite_score = 0.5 * ((999 - rank) / 999) + 0.5 * sim
            results.append({
                'program': program.get('Prodi', ''),
                'university': program.get('Universitas', ''),
                'rank': rank,
                'similarity': float(sim),
                'composite_score': float(composite_score)
            })
        except Exception as e:
            print(f"Error processing program data at index {idx}: {e}")
            raise

    sorted_results = sorted(results, key=lambda x: x['composite_score'], reverse=True)
    print(f"Sorted results: {sorted_results}")
    return sorted_results[:top_n]


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
