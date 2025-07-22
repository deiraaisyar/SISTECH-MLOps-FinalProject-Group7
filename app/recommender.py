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


def recommend_careers(r:int, i:int, a:int, s:int, e:int, c:int, top_n:int = 3) -> dict:    
    url = f"https://services.onetcenter.org/ws/mnm/interestprofiler/careers?Realistic={r}&Investigative={i}&Artistic={a}&Social={s}&Enterprising={e}&Conventional={c}"
    
    load_dotenv()
    headers={'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.standard_b64encode((os.getenv('ONET_USERNAME') + ':' + os.getenv('ONET_PASSWORD')).encode()).decode(),
            'Accept': 'application/json' }
    
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"Error fetching data from O*NET: {r.status_code} - {r.text}")
    
    data = r.json()
    
    top_n_careers = []
    
    for career in data['career'][:top_n]:
        report_url = f"https://services.onetcenter.org/ws/mnm/careers/{career['code']}/report"
        r = requests.get(report_url, headers=headers)
        if r.status_code != 200:
            raise Exception(f"Error fetching career report: {r.status_code} - {r.text}")
        print("Successfully fetched career report for:", career['code'])
        report_data = r.json()
        code = report_data.get('career', {}).get('code', '')
        title = report_data.get('career', {}).get('title', '')
        also_called = ", ".join(report_data.get('career', {}).get('also_called', {}).get('title', []))
        what_they_do = report_data.get('career', {}).get('what_they_do', '')
        on_the_job = ", ".join(report_data.get('career', {}).get('on_the_job', {}).get('task', []))

        
        c_knowledges = []
        knowledges = report_data.get('knowledge', {}).get('group', [])

        for knowledge in knowledges:
            c_knowledges.append(knowledge['title']['name'])
            elements = knowledge['element']
            for element in elements:
                c_knowledges.append(element['name'])
        
        c_abilities = []
        abilities = report_data.get('abilities', {}).get('group', [])

        for ability in abilities:
            c_abilities.append(ability['title']['name'])
            elements = ability['element']
            for element in elements:
                c_abilities.append(element['name'])
                
        c_skills = []
        skills = report_data.get('skills', {}).get('group', [])

        for skill in skills:
            c_skills.append(skill['title']['name'])
            elements = skill['element']
            for element in elements:
                c_skills.append(element['name'])
                
        c_technologies = []
        technologies = report_data.get('technology', {}).get('category', [])

        for tech in technologies:
            c_technologies.append(tech['title']['name'])
            examples = tech['example']
            for ex in examples:
                c_technologies.append(ex['name'])

        outlook = report_data.get('job_outlook', {}).get('outlook', '')

        title_preprocessed = preprocessing(title)
        also_called_preprocessed = preprocessing(also_called)
        what_they_do_preprocessed = preprocessing(what_they_do)
        on_the_job_preprocessed = preprocessing(on_the_job)
        # c_knowledges_preprocessed = " ".join([preprocessing(k) for k in c_knowledges])
        # c_skills_preprocessed = " ".join([preprocessing(s) for s in c_skills])
        # c_abilities_preprocessed = " ".join([preprocessing(a) for a in c_abilities])
        preprocessed_text = title_preprocessed + " " + also_called_preprocessed + " " + what_they_do_preprocessed + " " + on_the_job_preprocessed
        
        top_n_careers.append({
            'code': code,
            'title': title,
            'also_called': also_called,
            'what_they_do': what_they_do,
            'on_the_job': on_the_job,
            'knowledges': c_knowledges,
            'skills': c_skills,
            'abilities': c_abilities,
            'outlook': outlook,
            'preprocessed_text': preprocessed_text,
        })
        
    return top_n_careers



def recommend_jobs(career_text:str, model, index, job_data, top_n:int = 3) -> dict:    
    if isinstance(model, SentenceTransformer):
        query_emb = np.array(model.encode([career_text]))
        query_emb = normalize(query_emb)
    else:
        query_emb = model.transform([career_text]).toarray()
    
    D, I = index.search(query_emb, top_n)
    print(I)
    results = []
    
    
    for idx, dist in zip(I[0], D[0]):
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
            "num_applicants": job.get('num_applicants', ''),
            "score": float(dist)
        })
    level = ['Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']
    results_sorted = sorted(results, key=lambda x: (level.index(x['level']) if x['level'] in level else len(level), x['score']))
    return results_sorted



def recommend_courses(career_text:str, model, index, course_data, top_n:int = 3) -> dict:    
    if isinstance(model, SentenceTransformer):
        query_emb = np.array(model.encode([career_text]))
        query_emb = normalize(query_emb)
    else:
        query_emb = model.transform([career_text]).toarray()

    D, I = index.search(query_emb, index.ntotal)
    top_courses = []
    top_programs = []

    for idx, dist in zip(I[0], D[0]):
        course = course_data[idx]
        product_type = course.get('product', '')
        program_type = course.get('program_type', '')

        result = {
            "title": course.get('title', ''),
            "partner": course.get('partner', ''),
            "primary_description": course.get('primary_description', ''),
            "secondary_description": course.get('secondary_description', ''),
            "tertiary_description": course.get('tertiary_description', ''),
            "availability": course.get('availability', ''),
            "subject": course.get('subject', ''),
            "level": course.get('level', ''),
            "language": course.get('language', ''),
            "product": product_type,
            "program_type": program_type,
            "staff": course.get('staff', ''),
            "translation_language": course.get('translation_language', ''),
            "transcription_language": course.get('transcription_language', ''),
            "recent_enrollment_count": int(course['recent_enrollment_count']) if course.get('recent_enrollment_count') not in [None, ''] else '',
            "marketing_url": course.get('marketing_url', ''),
            "weeks_to_complete": int(course['weeks_to_complete']) if course.get('weeks_to_complete') not in [None, ''] else '',
            "skill": course.get('skill', ''),
            "score": float(dist)
        }

        if product_type == "Course" and len(top_courses) < top_n:
            top_courses.append(result)
        elif product_type == "Program" and "Professional Certificate" in program_type and len(top_programs) < top_n:
            top_programs.append(result)

        
        if len(top_courses) == top_n and len(top_programs) == top_n:
            break

    results = {
        "courses": top_courses,
        "certifications": top_programs
    }

    return results

def recommend_programs(query_text: str, model, index, program_data, top_n: int = 3) -> dict:
    if isinstance(model, SentenceTransformer):
        query_emb = np.array(model.encode([query_text]))
        query_emb = normalize(query_emb)
    else:
        query_emb = model.transform([query_text]).toarray()

    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        program = program_data.iloc[idx] if isinstance(program_data, pd.DataFrame) else program_data[idx]
        results.append({
            "university": program['Universitas'] if pd.notna(program['Universitas']) else '',
            "program": program['Prodi'] if pd.notna(program['Prodi']) else '',
            "rank": int(program['Rank']) if not pd.isna(program['Rank']) else '',
            "text": program['text'] if pd.notna(program['text']) else '',
            "score": float(dist)
        })

    # Sort results by rank (ascending order, higher rank is better)
    results = sorted(results, key=lambda x: x['rank'])
    return results


def get_job_trends(query:str, top_n:int = 3) -> list:
    """
    Fetches job trends from Google Custom Search API based on the query.
    
    Args:
        query (str): The search query for job trends.
        top_n (int): Number of top results to return.
        
    Returns:
        list: A list of dictionaries containing job trend information.
    """
    # Load environment variables
    dotenv.load_dotenv()

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