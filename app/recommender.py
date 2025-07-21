import numpy as np
import pandas as pd
import requests
import base64
from dotenv import load_dotenv
import os
from app.text_preprocessing import preprocessing
from app.data_processing import normalize
from sentence_transformers import SentenceTransformer


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
        code = report_data['career']['code']
        title = report_data['career']['title']
        also_called = ", ".join(report_data['career']['also_called']['title'])
        what_they_do = report_data['career']['what_they_do']
        on_the_job = ", ".join(report_data['career']['on_the_job']['task'])
        
        c_knowledges = []
        knowledges = report_data['knowledge']['group']
        
        for knowledge in knowledges:
            c_knowledges.append(knowledge['title']['name'])
            elements = knowledge['element']
            for element in elements:
                c_knowledges.append(element['name'])
        
        c_abilities = []
        abilities = report_data['abilities']['group']
        
        for ability in abilities:
            c_abilities.append(ability['title']['name'])
            elements = ability['element']
            for element in elements:
                c_abilities.append(element['name'])
                
        c_skills = []
        skills = report_data['skills']['group']
        
        for skill in skills:
            c_skills.append(skill['title']['name'])
            elements = skill['element']
            for element in elements:
                c_skills.append(element['name'])
                
        c_technologies = []
        technologies = report_data['technology']['category']
        
        for tech in technologies:
            c_technologies.append(tech['title']['name'])
            examples = tech['example']
            for ex in examples:
                c_technologies.append(ex['name'])
                
        print(career['title'])
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
            'preprocessed_text': preprocessed_text,
        })
        
    return top_n_careers



def recommend_jobs(career_text:str, model, index, job_df, top_n:int = 3) -> dict:    
    if isinstance(model, SentenceTransformer):
        query_emb = np.array(model.encode([career_text]))
        query_emb = normalize(query_emb)
    else:
        query_emb = model.transform([career_text]).toarray()
    
    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        job = job_df.iloc[idx]
        results.append({
           "link": job['job_link'] if pd.notna(job['job_link']) else '',
            "title": job['job_title'] if pd.notna(job['job_title']) else '',
            "company_name": job['company_name'] if pd.notna(job['company_name']) else '',
            "location": job['location'] if pd.notna(job['location']) else '',
            "responsibilities": job['responsibilities'] if pd.notna(job['responsibilities']) else '',
            "requirements": job['requirements'] if pd.notna(job['requirements']) else '',
            "level": job['level'] if pd.notna(job['level']) else '',
            "employment_type": job['employment_type'] if pd.notna(job['employment_type']) else '',
            "function": job['job_function'] if pd.notna(job['job_function']) else '',
            "industries": job['industries'] if pd.notna(job['industries']) else '',
            "time_posted": job['time_posted'] if pd.notna(job['time_posted']) else '',
            "num_applicants": job['num_applicants'] if pd.notna(job['num_applicants']) else '',
            "score": float(dist)
        })
    print("RESULTS:", results)
    return results



def recommend_courses(career_text:str, model, index, course_df, top_n:int = 3) -> dict:    
    if isinstance(model, SentenceTransformer):
        query_emb = np.array(model.encode([career_text]))
        query_emb = normalize(query_emb)
    else:
        query_emb = model.transform([career_text]).toarray()
    
    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        course = course_df.iloc[idx]
        results.append({
            "title": '' if pd.isna(course['title']) else course['title'],
            "partner": '' if pd.isna(course['partner']) else course['partner'],
            "primary_description": '' if pd.isna(course['primary_description']) else course['primary_description'],
            "secondary_description": '' if pd.isna(course['secondary_description']) else course['secondary_description'],
            "tertiary_description": '' if pd.isna(course['tertiary_description']) else course['tertiary_description'],
            "availability": '' if pd.isna(course['availability']) else course['availability'],
            "subject": '' if pd.isna(course['subject']) else course['subject'],
            "level": '' if pd.isna(course['level']) else course['level'],
            "language": '' if pd.isna(course['language']) else course['language'],
            "product": '' if pd.isna(course['product']) else course['product'],
            "program_type": '' if pd.isna(course['program_type']) else course['program_type'],
            "staff": '' if pd.isna(course['staff']) else course['staff'],
            "translation_language": '' if pd.isna(course['translation_language']) else course['translation_language'],
            "transcription_language": '' if pd.isna(course['transcription_language']) else course['transcription_language'],
            "recent_enrollment_count": int(course['recent_enrollment_count']) if not pd.isna(course['recent_enrollment_count']) else "",
            "marketing_url": '' if pd.isna(course['marketing_url']) else course['marketing_url'],
            "weeks_to_complete": int(course['weeks_to_complete']) if not pd.isna(course['weeks_to_complete']) else "",
            "skill": '' if pd.isna(course['skill']) else course['skill'],
            "score": float(dist)
        })
    return results