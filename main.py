import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import requests
import base64
import os

# define variables
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# create and save job embeddings
# tfidf_vectorizer_job = TfidfVectorizer()
# job_df = pd.read_csv('job_preprocessed.csv')
# job_embeddings_tfidf = tfidf_vectorizer_job.fit_transform(job_df['text'])
# with open('job_embeddings_tfidf.pkl', 'wb') as f:
#     pickle.dump(job_embeddings_tfidf, f)

# with open('job_vectorizer_tfidf.pkl', 'wb') as f:
#     pickle.dump(tfidf_vectorizer_job, f)

# # create course embeddings
# tfidf_vectorizer_course = TfidfVectorizer()
# course_df = pd.read_csv('edx_courses.csv')
# course_embeddings_tfidf = tfidf_vectorizer_course.fit_transform(course_df['text'])

def lowering(text: str) -> str:
    text = text.lower()
    return text

def remove_punctuation_and_symbol(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text

def stopword_removal(text: str) -> str:
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# use lemmatization instead of stemming for better accuracy and context understanding
def lemmatization(text: str) -> str:
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def preprocessing(text: str) -> str:

    text = lowering(text)
    text = remove_punctuation_and_symbol(text)
    text = stopword_removal(text)
    text = lemmatization(text)

    return text


def recommend_career(r:int, i:int, a:int, s:int, e:int, c:int, top_n:int = 5) -> dict:    
    url = f"https://services.onetcenter.org/ws/mnm/interestprofiler/careers?Realistic={r}&Investigative={i}&Artistic={a}&Social={s}&Enterprising={e}&Conventional={c}"
    
    headers={'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.standard_b64encode(('career_recommendatio6' + ':' + '7627prw').encode()).decode(),
            'Accept': 'application/json' }
    
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise Exception(f"Error fetching data from O*NET: {r.status_code} - {r.text}")
    
    data = r.json()
    
    top_3_careers = []
    
    for career in data['career'][:3]:
        report_url = f"https://services.onetcenter.org/ws/mnm/careers/{career['code']}/report"
        r = requests.get(report_url, headers=headers)
        if r.status_code != 200:
            raise Exception(f"Error fetching career report: {r.status_code} - {r.text}")
        print("Successfully fetched career report for:", career['code'])
        report_data = r.json()
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
        
        title_preprocessed = preprocessing(title)
        also_called_preprocessed = preprocessing(also_called)
        what_they_do_preprocessed = preprocessing(what_they_do)
        on_the_job_preprocessed = preprocessing(on_the_job)
        c_knowledges_preprocessed = " ".join([preprocessing(k) for k in c_knowledges])
        c_skills_preprocessed = " ".join([preprocessing(s) for s in c_skills])
        c_abilities_preprocessed = " ".join([preprocessing(a) for a in c_abilities])
        c_technologies_preprocessed = " ".join([preprocessing(t) for t in c_technologies])
        preprocessed_text = title_preprocessed + " " + also_called_preprocessed + " " + what_they_do_preprocessed + " " + on_the_job_preprocessed + " " + c_knowledges_preprocessed + " " + c_skills_preprocessed + " " + c_abilities_preprocessed + " " + c_technologies_preprocessed
        
        top_3_careers.append({
            'title': title,
            'also_called': also_called,
            'what_they_do': what_they_do,
            'on_the_job': on_the_job,
            'knowledges': c_knowledges,
            'skills': c_skills,
            'abilities': c_abilities,
            'technologies': c_technologies,
            'preprocessed_text': preprocessed_text
        })
    
    return top_3_careers    



def recommend_job(text:str, top_n:int = 3) -> dict:
    job_df = pd.read_csv('preprocessed/linkedin_jobs.csv')
    job_vectorizer = TfidfVectorizer()
    job_vectors = job_vectorizer.fit_transform(job_df['text'].astype(str))
    text_vector = job_vectorizer.transform([text])
    dists = 1-cosine_similarity(text_vector, job_vectors).flatten()
    top_n_jobs = job_df.iloc[np.argsort(dists)[:top_n], :].to_dict(orient='records')
    return top_n_jobs

def recommend_courses(text:str, top_n:int = 3) -> dict:
    course_df = pd.read_csv('preprocessed/edx_courses.csv')
    course_vectorizer = TfidfVectorizer()
    course_vectors = course_vectorizer.fit_transform(course_df['text'].astype(str))
    text_vector = course_vectorizer.transform([text])
    dists = 1-cosine_similarity(text_vector, course_vectors).flatten()
    top_n_courses = course_df.iloc[np.argsort(dists)[:top_n], :].to_dict(orient='records')
    return top_n_courses
    
res = recommend_career(0, 5, 0, 0, 5, 5)
print(res)
jobs = recommend_job(res[0]['preprocessed_text'])
print(jobs)
courses = recommend_courses(jobs[0]['text'])
print(courses)