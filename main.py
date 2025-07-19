import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pickle
import requests
import base64
import os
import json

# define variables


# major_df = pd.read_csv('preprocessed/majors.csv')

job_p_df = pd.read_csv('preprocessed/linkedin_jobs.csv')
job_df = pd.read_csv('scrape_result/linkedin_jobs.csv')

course_p_df = pd.read_csv('preprocessed/edx_courses.csv')
course_df = pd.read_csv('scrape_result/edx_courses.csv')

# uncomment if you want to have jobs and courses on the same vector space
# if not os.path.exists('pkl/tfidf_vectorizer.pkl'):
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_vectorizer.fit(pd.concat([job_p_df['text'].astype(str), course_p_df['text'].astype(str)], ignore_index=True))
#     with open('pkl/tfidf_vectorizer.pkl', 'wb') as f:
#         pickle.dump(tfidf_vectorizer, f)
# else:
#     with open('pkl/tfidf_vectorizer.pkl', 'rb') as f:
#         tfidf_vectorizer = pickle.load(f)

# ST Method
if not os.path.exists('pkl/st_model.pkl'):
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    with open('pkl/st_model.pkl', 'wb') as f:
        pickle.dump(st_model, f)
else:
    with open('pkl/st_model.pkl', 'rb') as f:
        st_model = pickle.load(f)


def lowering(text: str) -> str:
    text = text.lower()
    return text

def remove_punctuation_and_symbol(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text

def stopword_removal(text: str) -> str:
    stop_words = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# use lemmatization instead of stemming for better accuracy and context understanding
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def preprocessing(text: str) -> str:

    text = lowering(text)
    text = remove_punctuation_and_symbol(text)
    text = stopword_removal(text)
    text = lemmatization(text)

    return text

def recommend_job(text:str, top_n:int = 3) -> dict:
    
    # TF-IDF Method
    
    # if not os.path.exists('pkl/job_vectorizer.pkl'):
    #     job_vectorizer = TfidfVectorizer()
    #     job_vectorizer.fit(job_p_df['text'].astype(str), ignore_index=True)
    #     with open('pkl/job_vectorizer.pkl', 'wb') as f:
    #             pickle.dump(job_vectorizer, f)
    # else:
    #     with open('pkl/job_vectorizer.pkl', 'rb') as f:
    #         job_vectorizer = pickle.load(f)
    # # job_vectors = tfidf_vectorizer.transform(job_p_df['text'].astype(str))
    # # text_vector = tfidf_vectorizer.transform([text])
    # job_vectors = job_vectorizer.transform(job_p_df['text'].astype(str))
    # text_vector = job_vectorizer.transform([text])
    
    # dists = 1-cosine_similarity(text_vector, job_vectors).flatten()
    
    
    # ST Method
    if not os.path.exists('emb/st_jobs_embeddings.pkl'):
        st_jobs_embeddings = st_model.encode(job_p_df['text'].astype(str))
        with open('emb/st_jobs_embeddings.pkl', 'wb') as f:
            pickle.dump(st_jobs_embeddings, f)
    else:
        with open('emb/st_jobs_embeddings.pkl', 'rb') as f:
            st_jobs_embeddings = pickle.load(f)
    
    text_vector_st = st_model.encode(text)

    dists = 1-cosine_similarity([text_vector_st], st_jobs_embeddings).flatten()
    
    job_df_sorted = job_df.iloc[np.argsort(dists)[:5], :]
    top_n_jobs = []
    level = ['Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive']

    for lvl in level:
        job = job_df_sorted[job_df_sorted['level'] == lvl].head(1).to_dict(orient='records')
        if job:
            top_n_jobs.append(job)
        if len(top_n_jobs) >= top_n:
            break

    return top_n_jobs

def recommend_courses(text:str, top_n:int = 3) -> dict: 
    # TF-IDF Method 
    # if not os.path.exists('pkl/course_vectorizer.pkl'):
    #     course_vectorizer = TfidfVectorizer()
    #     course_vectorizer.fit(course_p_df['text'].astype(str), ignore_index=True)
    #     with open('pkl/course_vectorizer.pkl', 'wb') as f:
    #         pickle.dump(course_vectorizer, f)
    # else:
    #     with open('pkl/course_vectorizer.pkl', 'rb') as f:
    #         course_vectorizer = pickle.load(f)
    # # course_vectors = tfidf_vectorizer.transform(course_p_df['text'].astype(str))
    # # text_vector = tfidf_vectorizer.transform([text])
    # course_vectors = course_vectorizer.transform(course_p_df['text'].astype(str))
    # text_vector = course_vectorizer.transform([text])
    # dists = 1-cosine_similarity(text_vector, course_vectors).flatten()
    
    
    # ST Method
    if not os.path.exists('emb/st_courses_embeddings.pkl'):
        st_courses_embeddings = st_model.encode(course_p_df['text'].astype(str))
        with open('emb/st_courses_embeddings.pkl', 'wb') as f:
            pickle.dump(st_courses_embeddings, f)
    else:
        with open('emb/st_courses_embeddings.pkl', 'rb') as f:
            st_courses_embeddings = pickle.load(f)
    text_vector_st = st_model.encode(text)
    dists = 1-cosine_similarity([text_vector_st], st_courses_embeddings).flatten()
    
    course_df_sorted = course_df.iloc[np.argsort(dists), :]
    top_n_courses = {}
    top_n_courses['courses'] = course_df_sorted[course_df_sorted['product'] == 'Course'].head(top_n).to_dict(orient='records')
    top_n_courses['certifications'] = course_df_sorted[(course_df_sorted['product'] == 'Program') & (course_df_sorted['program_type'].apply(lambda x: "Professional Certificate" in x))].head(top_n).to_dict(orient='records')

    return top_n_courses

def recommend_career(r:int, i:int, a:int, s:int, e:int, c:int, top_n:int = 3) -> dict:    
    url = f"https://services.onetcenter.org/ws/mnm/interestprofiler/careers?Realistic={r}&Investigative={i}&Artistic={a}&Social={s}&Enterprising={e}&Conventional={c}"
    
    headers={'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.standard_b64encode(('career_recommendatio6' + ':' + '7627prw').encode()).decode(),
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
        
        top_n_careers.append({
            'title': title,
            'also_called': also_called,
            'what_they_do': what_they_do,
            'on_the_job': on_the_job,
            'knowledges': c_knowledges,
            'skills': c_skills,
            'abilities': c_abilities,
            'technologies': c_technologies
        })
    
    for career in top_n_careers:
        title_preprocessed = preprocessing(career['title'])
        also_called_preprocessed = preprocessing(career['also_called'])
        what_they_do_preprocessed = preprocessing(career['what_they_do'])
        on_the_job_preprocessed = preprocessing(career['on_the_job'])
        c_knowledges_preprocessed = " ".join([preprocessing(k) for k in career['knowledges']])
        c_skills_preprocessed = " ".join([preprocessing(s) for s in career['skills']])
        c_abilities_preprocessed = " ".join([preprocessing(a) for a in career['abilities']])
        c_technologies_preprocessed = " ".join([preprocessing(t) for t in career['technologies']])
        preprocessed_text = title_preprocessed + " " + also_called_preprocessed + " " + what_they_do_preprocessed + " " + on_the_job_preprocessed + " " + c_knowledges_preprocessed + " " + c_skills_preprocessed + " " + c_abilities_preprocessed + " " + c_technologies_preprocessed
        
        career['job'] = recommend_job(preprocessed_text)
        career['course'] = recommend_courses(preprocessed_text)
        
        # uncomment if you want to recommend courses based on the job text
        # for job in career['job']:
        #     job_text = job_df[job_df['job_link'] == job['job_link']]['text'].values[0]
        #     job['course'] = recommend_courses(job_text)
        
    
    return top_n_careers    
    
print(json.dumps(recommend_career(0, 5, 0, 0, 5, 5)))