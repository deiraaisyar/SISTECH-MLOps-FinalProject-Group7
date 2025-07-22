from fastapi import FastAPI
from pydantic import BaseModel, Field
import app.data_processing as dp
import app.recommender as rec

JOB_CSV_PATH = "preprocessed/linkedin_jobs.csv"
JOB_JSON_PATH = "preprocessed/linkedin_jobs.json"
JOB_INDEX_PATH = "app/jobs_tfidf.index"
COURSE_CSV_PATH = "preprocessed/edx_courses.csv"
COURSE_JSON_PATH = "preprocessed/edx_courses.json"
COURSE_INDEX_PATH = "app/courses_tfidf.index"
MAJOR_CSV_PATH = "preprocessed/major_final.csv"
MAJOR_INDEX_PATH = "app/major_tfidf.index"

app = FastAPI(title="Recommender")

class BaseQuery(BaseModel):
    request_id: int = Field(..., description="Unique request ID", example=1234)
    top_n: int = Field(3, description="Number of results to return", example=3)

class TextQuery(BaseQuery):
    article_query: str = Field(..., description="Job trends query", example="Data Science trends 2025")
    query: str = Field(..., description="Input is from career recommendation's text")

class CareerQuery(BaseQuery):
    r: int = Field(..., description="Realistic score", example=5)
    i: int = Field(..., description="Investigative score", example=4)
    a: int = Field(..., description="Artistic score", example=3)
    s: int = Field(..., description="Social score", example=2)
    e: int = Field(..., description="Enterprising score", example=1)
    c: int = Field(..., description="Conventional score", example=0)


# Load index and model at startup
# jobs_data, job_model, job_index = dp.process_data(JOB_CSV_PATH, JOB_INDEX_PATH, "tfidf")
# courses_data, course_model, course_index = dp.process_data(COURSE_CSV_PATH, COURSE_INDEX_PATH, "tfidf")

jobs_data, job_model, job_index = dp.process_data(JOB_JSON_PATH, JOB_INDEX_PATH, "tfidf")
courses_data, course_model, course_index = dp.process_data(COURSE_JSON_PATH, COURSE_INDEX_PATH, "tfidf")
programs_df, program_model, program_index = dp.process_data(MAJOR_CSV_PATH, MAJOR_INDEX_PATH, "tfidf")


@app.post("/recommend-careers")
def recommend_careers(query: CareerQuery):
    results = rec.recommend_careers(r = query.r, i = query.i, a = query.a, s = query.s, e = query.e, c = query.c, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-jobs")
def recommend_jobs(query: TextQuery):
    results = rec.recommend_jobs(query.query, job_model, job_index, jobs_data, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-courses")
def recommend_courses(query: TextQuery):
    results = rec.recommend_courses(query.query, course_model, course_index, courses_data, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-programs")
def recommend_programs(query: TextQuery):
    results = rec.recommend_programs(query.query, program_model, program_index, programs_df, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/get-job-articles")
def get_job_articles(query: TextQuery):
    results = rec.get_job_articles(query.article_query, top_n=query.top_n)
    return {"request_id": query.request_id, "articles": results}

@app.get("/health")
def health():
    return {"status": "ok"}
