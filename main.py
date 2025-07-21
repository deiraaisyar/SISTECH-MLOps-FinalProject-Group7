from fastapi import FastAPI
from pydantic import BaseModel, Field
import app.data_processing as dp
import app.recommender as rec

JOB_CSV_PATH = "preprocessed/linkedin_jobs.csv"
JOB_INDEX_PATH = "app/jobs_tfidf.index"
COURSE_CSV_PATH = "preprocessed/edx_courses.csv"
COURSE_INDEX_PATH = "app/courses_tfidf.index"

app = FastAPI(title="Recommender")

class BaseQuery(BaseModel):
    request_id: int = Field(..., description="Unique request ID", example=1234)
    top_n: int = Field(5, description="Number of results to return", example=5)

class TextQuery(BaseQuery):
    description: str = Field(..., description="Free-form user input", example="Looking for a backend role.")

class CareerQuery(BaseQuery):
    r: int = Field(..., description="Realistic score", example=5)
    i: int = Field(..., description="Investigative score", example=4)
    a: int = Field(..., description="Artistic score", example=3)
    s: int = Field(..., description="Social score", example=2)
    e: int = Field(..., description="Enterprising score", example=1)
    c: int = Field(..., description="Conventional score", example=0)


# Load index and model at startup
jobs_df, job_model, job_index = dp.process_data(JOB_CSV_PATH, JOB_INDEX_PATH, "tfidf")
courses_df, course_model, course_index = dp.process_data(COURSE_CSV_PATH, COURSE_INDEX_PATH, "tfidf")


@app.post("/recommend-careers")
def recommend_careers(query: CareerQuery):
    results = rec.recommend_careers(r = query.r, i = query.i, a = query.a, s = query.s, e = query.e, c = query.c, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}


@app.post("/recommend-jobs")
def recommend_jobs(query: TextQuery):
    results = rec.recommend_jobs(query.description, job_model, job_index, jobs_df, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-courses")
def recommend_courses(query: TextQuery):
    results = rec.recommend_courses(query.description, course_model, course_index, courses_df, top_n=query.top_n)
    return {"request_id": query.request_id, "recommendations": results}

@app.get("/health")
def health():
    return {"status": "ok"}

# For local dev: run with 'uvicorn main:app --reload'