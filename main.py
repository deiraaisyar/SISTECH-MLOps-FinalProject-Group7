from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
import app.data_processing as dp
import app.recommender as rec
from typing import Optional

JOB_JSON_PATH = "preprocessed/linkedin_jobs.json"
JOB_INDEX_PATH = "app/embeddings/jobs_st.index"
COURSE_JSON_PATH = "preprocessed/edx_courses.json"
COURSE_INDEX_PATH = "app/embeddings/courses_st.index"
MAJOR_JSON_PATH = "preprocessed/major_final.json"
MAJOR_INDEX_PATH = "app/embeddings/major_st.index"
CAREER_JSON_PATH = "preprocessed/onet_careers.json"
CAREER_INDEX_PATH = "app/embeddings/careers_st.index"
MODEL_PATH = "app/models/st_model"

app = FastAPI(title="Recommender")

class BaseQuery(BaseModel):
    request_id: int = Field(..., description="Unique request ID", example=1234)
    top_n: int = Field(5, description="Number of results to return", example=5)

class TextQuery(BaseQuery):
    query: str = Field(..., description="Input is from career recommendation's preprocessed_text or career recommendation's title")

class CareerQuery(BaseQuery):
    in_highschool: bool = Field(False, description="Whether the user is in high school", example=False)
    major: Optional[str] = Field(None, description="User's major or field of study", example="Computer Science")
    skills: str = Field(..., description="User's skills", example="Python, Machine Learning")
    interests: str = Field(..., description="User's interests", example="AI, Data Science")
    r: int = Field(..., description="Realistic score", example=5)
    i: int = Field(..., description="Investigative score", example=4)
    a: int = Field(..., description="Artistic score", example=3)
    s: int = Field(..., description="Social score", example=2)
    e: int = Field(..., description="Enterprising score", example=1)
    c: int = Field(..., description="Conventional score", example=0)
    
    @model_validator(mode="after")
    def validate_major_requirement(self) -> "CareerQuery":
        if not self.in_highschool and not self.major:
            raise ValueError("Field 'major' is required when 'in_highschool' is False.")
        return self


# Load index and model at startup
model = dp.get_model(MODEL_PATH)
jobs_data, jobs_index = dp.process_data(input_path=JOB_JSON_PATH, output_path=JOB_INDEX_PATH, model_path=MODEL_PATH)
courses_data, courses_index = dp.process_data(input_path=COURSE_JSON_PATH, output_path=COURSE_INDEX_PATH, model_path=MODEL_PATH)
careers_data, careers_index = dp.process_data(input_path=CAREER_JSON_PATH, output_path=CAREER_INDEX_PATH, model_path=MODEL_PATH)
programs_data, programs_index = dp.process_data(input_path=MAJOR_JSON_PATH, output_path=MAJOR_INDEX_PATH, model_path=MODEL_PATH)


@app.post("/recommend-careers")
def recommend_careers(query: CareerQuery):
    results = rec.recommend_careers(
        data=careers_data,
        model=model,
        index=careers_index,
        in_highschool=query.in_highschool,
        major=query.major,
        skills=query.skills,
        interests=query.interests,
        r=query.r, 
        i=query.i, 
        a=query.a, 
        s=query.s, 
        e=query.e, 
        c=query.c, 
        top_n=query.top_n
    )
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-jobs")
def recommend_jobs(query: TextQuery):
    results = rec.recommend_jobs(
        query_text = query.query, 
        model = model, 
        data=jobs_data,
        index=jobs_index,
        top_n=query.top_n 
    )
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-courses")
def recommend_courses(query: TextQuery):
    results = rec.recommend_courses(
        query_text = query.query, 
        model = model, 
        data=courses_data,
        index=courses_index,
        top_n=query.top_n
    )
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/recommend-programs")
def recommend_programs(query: TextQuery):
    results = rec.recommend_programs(
        query_text = query.query, 
        model = model, 
        data=programs_data,
        index=programs_index,
        top_n=query.top_n
    )
    return {"request_id": query.request_id, "recommendations": results}

@app.post("/get-job-articles")
def get_job_articles(query: TextQuery):
    results = rec.get_job_articles(query.query, top_n=query.top_n)
    return {"request_id": query.request_id, "articles": results}

@app.get("/health")
def health():
    return {"status": "ok"}
