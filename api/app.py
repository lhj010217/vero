import os
import importlib.util
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.management_companyAPI import router as create_company_router, load_all_company_routes
from models.predict import ModelPredictor

app = FastAPI(title="Company API Generator")

BASE_DIR = os.path.dirname(__file__)
ENDPOINTS_DIR = os.path.join(BASE_DIR, "endpoints")

app.include_router(create_company_router)
load_all_company_routes(app)

@app.get("/")
def read_root():
    return {"message": "Prompt Filtering API is running"}

predictor = ModelPredictor()
class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
def filter_prompt(request: PromptRequest):
    result = predictor.predict(request.prompt)
    if result["predicted_class"] == 1: 
        raise HTTPException(status_code=400, detail="This prompt is not allowed.")
    return {"filtered_prompt": request.prompt}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
