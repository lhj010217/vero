import json
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.routing import APIRoute
from pydantic import BaseModel
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ML_utils import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANY_FILE = os.path.join(BASE_DIR, "companies.json")

router = APIRouter()
class CompanyRequest(BaseModel):
    company_name: str

class PromptRequest(BaseModel):
    prompt: str

def load_companies():
    if not os.path.exists(COMPANY_FILE):
        return []

    try:
        with open(COMPANY_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def save_companies(companies):
    with open(COMPANY_FILE, "w") as f:
        json.dump(companies, f, indent=4)

def create_company_router(company_name: str) -> APIRouter:
    company_router = APIRouter(prefix=f"/api/{company_name}")

    @company_router.post("/info")
    async def get_company_info(request: CompanyRequest = Body(...)):
        companies = load_companies()
        company = next((c for c in companies if c["name"] == request.company_name), None)
        if not company:
            raise HTTPException(status_code=404, detail="Company not found.")
        return {"company": company["name"], "info": company}

    @company_router.post("/train")
    async def train(
        file: UploadFile = File(...), 
        company_name: str = Form(...)
    ):
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name parameter is required.")
        
        dataset = preprocess_csv(file)
        model_path, save_path = load_existing_model(company_name)
        train_model(dataset, model_path, save_path)
        return {"message": "Update complete", "company_name": company_name}

    @company_router.post("/predict")
    async def predict(
        request: PromptRequest
    ):
        predictor = ModelPredictor(company_name)
        result = predictor.predict(request.prompt)
        if result["predicted_class"] == 1:
            raise HTTPException(status_code=400, detail="This prompt is not allowed.")
        return {"filtered_prompt": request.prompt}

    return company_router

def remove_company_routes(app, company_name: str):
    prefix = f"/api/{company_name}"
    app.router.routes = [
        route for route in app.router.routes
        if not (isinstance(route, APIRoute) and route.path.startswith(prefix))
    ]

def load_all_company_routes(app):
    companies = load_companies()
    for company in companies:
        app.include_router(create_company_router(company["name"]))

@router.post("/api/create_company")
async def create_company(request: CompanyRequest):
    companies = load_companies()

    if any(company["name"] == request.company_name for company in companies):
        raise HTTPException(status_code=400, detail="Company already exists.")

    new_company = {
        "name": request.company_name,
        "created_at": datetime.utcnow().isoformat(),
        "routes": [
            f"/api/{request.company_name}/info",
            f"/api/{request.company_name}/train",
            f"/api/{request.company_name}/predict"
        ]
    }

    companies.append(new_company)
    save_companies(companies)

    from api.app import app
    app.include_router(create_company_router(request.company_name))

    return {"status": "success", "company": new_company}

@router.delete("/api/delete_company")
async def delete_company(request: CompanyRequest):
    companies = load_companies()

    company = next((c for c in companies if c["name"] == request.company_name), None)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found.")

    companies.remove(company)
    save_companies(companies)

    from api.app import app
    remove_company_routes(app, request.company_name)

    return {"status": "success", "deleted_company": request.company_name}

class UpdateCompanyRequest(BaseModel):
    old_name: str
    new_name: str

@router.put("/api/update_company")
async def update_company(request: UpdateCompanyRequest):
    companies = load_companies()

    company = next((c for c in companies if c["name"] == request.old_name), None)
    if not company:
        raise HTTPException(status_code=404, detail="Old company not found.")
    if any(c["name"] == request.new_name for c in companies):
        raise HTTPException(status_code=400, detail="New company name already exists.")

    company["name"] = request.new_name
    company["routes"] = [
        f"/api/{request.new_name}/info",
        f"/api/{request.new_name}/train",
        f"/api/{request.new_name}/predict"
    ]
    save_companies(companies)

    from api.app import app
    remove_company_routes(app, request.old_name)
    app.include_router(create_company_router(request.new_name))

    return {"status": "success", "old_name": request.old_name, "new_name": request.new_name}

@router.get("/api/companies")
async def list_companies():
    return {"companies": load_companies()}