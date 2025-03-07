import json
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.routing import APIRoute
from pydantic import BaseModel
from datetime import datetime
from .ML_utils import *
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANY_FILE = os.path.join(THIS_DIR, "companies.json")

router = APIRouter()
class CompanyRequest(BaseModel):
    company_name: str

class PromptRequest(BaseModel):
    prompt: str

class UpdateCompanyRequest(BaseModel):
    old_name: str
    new_name: str

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
        company_name: str = Form(...),
        file: UploadFile = File(None),
        string: str = Form(None)
    ):
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name parameter is required.")
        
        if file is None and string is None:
            raise HTTPException(status_code=400, detail="Either file or string must be provided.")
        
        datasets = []
        if file:
            file_ext = file.filename.split(".")[-1].lower()
            if file_ext == "csv":
                datasets.append(preprocess_csv(file))
            elif file_ext == "txt":
                datasets.append(preprocess_txt(file))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or TXT file.")
        
        if string:
            datasets.append(process_string(string))
        
        combined_dataset = Dataset.from_dict({
            "text": sum([ds["text"] for ds in datasets], []),
            "labels": sum([ds["labels"] for ds in datasets], [])
        })
        
        translator = AutoMultiLangTranslator()
        
        def translate_sample(data):
            try:
                data["text"] = translator.translate(data["text"])
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return data
        
        translated_dataset = combined_dataset.map(translate_sample)
        model_path, save_path = load_existing_model(company_name)
        train_model(translated_dataset, model_path, save_path)
        
        return {"message": "Update complete", "company_name": company_name}

    @company_router.post("/predict")
    async def predict(
        request: PromptRequest
    ):
        translator = AutoMultiLangTranslator()
        try:
            translated_prompt = translator.translate(request.prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        predictor = ModelPredictor(company_name)
        result = predictor.predict(translated_prompt)
        if result["predicted_class"] == 1:
            raise HTTPException(status_code=400, detail="This prompt is not allowed.")
        return {"filtered_prompt": request.prompt}

    @company_router.get("/status")
    async def get_training_status():
        """ 회사의 모델 학습 상태를 반환 """
        company_model_dir = os.path.join(MODELS_DIR, company_name)
        status_file = os.path.join(company_model_dir, "status.txt")

        if not os.path.exists(status_file):
            return {"status": "Training has not been started for this company."}

        with open(status_file, "r", encoding="utf-8") as f:
            status_message = f.read().strip()

        return {"status": status_message}
    
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

    existing_company = next((c for c in companies if c["name"] == request.company_name), None)
    if existing_company:
        return {"status": "success", "message": "Company already exists.", "company": existing_company}

    new_company = {
        "name": request.company_name,
        "created_at": datetime.utcnow().isoformat(),
        "routes": [
            f"/api/{request.company_name}/info",
            f"/api/{request.company_name}/train",
            f"/api/{request.company_name}/predict",
            f"/api/{request.company_name}/status",
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

    company_model_dir = os.path.join(MODELS_DIR, request.company_name)
    if os.path.exists(company_model_dir):
        shutil.rmtree(company_model_dir)

    return {"status": "success", "deleted_company": request.company_name}


@router.put("/api/update_company")
async def update_company(request: UpdateCompanyRequest):
    companies = load_companies()

    company = next((c for c in companies if c["name"] == request.old_name), None)
    if not company:
        raise HTTPException(status_code=404, detail="Old company not found.")
    
    existing_company = next((c for c in companies if c["name"] == request.new_name), None)
    if existing_company:
        return {"status": "success", "message": "Company already exists.", "company": existing_company}

    company["name"] = request.new_name
    company["routes"] = [
        f"/api/{request.new_name}/info",
        f"/api/{request.new_name}/train",
        f"/api/{request.new_name}/predict",
        f"/api/{request.new_name}/status",
    ]
    save_companies(companies)

    old_company_model_dir = os.path.join(MODELS_DIR, request.old_name)
    new_company_model_dir = os.path.join(MODELS_DIR, request.new_name)

    if os.path.exists(old_company_model_dir):
        os.rename(old_company_model_dir, new_company_model_dir)

    from api.app import app
    remove_company_routes(app, request.old_name)
    app.include_router(create_company_router(request.new_name))

    return {"status": "success", "old_name": request.old_name, "new_name": request.new_name}

@router.get("/api/companies")
async def list_companies():
    return {"companies": load_companies()}