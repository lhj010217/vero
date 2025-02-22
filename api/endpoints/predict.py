from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.predict import predictor

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/predict")
def filter_prompt(request: PromptRequest):
    result = predictor.predict(request.prompt)
    if result["predicted_class"] == 1: 
        raise HTTPException(status_code=400, detail="This prompt is not allowed.")

    return {"filtered_prompt": request.prompt}
