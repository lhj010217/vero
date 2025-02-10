from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/test/predict")
def filter_prompt(request: PromptRequest):
    if "400" in request.prompt.lower(): 
        raise HTTPException(status_code=400, detail="This prompt is not allowed.")

    return {"filtered_prompt": request.prompt}