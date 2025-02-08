from fastapi import FastAPI
from api.endpoints.filter import router as filter_router

app = FastAPI(title="Prompt Filtering API")

app.include_router(filter_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Prompt Filtering API is running"}




