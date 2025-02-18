import os
import importlib.util
from fastapi import FastAPI
from fastapi.routing import APIRouter
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Prompt Filtering API")
endpoints_dir = os.path.join(os.path.dirname(__file__), "endpoints")

for filename in os.listdir(endpoints_dir):
    if filename.endswith(".py") and filename != "__init__.py":

        module_name = f"api.endpoints.{filename[:-3]}"  
        module_spec = importlib.util.find_spec(module_name)

        if module_spec is not None:
            module = importlib.import_module(module_name)
            if hasattr(module, "router"):
                app.include_router(module.router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Prompt Filtering API is running"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
