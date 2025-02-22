from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from datasets import Dataset
import os
import pandas as pd
from models.train import ModelTrainer
from models.predict import ModelPredictor


router = APIRouter()
class PromptRequest(BaseModel):
    prompt: str

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_VERSION = "version_0.3"
    
def preprocess_csv(file: UploadFile) -> Dataset:
    try:
        df = pd.read_csv(file.file)
        df = df.rename(columns={"instruction": "text", "label": "labels"})
        df = df.dropna()
        return Dataset.from_pandas(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocessing error: {str(e)}")

def load_existing_model(company_name: str):
    company_model_path = os.path.join(MODELS_DIR, company_name, "model")
    base_model_path = os.path.join(MODELS_DIR, "basemodel", MODEL_VERSION)
    model_path = company_model_path if os.path.exists(company_model_path) else base_model_path
    save_path = os.path.join(MODELS_DIR, company_name)
    os.makedirs(save_path, exist_ok=True)
    return model_path, save_path

def train_model(dataset: Dataset, model_path: str, save_path: str):
    trainer = ModelTrainer(
        base_dir=BASE_DIR,
        pii_data_path="",
        instruction_data_path=os.path.join(BASE_DIR, "data", "preprocessed", "instructions.csv"),
        model_save_path=save_path,
        model_name=model_path,
        num_labels=2
    )
    trainer.pii_dataset = dataset
    trainer.instruction_dataset = trainer.load_instruction_data()
    trainer.dataset = trainer.merge_datasets()
    trainer.train()
    trainer.save_model()





