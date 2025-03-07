import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from datasets import Dataset
import pandas as pd

from models.train import ModelTrainer
from models.predict import ModelPredictor
from models.translator import AutoMultiLangTranslator
import nltk
nltk.download('punkt')

router = APIRouter()
class PromptRequest(BaseModel):
    prompt: str

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_VERSION = "version_0.4"

def preprocess_csv(file: UploadFile) -> Dataset:
    try:
        df = pd.read_csv(file.file)
        
        if set(df.columns) != {"text", "labels"}:
            raise ValueError("CSV file must contain exactly 'text' and 'labels' columns, with no additional columns.")
        
        df = df.dropna()
        return Dataset.from_pandas(df)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocessing error with .csv : {str(e)}")

def preprocess_txt(file: UploadFile) -> Dataset:
    try:
        contents = file.file.read().decode("utf-8")
        
        nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(contents)
        
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        data = {"text": sentences, "labels": [1] * len(sentences)}

        return Dataset.from_dict(data)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocessing error with .txt : {str(e)}")

def process_string(input_string: str) -> Dataset:
    try:
        nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(input_string)
        
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        data = {"text": sentences, "labels": [1] * len(sentences)}

        return Dataset.from_dict(data)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocessing error with string input: {str(e)}")

def load_existing_model(company_name: str):
    company_model_path = os.path.join(MODELS_DIR, company_name, "model")
    base_model_path = os.path.join(MODELS_DIR, "basemodel", MODEL_VERSION)
    model_path = company_model_path if os.path.exists(company_model_path) else base_model_path
    save_path = os.path.join(MODELS_DIR, company_name)
    os.makedirs(save_path, exist_ok=True)
    return model_path, save_path

def update_training_status(save_path: str, message: str):
    """ 모델 학습 상태를 save_path의 status.txt 파일에 기록 """
    status_file = os.path.join(save_path, "status.txt")
    with open(status_file, "w", encoding="utf-8") as f:
        f.write(message + "\n")


def train_model(dataset: Dataset, model_path: str, save_path: str):
    update_training_status(save_path, "Training started...")
    trainer = ModelTrainer(
        base_dir=BASE_DIR,
        pii_data_path="",
        instruction_data_path=os.path.join(BASE_DIR, "data", "preprocessed", "instructions.csv"),
        model_save_path=save_path,
        model_name=model_path,
        num_labels=2
    )

    update_training_status(save_path, "Loading datasets...")
    trainer.pii_dataset = dataset
    trainer.instruction_dataset = trainer.load_instruction_data()
    trainer.dataset = trainer.merge_datasets()

    update_training_status(save_path, "Training in progress...")
    trainer.train()

    update_training_status(save_path, "Saving trained model...")
    trainer.save_model()

    update_training_status(save_path, "Training completed!")






