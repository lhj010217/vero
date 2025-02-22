import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

class ModelPredictor:
    def __init__(self, company_name=None):
        # BASE_DIR 설정
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # MODEL_PATH 설정: company_name이 주어지면 해당 경로, 아니면 기본 경로로 설정
        if company_name:
            self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', company_name)
        else:
            self.MODEL_PATH = os.path.join(self.BASE_DIR, 'models', 'basemodel', 'version_0.3')

        # 모델 로드
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_PATH)

        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        # 모델 예측 수행
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        return {
            "logits": logits.numpy(),
            "probabilities": probabilities.numpy(),
            "predicted_class": predicted_class
        }
