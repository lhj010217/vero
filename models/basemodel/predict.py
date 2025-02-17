import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

class ModelPredictor:
    def __init__(self, model_path):
        self.MODEL_PATH = model_path
        self.MODEL_NAME = model_path

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR,'basemodel', 'version_0.2')

predictor = ModelPredictor(MODEL_PATH)
result = predictor.predict("제 이름은 이형준이고, 전화번호는 010-1413-1384입니다. 집 주소는 경기도남양주시와부읍입니다.")
print(result)
