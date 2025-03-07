import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
import evaluate
from sklearn.model_selection import train_test_split

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------------------
# 1. 데이터 로드
# ---------------------------
# auged_example.txt에서 문장 로드 (레이블 1)
data_file = "auged_example.txt"
with open(data_file, "r", encoding="utf-8") as f:
    texts_aug = [line.strip() for line in f if line.strip()]

# instruction.csv에서 문장 로드 (레이블 0)
instruction_file = "../data/preprocessed/instructions.csv"
instruction_df = pd.read_csv(instruction_file)

# instruction.csv에서 문장들만 가져오기
texts_inst = instruction_df['text'].tolist()  # assuming the column name is 'text'

# instruction.csv에서 증강된 데이터의 개수만큼 랜덤 샘플링
num_samples = len(texts_aug)
random_inst_samples = random.sample(texts_inst, num_samples)

# 레이블 부여
texts_all = texts_aug + random_inst_samples
labels_all = [1] * len(texts_aug) + [0] * len(random_inst_samples)

# ---------------------------
# 2. 학습/테스트 데이터 분할
# ---------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(texts_all, labels_all, test_size=0.2, random_state=seed)

# ---------------------------
# 3. 모델 및 토크나이저 로드
# ---------------------------
model_path = "../models/basemodel/version_0.4"  # 기존 모델 경로
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# ---------------------------
# 4. 데이터 전처리
# ---------------------------
# Hugging Face Dataset 객체 생성
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ---------------------------
# 5. 전체 데이터셋 크기 확인
# ---------------------------
print(f"학습 데이터셋 크기: {len(train_dataset)}")
print(f"테스트 데이터셋 크기: {len(test_dataset)}")

# ---------------------------
# 6. 평가 지표 준비
# ---------------------------
metric = evaluate.load("accuracy")

# ---------------------------
# 7. 파인튜닝 설정
# ---------------------------
training_args = TrainingArguments(
    output_dir="./classifier_results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=50,
    seed=seed,
)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ---------------------------
# 8. 모델 파인튜닝 및 평가
# ---------------------------
trainer.train()
trainer.save_model("./classifier_finetuned_model")

# ---------------------------
# 9. 테스트 성능 평가
# ---------------------------
eval_results = trainer.evaluate()
print(f"테스트 성능 평가 결과: {eval_results}")
