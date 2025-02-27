import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch
from opacus import PrivacyEngine
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from opacus.grad_sample import GradSampleModule

MODEL_VERSION = "version_0.4"

class ModelTrainer:
    def __init__(self, base_dir, pii_data_path, instruction_data_path, model_save_path, 
                 model_name="bert-base-uncased", num_labels=2, use_ghost_clipping=False):
        self.BASE_DIR = base_dir
        self.PII_DATA_PATH = pii_data_path
        self.INSTRUCTION_DATA_PATH = instruction_data_path
        self.MODEL_SAVE_PATH = model_save_path
        self.MODEL_NAME = model_name
        self.NUM_LABELS = num_labels
        self.use_ghost_clipping = use_ghost_clipping

        # 모델과 토크나이저 로드
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.NUM_LABELS)
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)

        # 튜토리얼에 따라, BERT 전체 파라미터를 freeze하고 마지막 인코더 레이어, pooler, classifier만 학습하도록 설정
        for param in self.model.bert.parameters():
            param.requires_grad = False
        for param in self.model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.model.bert.pooler.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.pii_dataset = None
        self.instruction_dataset = None
        self.dataset =  None

        self.train_dataset = None
        self.eval_dataset = None

    def load_dataset(self, path):
        print(f"Loading dataset from: {path}")
        df = pd.read_csv(path)
        df = df.rename(columns={"sentence": "text", "label": "labels"})
        df = df[df["labels"] == 1]

        dataset = Dataset.from_pandas(df)
        print(f"Dataset loaded. {len(dataset)} samples with label=1.")
        return dataset

    def load_instruction_data(self):
        print(f"Loading instruction data from: {self.INSTRUCTION_DATA_PATH}")
        df = pd.read_csv(self.INSTRUCTION_DATA_PATH)
        df = df.rename(columns={"instruction": "text", "label": "labels"})

        pii_size = len(self.pii_dataset)
        sampled_df = df.sample(n=pii_size, random_state=42)
        print(f"Instruction data sampled: {pii_size} samples.")
        return Dataset.from_pandas(sampled_df)

    def merge_datasets(self):
        print("Merging datasets...")
        merged_data = {
            "text": list(self.pii_dataset["text"]) + list(self.instruction_dataset["text"]),
            "labels": list(self.pii_dataset["labels"]) + list(self.instruction_dataset["labels"])
        }
        merged_dataset = Dataset.from_dict(merged_data)
        print(f"Merged dataset size: {len(merged_dataset)} samples.")
        return merged_dataset

    def tokenize_data(self):
        def tokenize_function(examples):
            tokenized_inputs = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=297
            )
            # 라벨이 중첩 리스트가 아니도록 처리
            tokenized_inputs["labels"] = examples["labels"]
            return tokenized_inputs
        
        print("Tokenizing dataset...")
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        # 학습/평가 데이터셋으로 분리
        split_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = split_datasets["train"]
        self.eval_dataset = split_datasets["test"]
        print(f"Train size: {len(self.train_dataset)}, Eval size: {len(self.eval_dataset)}")

    def set_training_args(self):
        dataset_size = len(self.dataset)
        num_train_epochs = 7 if dataset_size < 1000 else 5 if dataset_size < 5000 else 3
        
        return TrainingArguments(
            output_dir=os.path.join(self.BASE_DIR, "results"),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.001,
            logging_dir=os.path.join(self.BASE_DIR, "models", "basemodel", MODEL_VERSION, "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="no"
        )

    def compute_metrics(self, eval_pred):
        metric_acc = load("accuracy")
        metric_prec = load("precision")
        metric_rec = load("recall")
        metric_f1 = load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        prec = metric_prec.compute(predictions=predictions, references=labels, average="binary")["precision"]
        rec = metric_rec.compute(predictions=predictions, references=labels, average="binary")["recall"]
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="binary")["f1"]

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1-score": f1}

    def train(self):
        # 데이터 tokenize 및 분할
        self.tokenize_data()
        training_args = self.set_training_args()
        self.train_dataset.set_format(type="torch")
        self.model.train()
        # 학습 가능한 파라미터만 업데이트하도록 optimizer 설정
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=5e-5)

        # DP에 적합한 큰 배치 사이즈 사용 (여기서는 train_dataset 사용)
        dp_batch_size = 512
        train_dataloader = DataLoader(self.train_dataset, batch_size=dp_batch_size, shuffle=True)

        # PrivacyEngine 설정 (ghost clipping 옵션이 True이면 grad_sample_mode 추가)
        privacy_engine = PrivacyEngine()
        dp_engine_kwargs = {
            "module": self.model,
            "optimizer": optimizer,
            "data_loader": train_dataloader,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            "max_grad_norm": 0.1,
            "epochs": training_args.num_train_epochs,
        }
        if self.use_ghost_clipping:
            dp_engine_kwargs["grad_sample_mode"] = "ghost"

        # DP용으로 wrapping (이 과정에서 optimizer, dataloader, model이 DP로 변환됨)
        self.model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(**dp_engine_kwargs)
        for epoch in range(training_args.num_train_epochs):
            print(f"Epoch {epoch+1}/{training_args.num_train_epochs}")
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                # 입력 배치 처리 (Trainer를 사용하지 않으므로, 직접 모델의 forward를 호출)
                # 배치 데이터가 dictionary 형식이라고 가정 (예: {"input_ids": ..., "attention_mask": ..., "labels": ...})
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss

                # 여기서 반드시 매 batch마다 optimizer.zero_grad(), loss.backward(), optimizer.step()를 호출합니다.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} loss: {avg_loss}")


    def save_model(self):
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        self.model._module.save_pretrained(self.MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(self.MODEL_SAVE_PATH)
        print(f"\nModel saved at: {self.MODEL_SAVE_PATH}")

    def evaluate(self):
        print("\nEvaluating the model...")
        self.model.eval()
        self.eval_dataset.set_format(type="torch")
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=64)
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                logits = outputs.logits
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        eval_pred = (all_logits, all_labels)
        metrics = self.compute_metrics(eval_pred)
        print("Evaluation results:", metrics)    

if __name__ == "__main__":
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "base_dataset", "base.csv")
    INSTRUCTION_DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "instructions.csv")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "basemodel", MODEL_VERSION)

    trainer = ModelTrainer(BASE_DIR, PREPROCESSED_DATA_PATH, INSTRUCTION_DATA_PATH, MODEL_SAVE_PATH, use_ghost_clipping=False)
    
    trainer.pii_dataset = trainer.load_dataset(PREPROCESSED_DATA_PATH)
    trainer.instruction_dataset = trainer.load_instruction_data()
    trainer.dataset = trainer.merge_datasets()
    
    trainer.train()
    trainer.save_model()
    trainer.evaluate()