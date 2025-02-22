import os
import numpy as np
import pandas as pd
from datasets import Dataset
from evaluate import load as load_metric
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

MODEL_VERSION = "version_0.3"

class ModelTrainer:
    def __init__(self, base_dir, pii_data_path, instruction_data_path, model_save_path, model_name="bert-base-uncased", num_labels=2):
        self.BASE_DIR = base_dir
        self.PII_DATA_PATH = pii_data_path
        self.INSTRUCTION_DATA_PATH = instruction_data_path
        self.MODEL_SAVE_PATH = model_save_path
        self.MODEL_NAME = model_name
        self.NUM_LABELS = num_labels

        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.NUM_LABELS)
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)

        self.pii_dataset = None
        self.instruction_dataset = None
        self.dataset =  None

        self.train_dataset = None
        self.eval_dataset = None

    def load_dataset(self, path):
        print(f"Loading dataset from: {path}")
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            if "sentence" in df.columns:
                df = df.rename(columns={"sentence": "text"})
            if "label" in df.columns:
                df = df.rename(columns={"label": "labels"})
            
            df = df[df["labels"] == 1]
            dataset = Dataset.from_pandas(df)
            print(f"Dataset loaded from CSV. (Filtered: {len(dataset)} samples with label=1)")
        else:
            dataset = Dataset.load_from_disk(path)
            print("Dataset loaded from disk.")
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
        print("Merging PII and instruction datasets...")
        merged_dataset = Dataset.from_dict({
            "text": self.pii_dataset["text"] + self.instruction_dataset["text"],
            "labels": self.pii_dataset["labels"] + self.instruction_dataset["labels"]
        })
        print(f"Merged dataset size: {len(merged_dataset)} samples.")
        return merged_dataset

    def tokenize_data(self):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=297
            )
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)
        self.train_dataset, self.eval_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42).values()

    def set_training_args(self):
    # 데이터셋 크기에 따른 epoch 수 계산
        dataset_size = len(self.dataset)
        
        if dataset_size < 1000:
            num_train_epochs = 5  # 작은 데이터셋은 더 많은 epoch
        elif dataset_size < 5000:
            num_train_epochs = 3  # 중간 크기 데이터셋
        else:
            num_train_epochs = 1  # 큰 데이터셋은 적은 epoch
        
        return TrainingArguments(
            save_strategy="no",
            output_dir=os.path.join(self.BASE_DIR, "results"),
            num_train_epochs=num_train_epochs,  # 동적으로 설정된 epoch 사용
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.001,
            logging_dir=os.path.join(self.BASE_DIR, "models", "basemodel", MODEL_VERSION, "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
        )

    def compute_metrics(self, eval_pred):
        metric_acc = load_metric("accuracy")
        metric_prec = load_metric("precision")
        metric_rec = load_metric("recall")
        metric_f1 = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = metric_acc.compute(predictions=predictions, references=labels)
        prec = metric_prec.compute(predictions=predictions, references=labels, average="binary")
        rec = metric_rec.compute(predictions=predictions, references=labels, average="binary")
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="binary")

        return {
            "accuracy": acc["accuracy"],
            "precision": prec["precision"],
            "recall": rec["recall"],
            "f1-score": f1["f1"]
        }

    def train(self):
        self.tokenize_data()
        training_args = self.set_training_args()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        print("\nEvaluation metrics:")
        for metric, score in eval_metrics.items():
            print(f"{metric}: {score}")

    def save_model(self):
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        self.model.save_pretrained(self.MODEL_SAVE_PATH)
        self.tokenizer.save_pretrained(self.MODEL_SAVE_PATH)
        print(f"\n Model saved at: {self.MODEL_SAVE_PATH}")


'''
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PII_DATA_PATH = os.path.join(BASE_DIR, "data", "base_dataset", "base.csv")
INSTRUCTION_DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "instructions.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "basemodel", MODEL_VERSION)
'''