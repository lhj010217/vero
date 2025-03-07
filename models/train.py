import os
import random
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
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

MODEL_VERSION = "version_0.5"

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

        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.NUM_LABELS)
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)

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
        df = df[df["labels"] == 1]

        dataset = Dataset.from_pandas(df)
        print(f"Dataset loaded. {len(dataset)} samples with label=1.")
        return dataset

    def load_instruction_data(self):
        print(f"Loading instruction data from: {self.INSTRUCTION_DATA_PATH}")
        df = pd.read_csv(self.INSTRUCTION_DATA_PATH)

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
            tokenized_inputs["labels"] = examples["labels"]
            return tokenized_inputs
        
        print("Tokenizing dataset...")
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        split_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
        self.train_dataset = split_datasets["train"]
        self.eval_dataset = split_datasets["test"]
        print(f"Train size: {len(self.train_dataset)}, Eval size: {len(self.eval_dataset)}")

    def augment_sentence(self, sentence, n):
        """문장 데이터 증강 함수"""
        augmented = []
        n_synonym = int(n * 0.7)  # 70%는 동의어 치환
        n_delete = int(n * 0.1)   # 10%는 단어 삭제
        n_swap = int(n * 0.1)     # 10%는 단어 교환
        n_keyboard = n - (n_synonym + n_delete + n_swap)  # 남은 개수 (약 10%)는 키보드 오타
        
        # 동의어 치환: aug_p를 매번 랜덤하게 (0.0~1.0) 부여
        for _ in range(n_synonym):
            random_aug_p = random.uniform(0.0, 1.0)
            aug = naw.SynonymAug(aug_p=random_aug_p)
            aug_sent = aug.augment(sentence)
            augmented.append(aug_sent[0] if isinstance(aug_sent, list) else aug_sent)
        
        # 단어 삭제 (aug_p = 0.1)
        aug_del = naw.RandomWordAug(action="delete", aug_p=0.1)
        for _ in range(n_delete):
            aug_sent = aug_del.augment(sentence)
            augmented.append(aug_sent[0] if isinstance(aug_sent, list) else aug_sent)
        
        # 단어 교환 (swap) (aug_p = 0.1)
        aug_swap = naw.RandomWordAug(action="swap", aug_p=0.1)
        for _ in range(n_swap):
            aug_sent = aug_swap.augment(sentence)
            augmented.append(aug_sent[0] if isinstance(aug_sent, list) else aug_sent)
        
        # 타이포그래피 변형 (KeyboardAug)
        aug_key = nac.KeyboardAug()
        for _ in range(n_keyboard):
            aug_sent = aug_key.augment(sentence)
            augmented.append(aug_sent[0] if isinstance(aug_sent, list) else aug_sent)
        
        return augmented
    
    def augment_dataset(self, dataset, augment_factor=20):
        print(f"Augmenting dataset with factor {augment_factor}...")
        
        augmented_texts = []
        augmented_labels = []
        
        for sample in tqdm(dataset, desc="Augmenting data"):
            text = sample["text"]
            label = sample["labels"]
            
            # 원본 데이터 포함
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # 레이블이 1인 데이터에 대해서만 증강 적용 (PII 데이터)
            if label == 1:
                # 데이터 증강 적용
                augmented = self.augment_sentence(text, augment_factor)
                augmented_texts.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
        
        augmented_dataset = Dataset.from_dict({
            "text": augmented_texts,
            "labels": augmented_labels
        })
        
        print(f"Original dataset size: {len(dataset)}, Augmented dataset size: {len(augmented_dataset)}")
        return augmented_dataset
    
    def set_training_args(self):
        dataset_size = len(self.dataset)
        num_train_epochs = 5 if dataset_size < 1000 else 5 if dataset_size < 5000 else 3
        
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
        self.dataset = self.augment_dataset(self.dataset, augment_factor = 20)
        
        self.tokenize_data()
        training_args = self.set_training_args()
        self.train_dataset.set_format(type="torch")
        self.model.train()

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=5e-5)

        dp_batch_size = 128
        train_dataloader = DataLoader(self.train_dataset, batch_size=dp_batch_size, shuffle=True)

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

        self.model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(**dp_engine_kwargs)
        for epoch in range(training_args.num_train_epochs):
            print(f"Epoch {epoch+1}/{training_args.num_train_epochs}")
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss

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