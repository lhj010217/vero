import os
import pandas as pd
import re
from transformers import BertTokenizer
from datasets import Dataset

# 데이터 전처리기
class DataPreprocessor:
    def __init__(self, base_dir, raw_data_path, preprocessed_data_path):
        self.BASE_DIR = base_dir
        self.RAW_DATA_PATH = raw_data_path
        self.PREPROCESSED_DATA_PATH = preprocessed_data_path
        self.EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        self.PHONE_PATTERN = r"\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b"
        self.USERNAME_PATTERN = r"@[a-zA-Z0-9_]+"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.processed_data = []

    def preprocess_file(self, filename):
        if filename.endswith(".csv"):
            file_path = os.path.join(self.RAW_DATA_PATH, filename)
            df = pd.read_csv(file_path)

            # 필요한 컬럼만 선택
            selected_columns = ["text", "tokens", "labels", "email", "phone", "username"]
            df = df[[col for col in selected_columns if col in df.columns]]

            # 전처리
            df = df.fillna("") 
            df["labels"] = df["labels"].apply(lambda x: 1 if "PII" in x else 0)
            df["email"] = df["email"].apply(lambda x: self.extract_email(x))
            df["phone"] = df["phone"].apply(lambda x: self.extract_phone(x)) 
            df["username"] = df["username"].apply(lambda x: self.extract_username(x)) 
            
            # 토큰화 및 input_ids, attention_mask 생성
            df = self.tokenize_data(df)

            # datasets로 변환
            dataset = Dataset.from_pandas(df)
            self.processed_data.append(dataset)

    # 이메일 추출
    def extract_email(self, text):
        return re.findall(self.EMAIL_PATTERN, text)[0] if re.findall(self.EMAIL_PATTERN, text) else ""

    # 전화번호 추출
    def extract_phone(self, text):
        return re.findall(self.PHONE_PATTERN, text)[0] if re.findall(self.PHONE_PATTERN, text) else ""
    
    # 닉네임 추출
    def extract_username(self, text):
        return re.findall(self.USERNAME_PATTERN, text)[0] if re.findall(self.USERNAME_PATTERN, text) else ""

    # 토큰화
    def tokenize_data(self, df):
        # input_ids와 attention_mask 생성
        df["input_ids"] = df["tokens"].apply(lambda x: self.tokenizer.convert_tokens_to_ids(eval(x)))
        df["attention_mask"] = df["input_ids"].apply(lambda x: [1] * len(x))  # attention mask 생성
        return df

    def save_processed_data(self):
        if self.processed_data:
            # 최종 데이터셋 결합
            final_dataset = self.processed_data[0]  # 첫 번째 데이터셋을 기준으로 결합
            for dataset in self.processed_data[1:]:
                final_dataset = final_dataset.concatenate(dataset)
            
            # 결과 저장
            output_file = os.path.join(self.PREPROCESSED_DATA_PATH, "preprocessed_pii_data.arrow")
            final_dataset.save_to_disk(output_file)
            print(f"Successfully saved preprocessed data to {output_file}")
        else:
            print("Failed to preprocess data")

    def preprocess_all_files(self, filenames):
        for filename in filenames:
            self.preprocess_file(filename)
        self.save_processed_data()


# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PREPROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed")

# 파일 목록 설정
filenames = ["pii_dataset.csv"]

# 데이터 전처리 실행
preprocessor = DataPreprocessor(BASE_DIR, RAW_DATA_PATH, PREPROCESSED_DATA_PATH)
preprocessor.preprocess_all_files(filenames)
