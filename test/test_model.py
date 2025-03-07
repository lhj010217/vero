import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ---------------------------
# 1. 데이터 로드 및 준비
# ---------------------------
# example.txt는 각 줄에 문장만 있다고 가정
data_file = "example.txt"
sentences = []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sentences.append(line)

# ---------------------------
# 2. 데이터 증강
# ---------------------------
# 한 문장당 증강할 개수 n (예: 100)
n = 20

def augment_sentence(sentence, n):
    augmented = []
    # 증강 비율 산출: 전체 n개 중
    n_synonym = int(n * 0.7)
    n_delete  = int(n * 0.1)
    n_swap    = int(n * 0.1)
    n_keyboard = n - (n_synonym + n_delete + n_swap)  # 남은 개수 (약 10%)
    
    # 동의어 치환: aug_p를 매번 랜덤하게 (0.0~1.0) 부여
    for _ in range(n_synonym):
        random_aug_p = random.uniform(0.0, 1.0)
        aug = naw.SynonymAug(aug_p=random_aug_p)
        aug_sent = aug.augment(sentence)
        # nlpaug는 리스트로 반환하므로 첫 번째 요소 사용
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
    
    # 타이포그래피 변형 (KeyboardAug) (aug_p 인자 제거)
    aug_key = nac.KeyboardAug()
    for _ in range(n_keyboard):
        aug_sent = aug_key.augment(sentence)
        augmented.append(aug_sent[0] if isinstance(aug_sent, list) else aug_sent)
    
    return augmented

# 원본 데이터에 대해 증강 수행: 각 문장마다 n개 증강 + 원본 문장 포함
all_sentences = []
for sent in sentences:
    # 원본 문장 포함
    all_sentences.append(sent)
    augmented_sentences = augment_sentence(sent, n)
    all_sentences.extend(augmented_sentences)

print(f"전체 증강 후 데이터 개수: {len(all_sentences)}")

# ---------------------------
# 증강된 데이터를 파일로 저장 (2. 데이터 증강 단계 직후)
# ---------------------------
with open("auged_example.txt", "w", encoding="utf-8") as f:
    for sent in all_sentences:
        f.write(f"{sent}\n")
