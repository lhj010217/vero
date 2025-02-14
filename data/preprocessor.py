import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

df = pd.read_csv("raw/pii_dataset.csv")

prompts = df["prompt"].tolist()

sentences = []
labels = []

sensitive_keywords = ["name", "phone", "address", "url"]

for prompt in prompts:
    sents = sent_tokenize(prompt)
    for sent in sents:
        sent_lower = sent.lower()
        label = 1 if any(keyword in sent_lower for keyword in sensitive_keywords) else 0
        
        sentences.append(sent.strip())
        labels.append(label)

finetune_df = pd.DataFrame({
    "sentence": sentences,
    "label": labels
})

#print("Label Distribution:")
#print(finetune_df['label'].value_counts())

finetune_df.to_csv("base_dataset/base.csv", index=False)
