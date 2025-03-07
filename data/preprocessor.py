import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

input_file = "raw/pii_dataset.csv"
output_file = "base_dataset/base.csv"

df = pd.read_csv(input_file)

sentences = []
labels = []

for _, row in df.iterrows():
    prompt = row["text"]
    
    name = str(row.get("name", "")).strip()
    phone = str(row.get("phone", "")).strip()
    address = str(row.get("address", "")).strip()
    url = str(row.get("url", "")).strip()
    
    pii_values = [name, phone, address, url]
    pii_values = [value for value in pii_values if value] 
    
    for sent in sent_tokenize(prompt):
        sent_clean = sent.strip()
        
        label = 1 if any(value in sent_clean for value in pii_values) else 0
        
        sentences.append(sent_clean)
        labels.append(label)

finetune_df = pd.DataFrame({
    "text": sentences,
    "labels": labels
})

finetune_df.to_csv(output_file, index=False)

print("Label Distribution:")
print(finetune_df['labels'].value_counts())
