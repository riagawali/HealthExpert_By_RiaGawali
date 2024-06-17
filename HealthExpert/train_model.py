import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

df = pd.read_csv('data/medical_symptoms.csv')


df = df.dropna(subset=['Disease'])  


diseases = df['Disease'].unique()
label_to_id = {label: i for i, label in enumerate(diseases)}
df['label'] = df['Disease'].map(label_to_id)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

df['text'] = df.drop(columns=['Disease', 'label']).apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.map(tokenize, batched=True)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(diseases))
training_args = TrainingArguments(
    output_dir='./model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)


trainer.train()


model.save_pretrained('./model')
tokenizer.save_pretrained('./model')


import json
with open('./model/label_to_id.json', 'w') as f:
    json.dump(label_to_id, f)
