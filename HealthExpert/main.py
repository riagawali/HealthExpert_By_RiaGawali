from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
from expert_system import diagnose_symptoms


model = BertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model')


with open('./model/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

def diagnose(symptoms):
    
    expert_diagnosis = diagnose_symptoms(symptoms)
    
    inputs = tokenizer(' '.join(symptoms), return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    ml_diagnosis = id_to_label[predicted_class_id]

    return expert_diagnosis, ml_diagnosis
