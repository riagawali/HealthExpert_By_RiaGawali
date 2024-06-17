import pandas as pd

class ExpertSystem:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def diagnose(self, symptoms):
        for rule in self.rules:
            diagnosis = rule(symptoms)
            if diagnosis:
                return diagnosis
        return "Unknown diagnosis"

def create_rule(disease, symptoms):
    def rule(symptoms_input):
        if all(symptom.strip().lower() in [s.strip().lower() for s in symptoms_input] for symptom in symptoms):
            return disease
        return None
    return rule


df = pd.read_csv('data/medical_symptoms.csv')


expert_system = ExpertSystem()


for index, row in df.iterrows():
    disease = row['Disease']
    symptoms = row.dropna().tolist()[1:] 
    rule = create_rule(disease, symptoms)
    expert_system.add_rule(rule)

def diagnose_symptoms(symptoms):
    return expert_system.diagnose(symptoms)
