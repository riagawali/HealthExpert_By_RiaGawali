import tkinter as tk
from tkinter import messagebox
from main import diagnose

def diagnose_disease():
    symptoms = symptoms_entry.get().split(', ')
    expert_diagnosis, ml_diagnosis = diagnose(symptoms)
    messagebox.showinfo("Diagnosis Result", f"Expert System Diagnosis: {expert_diagnosis}\nMachine Learning Diagnosis: {ml_diagnosis}")


root = tk.Tk()
root.title("Health Expert System")


symptoms_label = tk.Label(root, text="Enter symptoms (comma separated):")
symptoms_label.pack()

symptoms_entry = tk.Entry(root, width=50)
symptoms_entry.pack()

diagnose_button = tk.Button(root, text="Diagnose", command=diagnose_disease)
diagnose_button.pack()


root.mainloop()
