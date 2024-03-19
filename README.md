# PatientDoctorTextClassifier
Here we propose Patient-Doctor text classifier where the context of the text is classified within three possible solution: Patient, Doctor or Neutral. 
This classifier is the result of a finetuned model from a pretrained model taken from Hugging Face. 
Please check it out at https://huggingface.co/LukeGPT88/patient_doctor_text_classifier

# How to Use 

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="LukeGPT88/patient_doctor_text_classifier")
classifier("I prescribe you some drugs and you need to stay at home for a couple of days")
```

```python
Output:
[{'label': 'DOCTOR', 'score': 0.9946979284286499}]
