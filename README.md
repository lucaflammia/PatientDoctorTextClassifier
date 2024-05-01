# The Patient-Doctor Text Classifier

Here we propose the Patient-Doctor text classifier where the context of the text is classified within three possible solution: Patient, Doctor or Neutral. 
This classifier is the result of a finetuned model from a pretrained model taken from Hugging Face. 
Please check out the user profile https://huggingface.co/LukeGPT88.

### Model Description 

DistilBERT is a transformer model that performs text classification. I fine-tuned the model on with the purpose of classifying patient, doctor or neutral content, specifically when text is related to the supposed context. The model predicts 3 classes, which are Patient, Doctor or Neutral. 

The model is a fine-tuned version of [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert).

You can find the model using the link (https://huggingface.co/LukeGPT88/patient-doctor-text-classifier-eng).

It was fine-tuned on the prepared dataset (https://huggingface.co/datasets/LukeGPT88/patient-doctor-text-classifier-eng-dataset).

### How to Use 

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="LukeGPT88/patient-doctor-text-classifier-eng")
classifier("I prescribe you some drugs and you need to stay at home for a couple of days")
```

```python
Output:
[{'label': 'DOCTOR', 'score': 0.9992749094963074}]
```

## Contact

Please reach out to [luca.flammia@gmail.com](luca.flammia@gmail.com) if you have any questions or feedback.
