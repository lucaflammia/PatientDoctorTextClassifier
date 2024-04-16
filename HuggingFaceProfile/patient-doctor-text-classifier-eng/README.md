---
language: "en"
tags:
- distilbert-base-uncased
- text-classification
- patient
- doctor 

widget:
- text: "I've got flu"
- text: "I prescribe you some drugs and you need to stay at home for a couple of days"
- text: "Let's move to the theatre this evening!"
---

# distilbert-base-uncased-finetuned-text-classification

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.

# Fine-tuned DistilBERT-base-uncased for Patient-Doctor Classification

# Model Description 

DistilBERT is a transformer model that performs text classification. I fine-tuned the model on with the purpose of classifying patient, doctor or neutral content, specifically when text is related to the supposed context. The model predicts 3 classes, which are Patient, Doctor or Neutral. 

The model is a fine-tuned version of [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert).

It was fine-tuned on the prepared dataset (https://huggingface.co/datasets/LukeGPT88/text-classification-dataset).

It achieves the following results on the evaluation set:
- Loss: 0.0501
- Accuracy: 0.9861

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.115         | 1.0   | 774  | 0.0486          | 0.9864   |
| 0.0301        | 2.0   | 1548 | 0.0501          | 0.9861   |


### Framework versions

- Transformers 4.37.0
- Pytorch 2.1.2
- Datasets 2.1.0
- Tokenizers 0.15.1

# How to Use 

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="LukeGPT88/patient-doctor-text-classifier")
classifier("I see you’ve set aside this special time to humiliate yourself in public.")
```

```python
Output:
[{'label': 'NEUTRAL', 'score': 0.9890775680541992}]
```

# Contact

Please reach out to [luca.flammia@gmail.com](luca.flammia@gmail.com) if you have any questions or feedback.

---
