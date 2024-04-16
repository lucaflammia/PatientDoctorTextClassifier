---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 2167613
    num_examples: 24746
  - name: validation
    num_bytes: 712512
    num_examples: 8249
  - name: test
    num_bytes: 716933
    num_examples: 8249
  download_size: 2372348
  dataset_size: 3597058
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

Label is used to give a context to the related text using the following map :

- 0 --> "PATIENT"
- 1 --> "DOCTOR"
- 2 --> "NEUTRAL"