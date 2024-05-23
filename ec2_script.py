import numpy as np
import pandas as pd
from transformers import pipeline
import csv
import os


HF_USERNAME = "LukeGPT88"
PROJECT_NAME = "patient-doctor-text-classifier"
SUB_PROJECT_NAME = "eng"
TASK = f"{PROJECT_NAME}-{SUB_PROJECT_NAME}"

FOLDERPATH = 'PulsarSearchesExport'
TOPIC_CC = 'HavasGlobal-Lupus'
TOPIC_SC = 'havas_global-lupus'

df = pd.read_excel(f'{FOLDERPATH}/{TOPIC_CC}/107503-all-2024-05-17-07-39-05-1163478.xlsx')

df.drop_duplicates(subset=['content'], inplace=True)
df.dropna(subset=['content'], inplace=True)

contents = df['content'].values
bios = df['bio'].values
data = {'contents': contents, 'bios': bios}
# bios = bios.where(bios.notna(), bios, None).values

def get_csv_file(data, split=0):
  out_df = pd.DataFrame({
    "Account": [res['account'] for res in data], "User": [res['user'] for res in data], "Content": [res['content']['text'] for res in data], "Predicted Content Type (Label / Score)": [f"{res['content']['label']} / {res['content']['score']}" for res in data], 
    "Bio": [res['bio']['text'] for res in data], "Predicted Bio Type (Label / Score)": [f"{res['bio']['label']} / {res['bio']['score']}" for res in data]
  })
  filename = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference_p{int(split)}.csv' if split != 0 else f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference.csv'
  out_df.to_csv(filename)
  if split > 1:
    filename_old = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference_p{int(split - 2)}.csv'
    if(os.path.exists(filename_old) and os.path.isfile(filename_old)):
      os.remove(filename_old)

classifier = pipeline("text-classification", model=f"{HF_USERNAME}/{TASK}")
outcomes = []
tot_rows = list(map(lambda x, w, y, z: (x, w, y, z), df['user name'].values,df['user screen name'].values, data['contents'], data['bios']))
for i, (account, user, content, bio) in enumerate(tot_rows):
  if len(content) <= 500:
    if type(bio) != str:
      bio = ''
    outcomes.append({"account": account, "user": user, "content": {**{"text": content}, **classifier(content)[0]}, "bio": {**{"text": bio}, **classifier(bio)[0]}})
  if i%1000 == 0 and i != 0:
    split = i/1000
    get_csv_file(outcomes, split)
  if i == len(tot_rows) - 1:
    get_csv_file(outcomes)
