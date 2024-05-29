import numpy as np
import argparse
import pandas as pd
import time
import re
from transformers import pipeline
import csv
import os


HF_USERNAME = "LukeGPT88"
PROJECT_NAME = "patient-doctor-text-classifier"
SUB_PROJECT_NAME = "eng"
TASK = f"{PROJECT_NAME}-{SUB_PROJECT_NAME}-0528"

FOLDERPATH = 'PulsarSearchesExport'
TOPIC_CC = 'BreastCancer'
TOPIC_SC = 'breast_cancer'

def get_data(df):
  contents = df['content'].values
  bios = df['bio'].values
  data = {'contents': contents, 'bios': bios}
  # bios = bios.where(bios.notna(), bios, None).values
  return data

def get_csv_file(data, check_bios, split=0):
  out_df = pd.DataFrame({
    "Account": [res['account'] for res in data], "User": [res['user'] for res in data], "Content": [res['content']['text'] for res in data], "Predicted Content Type (Label / Score)": [f"{res['content']['label']} / {res['content']['score']}" for res in data], 
    "Bio": [res['bio']['text'] for res in data], "Predicted Bio Type (Label / Score)": [f"{res['bio']['label']} / {res['bio']['score']}" for res in data]
  })
  if check_bios:
    filename = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_medical_bios_inference_p{int(split)}.csv' if split != 0 else f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_medical_bios_inference.csv'
    out_df.to_csv(filename)
    if split > 1:
      filename_old = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_medical_bios_inference_p{int(split - 2)}.csv'
      if(os.path.exists(filename_old) and os.path.isfile(filename_old)):
        os.remove(filename_old)
  else:
    filename = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference_p{int(split)}.csv' if split != 0 else f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference.csv'
    out_df.to_csv(filename)
    if split > 1:
      filename_old = f'{FOLDERPATH}/{TOPIC_CC}/output/{TOPIC_SC}_export_inference_p{int(split - 2)}.csv'
      if(os.path.exists(filename_old) and os.path.isfile(filename_old)):
        os.remove(filename_old)

def get_meaningful_text(text):
  """
    Remove meaningless words like everything starting with '#', '@', or containing 'http'
  """
  # Define the regex pattern to match words starting with '@', '#', or 'http'
  # pattern = r'\b(@\w+|#\w+|http\S*)'
  pattern = r'\B@\w+|#\w+|http\S*'

  # Use re.sub to replace words starting with '@', '#', or 'http' with an empty string
  cleaned_text = re.sub(pattern, '', text)

  # Optional: Clean up any extra spaces created by the replacements
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

  return cleaned_text

def get_classification(df, check_bios=False):

  classifier = pipeline("text-classification", model=f"{HF_USERNAME}/{TASK}")
  outcomes = []
  tot_rows = list(map(lambda x, w, y, z: (x, w, y, z), df['user name'].values, df['user screen name'].values, df['content'].values, df['bio'].values ))
  for i, (account, user, content, bio) in enumerate(tot_rows):
    meaningful_content = get_meaningful_text(content)
    if len(meaningful_content) <= 500:
      if type(bio) != str:
        meaningful_bio = 'None'
      else:
        meaningful_bio = get_meaningful_text(bio)
      outcomes.append({"account": account, "user": user, "content": {**{"text": content}, **classifier(meaningful_content)[0]}, "bio": {**{"text": bio}, **classifier(meaningful_bio)[0]}})
    if i%1000 == 0 and i != 0:
      split = i/1000
      get_csv_file(outcomes, check_bios, split)
    if i == len(tot_rows) - 1:
      get_csv_file(outcomes, check_bios)

def classify_medical_bios(df):
  # List of common medical terms or keywords
  medical_terms = [
    'doctor', 'phd', 'professor', 'biotech',  'nurse', 'hospital', 'clinic', 'medicine', 'surgery', 'pharmacy',
    'treatment', 'diagnosis', 'vet', 'therapy', 'health', 'disease', 'condition', 'rheumatologist', 'md',
    'symptom', 'patient', 'medical', 'prescription', 'vaccine', 'infection', 'healtcare',
    'cardiology', 'dermatology', 'neurology', 'kidney', 'breast', 'oncology', 'pathology', 'radiology', 'cancer', 'lupus'
  ]

  # Create a regex pattern from the list of medical terms
  pattern = re.compile(r'\b(?:' + '|'.join(medical_terms) + r')\b', re.IGNORECASE)
  check_bio = []
  data = get_data(df)
  for bio in data['bios']:
    if type(bio) != str or bio == '':
      continue
    for word in bio.lower().split(' '):
      if contains_medical_reference(word, pattern):
        check_bio.append(bio)
  
  df = df.loc[df['bio'].isin(check_bio) == True]
  df.to_csv('medical_bios_df.csv')

# Function to check if a word contains any reference to the medical world
def contains_medical_reference(word, pattern):
  return bool(pattern.search(word))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ops', type=str, required=False)
    args = parser.parse_args()
    start_time = time.time()
    
    df = pd.read_excel(f'{FOLDERPATH}/{TOPIC_CC}/74982-all-2024-05-17-07-08-08-1163476.xlsx')
    df.drop_duplicates(subset=['content'], inplace=True)
    df.dropna(subset=['content'], inplace=True)

    if 'classification' in parser.parse_args().ops:
        get_classification(df)
    elif 'medical_bios' in parser.parse_args().ops:
        classify_medical_bios(df)

    print(f'execution terminated in {time.time() - start_time} seconds')
