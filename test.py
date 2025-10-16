import requests
import json
import pandas as pd

url = "http://127.0.0.1:8000/predict"

# Load CSV and get last 30 rows (excluding 'Close') for sequence
df = pd.read_csv('data/processed/df.csv')
sequence = df.drop(columns=['Date','Close']).tail(30).values.tolist()  # 30 timesteps, input_size features
payload = {"sequence": sequence}

try:
    response = requests.post(url, json=payload, timeout=5)
    print(f"Status code: {response.status_code}")
    print("Response:", json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"Erro na requisição: {e}")