from fastapi.testclient import TestClient
from src.api.main import app
import pandas as pd

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_basic():
    # Load CSV to get realistic sequence
    df = pd.read_csv('data/processed/df.csv')
    sequence = df.drop(columns=['Close']).tail(30).values.tolist()  # Last 30 rows, excluding 'Close'
    payload = {"sequence": sequence}
    
    r = client.post('/predict', json=payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    
    body = r.json()
    assert 'prediction' in body, "Response missing 'prediction' key"
    assert isinstance(body['prediction'], float), "Prediction should be a float"
    assert 'details' in body, "Response missing 'details' key"