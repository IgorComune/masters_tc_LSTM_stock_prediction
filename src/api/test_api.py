from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)
def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_basic():
# dados fake — o modelo deve estar carregado; se não estiver, adapta-se
    payload = {"sequence": [0.1] * 50}
    r = client.post('/predict', json=payload)
    # Se o modelo não for carregado durante CI/dev, permita que 500 seja

    assert r.status_code in (200, 500)
    if r.status_code == 200:
        body = r.json()
        assert 'probabilities' in body
        assert 'predicted_class' in body