import torch
import numpy as np
from joblib import load as joblib_load
from typing import List
from pathlib import Path
from src.models.lstm_model import LSTMModel


MODEL_PATH = Path("models/saved_models/lstm_model.pth")
SCALER_PATH = Path("models/saved_models/scaler.pkl")


class Predictor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None

    def load(self, model_path: str | Path = None, scaler_path: str | Path = None):
        """Carrega o modelo e o scaler"""
        model_path = Path(model_path or MODEL_PATH)
        scaler_path = Path(scaler_path or SCALER_PATH)

        try:
            # Usa o método nativo da classe para carregar o modelo
            self.model = LSTMModel.load_model(model_path, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")

        try:
            self.scaler = joblib_load(scaler_path)
        except Exception:
            self.scaler = None

    def preprocess(self, seq: List[float]) -> torch.Tensor:
        """Transforma sequência para tensor adequado"""
        arr = np.array(seq, dtype=np.float32)

        # Se o scaler existir e espera 1 feature
        if self.scaler is not None:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = self.scaler.transform(arr)
        
        # Formato (1, seq_len, 1)
        tensor = torch.from_numpy(arr).float().unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, seq: List[float]) -> dict:
        """Realiza predição numérica com o modelo LSTM"""
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Chame .load() primeiro.")

        x = self.preprocess(seq)
        pred = self.model.predict(x)  # usa método nativo da LSTMModel
        pred = pred.flatten()

        # Desfaz scaling se necessário
        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        return {"prediction": float(pred[-1])}
