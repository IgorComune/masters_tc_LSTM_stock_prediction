"""
model_evaluation.py

Class-based evaluation script for LSTMModel on test set.
Performs predictions, descaling, metric calculation, and saves plots to ./results.
"""
import sys
import os
from typing import Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs('./results', exist_ok=True)

class LSTMEvaluator:
    def __init__(self, model_path: str, scaler_dir: str, window_size: int = 30, device: str = 'cpu'):
        self.model_path = model_path
        self.scaler_dir = scaler_dir
        self.window_size = window_size
        self.device = device
        self.model = self.load_model(model_path, device)
        self.scaler_x, self.scaler_y = self.load_scalers(scaler_dir)

    @staticmethod
    def load_test_data(csv_path: str, feature_columns: list, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        X = df[feature_columns].values.astype(float)
        y = df[[target_column]].values.astype(float)
        return X, y

    @staticmethod
    def load_scalers(scaler_dir: str) -> Tuple[object, object]:
        scaler_x = joblib.load(os.path.join(scaler_dir, 'scaler_x.joblib'))
        scaler_y = joblib.load(os.path.join(scaler_dir, 'scaler_y.joblib'))
        return scaler_x, scaler_y

    @staticmethod
    def load_model(model_path: str, device: str = 'cpu') -> torch.nn.Module:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
            from src.models.lstm_model import LSTMModel
            model = LSTMModel.load_model(model_path, device=device)
            return model

    def prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X) - self.window_size
        sequences = np.array([X[i:i+self.window_size] for i in range(n_samples)])
        return sequences

    @staticmethod
    def predict(model: torch.nn.Module, X_seq: np.ndarray, device: str = 'cpu') -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(device)
            preds = model(X_tensor).cpu().numpy()
        return preds

    @staticmethod
    def inverse_transform(y_scaled: np.ndarray, scaler) -> np.ndarray:
        return scaler.inverse_transform(y_scaled)

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    @staticmethod
    def plot_real_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Real')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Close Price')
        plt.title('Real vs Predicted')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 5))
        plt.plot(residuals, label='Residuals')
        plt.xlabel('Sample')
        plt.ylabel('Residual')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
        errors = y_true - y_pred
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=30, edgecolor='k')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def run(self, test_csv_path: str, target_column: str = 'Close') -> dict:
        df = pd.read_csv(test_csv_path)
        feature_columns = [col for col in df.columns if col != target_column]
        X_raw, y_raw = self.load_test_data(test_csv_path, feature_columns, target_column)

        X_scaled = self.scaler_x.transform(X_raw)
        y_scaled = self.scaler_y.transform(y_raw)

        X_seq = self.prepare_sequences(X_scaled)
        y_true_seq = y_scaled[self.window_size:]

        y_pred_scaled = self.predict(self.model, X_seq, device=self.device)
        y_pred = self.inverse_transform(y_pred_scaled, self.scaler_y)
        y_true = self.inverse_transform(y_true_seq, self.scaler_y)

        metrics = self.calculate_metrics(y_true, y_pred)

        self.plot_real_vs_pred(y_true, y_pred, './results/real_vs_pred.png')
        self.plot_residuals(y_true, y_pred, './results/residuals.png')
        self.plot_error_distribution(y_true, y_pred, './results/error_distribution.png')

        return metrics

# --------------------------
# Example usage (commented)
# --------------------------
evaluator = LSTMEvaluator(
    model_path='./models/checkpoints/best_checkpoint.pth',
    scaler_dir='./models/scalers',
    window_size=30,
    device='cuda'
)
metrics = evaluator.run(test_csv_path='data/processed/df.csv')
print(metrics)
