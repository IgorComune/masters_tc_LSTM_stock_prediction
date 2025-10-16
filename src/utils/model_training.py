"""
model_training.py

Script de treinamento para o LSTMModel fornecido.
Recebe CSV em original_data_path = '../data/processed/df.csv'

Assunções / To-do (variáveis para você ajustar se necessário):
- O CSV contém uma coluna chamada 'Close' (target).
- Todas as outras colunas são features (X).
- Você já tem o arquivo com a definição do LSTMModel e LSTMTrainer acessível via import: from lstm_model import LSTMModel, LSTMTrainer

O script faz:
- Leitura do CSV
- Escalonamento (StandardScaler para X, MinMaxScaler para y)
- Conversão para janelas temporais (sliding window)
- Split temporal train/val/test
- DataLoaders PyTorch
- Treinamento com LSTMTrainer (Early stopping + checkpoint)
- Salvamento do melhor checkpoint, do modelo final (usando save_model) e dos scalers (joblib)
- Gera um gráfico de curvas de loss (train/val) salvo em disk

Preencha as variáveis no bloco CONFIG abaixo conforme necessário.

"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os, sys

# adiciona a raiz do projeto ao sys.path para permitir imports "from src.models..."
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORT DO SEU MODELO (assume que você tenha salvo a classe em lstm_model.py)
try:
    from src.models.lstm_model import LSTMModel, LSTMTrainer
except Exception as e:
    raise ImportError("Não foi possível importar LSTMModel/LSTMTrainer. "
                      "Coloque o arquivo com as classes (o LSTMModel que você mostrou) no mesmo diretório ou no PYTHONPATH como 'lstm_model.py'.\n" 
                      f"Detalhe do erro: {e}")

# ==========================
# CONFIG - altere aqui
# ==========================
original_data_path = 'data/processed/df.csv'   # coloque o seu csv aqui
save_root = './models'                            # onde salvar checkpoints, scaler e modelo final
window_size = 30                                 # tamanho da sequência histórica (janela)
batch_size = 64
num_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-6
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
early_stopping_patience = 15
checkpoint_path = os.path.join(save_root, 'checkpoints', 'best_checkpoint.pth')
final_model_path = os.path.join(save_root, 'final_model')  # será salvo como final_model.pth e final_model_metadata.json
scaler_dir = os.path.join(save_root, 'scalers')
plot_path = os.path.join(save_root, 'training_loss.png')
random_seed = 42
use_cuda_if_available = True

# ==========================
# Helpers
# ==========================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    """Dataset para séries temporais com sliding window.
    X_scaled: np.array shape (n_samples, n_features)
    y_scaled: np.array shape (n_samples, 1)
    """
    def __init__(self, X_scaled, y_scaled, window_size):
        self.X = X_scaled
        self.y = y_scaled
        self.window = window_size
        self.length = len(self.X) - self.window
        if self.length <= 0:
            raise ValueError("window_size é maior ou igual ao numero de samples disponíveis")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window]
        y_target = self.y[idx + self.window]  # target é o valor imediatamente após a janela
        return torch.FloatTensor(x_seq), torch.FloatTensor(y_target)


def prepare_data(df, window_size, train_ratio, val_ratio, test_ratio, scaler_x=None, scaler_y=None):
    """Recebe dataframe, retorna dataloaders, scalers e shapes."""
    # Garantir ordenação temporal
    df = df.copy()

    if 'Close' not in df.columns:
        raise ValueError("Coluna 'Close' não encontrada no csv")

    X_df = df.drop(columns=['Close'])
    y_df = df[['Close']]

    # Convert to numpy
    X = X_df.values.astype(float)
    y = y_df.values.astype(float)

    # Fit scalers if não fornecidos
    if scaler_x is None:
        scaler_x = StandardScaler()
        scaler_x.fit(X)
    X_scaled = scaler_x.transform(X)

    if scaler_y is None:
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.fit(y)
    y_scaled = scaler_y.transform(y)

    n_samples = len(X_scaled)
    if n_samples - window_size <= 0:
        raise ValueError("Dados insuficientes para o window_size escolhido")

    # índices para o split temporal
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Garantir que cada split tenha pelo menos window_size+1 samples
    train_end = max(train_end, window_size + 1)
    val_end = max(val_end, train_end + window_size + 1)
    val_end = min(val_end, n_samples - 1)

    # Criar datasets usando as janelas, porém com slices para cada conjunto
    X_train = X_scaled[:train_end]
    y_train = y_scaled[:train_end]

    X_val = X_scaled[train_end:val_end]
    y_val = y_scaled[train_end:val_end]

    X_test = X_scaled[val_end:]
    y_test = y_scaled[val_end:]

    train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, window_size)

    return train_dataset, val_dataset, test_dataset, scaler_x, scaler_y


def build_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


# ==========================
# Main
# ==========================

def main(
    original_data_path=original_data_path,
    window_size=window_size,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    early_stopping_patience=early_stopping_patience,
    checkpoint_path=checkpoint_path,
    final_model_path=final_model_path,
    scaler_dir=scaler_dir,
    plot_path=plot_path,
    random_seed=random_seed,
    use_cuda_if_available=use_cuda_if_available
):

    set_seed(random_seed)

    device = 'cuda' if (torch.cuda.is_available() and use_cuda_if_available) else 'cpu'
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    # --- read data
    print(f"Lendo CSV: {original_data_path}")
    df = pd.read_csv(original_data_path)
    print(f"CSV carregado. Tamanho: {df.shape}")

    # --- prepare datasets
    train_dataset, val_dataset, test_dataset, scaler_x, scaler_y = prepare_data(
        df, window_size, train_ratio, val_ratio, test_ratio
    )

    input_size = df.drop(columns=['Close']).shape[1]

    train_loader, val_loader, test_loader = build_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # --- model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=30,
        num_layers=1,
        output_size=1,
        dropout=0.35
    )
    print(f"Modelo instanciado. Parâmetros treináveis: {model.count_parameters()}")

    trainer = LSTMTrainer(model, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path
    )

    # --- load best checkpoint and save final model + metadata
    # carregar o checkpoint salvo pelo trainer
    print("Carregando melhor checkpoint salvo e salvando modelo final com metadata...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    metadata = {
        'original_data_path': original_data_path,
        'window_size': window_size,
        'batch_size': batch_size,
        'num_epochs_ran': len(history['train_losses']),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    model.save_model(final_model_path, metadata=metadata)

    # --- salvar scalers
    scaler_x_path = os.path.join(scaler_dir, 'scaler_x.joblib')
    scaler_y_path = os.path.join(scaler_dir, 'scaler_y.joblib')
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"✓ Scalers salvos: {scaler_x_path} , {scaler_y_path}")

    # --- salvar histórico e plot
    history_path = os.path.join(save_root, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ Histórico salvo: {history_path}")

    # plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_losses'], label='train')
    plt.plot(history['val_losses'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Train x Val Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"✓ Gráfico de loss salvo em: {plot_path}")

    print('\nTreinamento concluído. Artefatos salvos em:')
    print(f" - checkpoint: {checkpoint_path}")
    print(f" - modelo final: {final_model_path}.pth")
    print(f" - scalers: {scaler_x_path}, {scaler_y_path}")
    print(f" - history: {history_path}")
    print(f" - loss plot: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar LSTMModel com CSV de preços')
    parser.add_argument('--data', type=str, default=original_data_path, help='Caminho para o CSV processado')
    parser.add_argument('--device', type=str, default=None, help='cpu ou cuda')
    args = parser.parse_args()

    if args.device:
        use_cuda_if_available = (args.device.lower() == 'cuda')

    main(original_data_path=args.data)
