import pandas as pd
import requests
from typing import Optional, Dict, Any
import numpy as np


class SequenceSamplerSimple:
    """
    Versão enxuta: carrega CSV, amostra 30 timesteps (contíguos por padrão),
    envia para a API e retorna {'sequence', 'close_values', 'payload', 'prediction'}.

    Parâmetros:
      csv_path: caminho do CSV
      url: endpoint da API
      window_size: número de timesteps (default 30)
      seed: int ou None (reprodutibilidade)
      mode: 'contiguous' (padrão) ou 'random' (linhas aleatórias sem reposição)
    """
    def __init__(
        self,
        csv_path: str = '../data/processed/df.csv',
        url: str = 'http://127.0.0.1:8000/predict',
        window_size: int = 30,
        seed: Optional[int] = None,
        mode: str = 'contiguous'
    ):
        self.csv_path = csv_path
        self.url = url
        self.window_size = int(window_size)
        self.rng = np.random.default_rng(seed)
        if mode not in ('contiguous', 'random'):
            raise ValueError("mode deve ser 'contiguous' ou 'random'")
        self.mode = mode

    def _load_and_prep(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path).reset_index(drop=True)
        required = {'Open', 'High', 'Low', 'Volume', 'Close'}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"Colunas faltando no CSV: {missing}")
        if len(df) < self.window_size:
            raise ValueError(f"DataFrame tem {len(df)} linhas < window_size ({self.window_size})")
        return df

    def sample_and_send(self, dry_run: bool = False, timeout: int = 5) -> Dict[str, Any]:
        """
        Retorna:
          {
            'sequence': [...],
            'close_values': [...],
            'payload': {...},
            'prediction': <json ou texto ou None se dry_run>
          }
        """
        df = self._load_and_prep()
        n = len(df)

        if self.mode == 'contiguous':
            start = int(self.rng.integers(0, n - self.window_size + 1))
            window = df.iloc[start:start + self.window_size].copy()
        else:  # random
            idx = self.rng.choice(n, size=self.window_size, replace=False)
            idx.sort()
            window = df.iloc[idx].copy()

        # ✅ Somente as features usadas no treinamento
        feature_cols = ['Open', 'High', 'Low', 'Volume']
        sequence = window[feature_cols].values.tolist()
        close_values = window['Close'].values.tolist()
        payload = {"sequence": sequence}

        # print(f"Mode: {self.mode}")
        # print(f"Sequence length: {len(sequence)} | features/timestep: {len(sequence[0]) if sequence else 0}")

        if dry_run:
            return {'sequence': sequence, 'close_values': close_values, 'payload': payload, 'prediction': None}

        try:
            resp = requests.post(self.url, json=payload, timeout=timeout)
            resp.raise_for_status()
            try:
                pred = resp.json()
            except ValueError:
                pred = resp.text
        except requests.exceptions.RequestException as e:
            return {
                'sequence': sequence,
                'close_values': close_values,
                'payload': payload,
                'prediction': None,
                'error': str(e)
            }

        return {'sequence': sequence, 'close_values': close_values, 'payload': payload, 'prediction': pred}


sampler = SequenceSamplerSimple(
    csv_path='data/processed/df.csv',
    url='http://127.0.0.1:8000/predict',
    window_size=30,
    seed=42,
    mode='contiguous'
)

# apenas checar o payload
resultado = sampler.sample_and_send(dry_run=True)
# enviar de verdade
resultado_real = sampler.sample_and_send(dry_run=False)

print(resultado_real['prediction']['prediction'])