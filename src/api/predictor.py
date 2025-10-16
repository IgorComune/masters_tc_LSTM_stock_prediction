from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import torch

# Import your model class (keep relative to your project layout)
# from src.models.lstm_model import LSTMModel


class Predictor:
    """
    Robust predictor for LSTMModel.

    - Loads model via LSTMModel.load_model(...)
    - Loads scalers: prefer scaler_x.joblib & scaler_y.joblib in scaler_dir,
      otherwise accepts a single scaler_path (treated as target scaler).
    - Preprocesses input sequence (validates window_size and input_size).
    - Returns a final scalar prediction and optional full prediction series.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        scaler_dir: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        window_size: Optional[int] = None,
        input_size: Optional[int] = 1,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_dir = Path(scaler_dir) if scaler_dir is not None else None
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None
        self.window_size = window_size
        self.input_size = int(input_size) if input_size is not None else 1
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler_x = None  # scaler for features (X)
        self.scaler_y = None  # scaler for target (y)

    def load(self) -> None:
        """Load model and scalers. Raises RuntimeError on failure."""
        # --- load model (use LSTMModel.load_model)
        try:
            # lazy import to avoid path issues if script run from different cwd
            from src.models.lstm_model import LSTMModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Cannot import LSTMModel from src.models.lstm_model: {e}")

        try:
            self.model = LSTMModel.load_model(str(self.model_path), device=self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading model from '{self.model_path}': {e}")

        # --- load scalers
        # Priority:
        # 1) scaler_dir with scaler_x.joblib and scaler_y.joblib
        # 2) scaler_path single file (assumed to be target scaler_y)
        if self.scaler_dir and self.scaler_dir.exists():
            sx = self.scaler_dir / "scaler_x.joblib"
            sy = self.scaler_dir / "scaler_y.joblib"
            if sx.exists():
                try:
                    self.scaler_x = joblib.load(str(sx))
                except Exception as e:
                    raise RuntimeError(f"Failed loading scaler_x from {sx}: {e}")
            if sy.exists():
                try:
                    self.scaler_y = joblib.load(str(sy))
                except Exception as e:
                    raise RuntimeError(f"Failed loading scaler_y from {sy}: {e}")

        # fallback: single scaler file
        if self.scaler_x is None and self.scaler_path:
            sp = Path(self.scaler_path)
            if sp.exists():
                try:
                    loaded = joblib.load(str(sp))
                    # Heuristics: if scaler has attribute n_features_in_ equal to 1 -> could be X or y
                    # We'll assume single scaler is target scaler (scaler_y), which is the more common case
                    self.scaler_y = loaded
                except Exception as e:
                    raise RuntimeError(f"Failed loading scaler from {sp}: {e}")

        # If we have scaler_x but input_size wasn't provided, try to infer
        if self.scaler_x is not None and hasattr(self.scaler_x, "n_features_in_"):
            try:
                inferred = int(getattr(self.scaler_x, "n_features_in_"))
                # only override if user didn't set input_size
                if self.input_size is None:
                    self.input_size = inferred
            except Exception:
                pass

        # If model exists, ensure it's on device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _validate_and_format_sequence(
        self, seq: Union[Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """
        Validate input sequence and return numpy array shape (seq_len, input_size).

        Rules:
        - Accept 1D sequence of length L when input_size == 1 -> returns (L,1)
        - Accept 2D sequence shape (L, input_size) -> returns same
        - If length > window_size -> take last window_size steps
        - If length < window_size -> raises ValueError
        """
        arr = np.asarray(seq, dtype=np.float32)

        # If 1D and model expects one feature, reshape
        if arr.ndim == 1:
            if self.input_size != 1:
                raise ValueError(
                    f"Received 1D sequence but model expects input_size={self.input_size}. "
                    "Provide 2D sequence with shape (seq_len, input_size)."
                )
            arr = arr.reshape(-1, 1)  # (L,1)

        # If 2D, ensure second dim matches input_size
        if arr.ndim == 2:
            if arr.shape[1] != self.input_size:
                raise ValueError(
                    f"Sequence second dimension ({arr.shape[1]}) does not match "
                    f"model input_size ({self.input_size})."
                )
        else:
            raise ValueError("Sequence must be 1D or 2D array-like.")

        seq_len = arr.shape[0]
        if self.window_size is not None:
            if seq_len < self.window_size:
                raise ValueError(
                    f"Sequence length ({seq_len}) smaller than required window_size ({self.window_size})."
                )
            if seq_len > self.window_size:
                # use last window_size values
                arr = arr[-self.window_size :, :]

        return arr  # shape (window_size, input_size) or (seq_len, input_size)

    def _apply_feature_scaler(self, arr: np.ndarray) -> np.ndarray:
        """Apply scaler_x to features if present, otherwise return arr unchanged."""
        if self.scaler_x is None:
            return arr
        # scaler.transform expects shape (n_samples, n_features)
        try:
            return self.scaler_x.transform(arr)
        except Exception as e:
            raise RuntimeError(f"Error applying feature scaler: {e}")

    def _tensor_from_arr(self, arr: np.ndarray) -> torch.Tensor:
        """Return tensor shaped (1, seq_len, input_size) on correct device."""
        tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, seq_len, input_size)
        return tensor.to(self.device)

    def predict(
        self,
        seq: Union[Sequence[float], np.ndarray],
        return_series: bool = False,
    ) -> dict:
        """
        Predict using the LSTM model.

        Returns:
            dict with keys:
              - 'prediction': float (last step predicted, inverse-transformed)
              - 'prediction_series': np.ndarray of shape (n_steps, ) if return_series True
              - 'raw_output': raw model output array (before inverse_transform) if scaler_y missing
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() before .predict().")

        arr = self._validate_and_format_sequence(seq)  # (window_size, input_size)
        arr_scaled = self._apply_feature_scaler(arr)  # same shape

        x_tensor = self._tensor_from_arr(arr_scaled)  # (1, seq_len, input_size)

        # Model forward
        with torch.no_grad():
            out = self.model(x_tensor)  # expected shape (batch=1, output_size)
            if isinstance(out, torch.Tensor):
                out_np = out.cpu().numpy().reshape(-1, 1)  # (1,1) or (n_steps,1) depending on model
            else:
                # fallback if model.predict returns numpy
                out_np = np.asarray(out).reshape(-1, 1)

        # If scaler_y exists, inverse transform; else return raw
        if self.scaler_y is not None:
            try:
                y_inv = self.scaler_y.inverse_transform(out_np).flatten()
            except Exception as e:
                raise RuntimeError(f"Error inverse transforming model output with scaler_y: {e}")
            result_value = float(y_inv[-1])
            result_series = y_inv if return_series else None
            return {"prediction": result_value, "prediction_series": result_series}
        else:
            # No target scaler available; return raw outputs
            raw = out_np.flatten()
            result_value = float(raw[-1])
            result_series = raw if return_series else None
            return {"prediction": result_value, "prediction_series": result_series, "raw_output": raw}

    # convenience method to load and predict in one call
    def load_and_predict(
        self,
        seq: Union[Sequence[float], np.ndarray],
        model_path: Optional[Union[str, Path]] = None,
        scaler_dir: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        return_series: bool = False,
    ) -> dict:
        """Helper: optionally override paths and run load() + predict()."""
        if model_path is not None:
            self.model_path = Path(model_path)
        if scaler_dir is not None:
            self.scaler_dir = Path(scaler_dir)
        if scaler_path is not None:
            self.scaler_path = Path(scaler_path)

        self.load()
        return self.predict(seq, return_series=return_series)
