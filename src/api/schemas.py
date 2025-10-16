from pydantic import BaseModel
from typing import List, Dict, Any

class PredictRequest(BaseModel):
    sequence: List[List[float]]  # 2D sequence: list of lists with input_size elements each

class PredictResponse(BaseModel):
    prediction: float  # Scalar prediction for regression
    details: Dict[str, Any]  # Optional details