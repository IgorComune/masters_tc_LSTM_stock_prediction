from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    # sequência de entrada — lista de floats (ajuste se for batch)
    sequence: List[float]

class PredictResponse(BaseModel):
    probabilities: List[float]
    predicted_class: Optional[int]
    details: Optional[dict]
