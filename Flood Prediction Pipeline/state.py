from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd

class FloodPredictionState(BaseModel):
    data_path: str
    df: Optional[pd.DataFrame] = None
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    models: Optional[Dict[str, Any]] = None
    best_model: Optional[Any] = None
    best_model_name: Optional[str] = None
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None
    feature_importance: Optional[Dict[str, float]] = None

    class Config:
        arbitrary_types_allowed = True