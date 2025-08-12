from langchain_core.pydantic_v1 import BaseModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, Any, Optional

class FloodPredictionState(BaseModel):
    data_path: str
    df: Optional[pd.DataFrame] = None
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    scaler: Optional[StandardScaler] = None
    models: Dict[str, Dict[str, Any]] = {}
    best_model: Any = None
    best_model_name: Optional[str] = None
    saved_model_path: Optional[str] = None
    sample_prediction: Optional[float] = None
    feature_importance: Optional[Dict] = None
    visualizations: Dict[str, str] = {}
    monitoring_metrics: Dict[str, list] = {'timestamps': [], 'r2': [], 'mse': []}