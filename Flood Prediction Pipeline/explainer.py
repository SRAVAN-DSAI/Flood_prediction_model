from state import FloodPredictionState
from logger import structured_log
import numpy as np

class ExplainerAgent:
    def __init__(self, config):
        self.config = config

    def explain_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Compute feature importance for the best model."""
        try:
            if state.best_model is None or state.X_train is None:
                raise ValueError("Best model or training data not available")
            
            model = state.best_model
            feature_names = state.X_train.columns
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                raise ValueError("Model does not support feature importance")
            
            state.feature_importance = dict(zip(feature_names, importance))
            structured_log('INFO', "Computed feature importance", features=state.feature_importance)
            
        except Exception as e:
            structured_log('ERROR', f"Error in explaining model: {str(e)}")
            raise
        return state