from state import FloodPredictionState
from logger import structured_log
import shap
import numpy as np

class ExplainerAgent:
    def __init__(self, config):
        self.config = config

    def explain_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Generate SHAP explanations for the best model."""
        try:
            if state.best_model is None:
                raise ValueError("No best model available for explanation")
            
            explainer = shap.KernelExplainer(state.best_model.predict, state.X_train)
            shap_values = explainer.shap_values(state.X_test)
            
            state.feature_importance = dict(zip(state.X_test.columns, np.abs(shap_values).mean(axis=0)))
            structured_log('INFO', "Generated SHAP feature importance", features=state.feature_importance)
            
        except Exception as e:
            structured_log('ERROR', f"Error in model explanation: {str(e)}")
            raise
        return state