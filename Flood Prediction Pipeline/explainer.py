from state import FloodPredictionState
from logger import structured_log
import numpy as np

class ExplainerAgent:
    def __init__(self, config):
        self.config = config

    def explain_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Generate feature importance for the best model without SHAP."""
        try:
            if state.best_model is None:
                raise ValueError("No best model available for explanation")
            
            model_name = state.best_model_name
            model = state.best_model
            feature_names = state.X_train.columns
            
            # Compute feature importance based on model type
            if model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                # Tree-based models provide feature_importances_
                importances = model.feature_importances_
                structured_log('INFO', f"Using feature_importances_ for {model_name}")
            elif model_name == 'LinearRegression':
                # LinearRegression provides coef_
                importances = np.abs(model.coef_)
                structured_log('INFO', f"Using absolute coefficients for {model_name}")
            else:
                raise ValueError(f"Unsupported model type for feature importance: {model_name}")
            
            # Store feature importance in state
            state.feature_importance = dict(zip(feature_names, importances))
            structured_log('INFO', "Generated feature importance", features=state.feature_importance)
            
        except Exception as e:
            structured_log('ERROR', f"Error in model explanation: {str(e)}")
            raise
        return state