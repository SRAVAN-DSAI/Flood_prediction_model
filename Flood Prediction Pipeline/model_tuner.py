from state import FloodPredictionState
from logger import structured_log
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

class ModelTunerAgent:
    def __init__(self, config):
        self.config = config
        self.model_classes = {
            'LinearRegression': LinearRegression,
            'RandomForest': RandomForestRegressor,
            'XGBoost': xgb.XGBRegressor,
            'LightGBM': lgb.LGBMRegressor
        }

    def tune_best_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Tune the best model using parameters from config."""
        try:
            if not state.models:
                raise ValueError("No models available for tuning")
            
            # Select the best model based on R2 score
            best_model_name = max(state.models, key=lambda x: state.models[x]['r2'])
            model_class = self.model_classes[best_model_name]
            params = self.config['model_params'][best_model_name]
            
            # Train the model with default parameters
            best_model = model_class(**params)
            best_model.fit(state.X_train, state.y_train)
            
            state.best_model = best_model
            state.best_model_name = best_model_name
            structured_log('INFO', f"Selected and trained best model: {best_model_name}", params=params)
            
        except Exception as e:
            structured_log('ERROR', f"Error in model tuning: {str(e)}")
            raise
        return state