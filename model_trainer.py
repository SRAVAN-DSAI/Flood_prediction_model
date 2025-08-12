from state import FloodPredictionState
from logger import structured_log
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

class ModelTrainerAgent:
    def __init__(self, config):
        self.config = config
        self.models = {
            'RandomForest': RandomForestRegressor,
            'XGBoost': XGBRegressor,
            'LightGBM': LGBMRegressor
        }

    def train_models(self, state: FloodPredictionState) -> FloodPredictionState:
        """Train multiple models and store performance metrics and models."""
        try:
            if state.X_train is None or state.y_train is None or state.X_test is None or state.y_test is None:
                raise ValueError("Training or test data is not available")
            
            state.model_metrics = {}
            state.models = {}
            
            for model_name, model_class in self.models.items():
                # Initialize model with parameters from config
                params = self.config['model_params'].get(model_name, {})
                model = model_class(**params)
                
                # Train model
                structured_log('INFO', f"Training {model_name}")
                model.fit(state.X_train, state.y_train)
                
                # Store model
                state.models[model_name] = model
                
                # Evaluate model
                y_pred = model.predict(state.X_test)
                r2 = r2_score(state.y_test, y_pred)
                mse = mean_squared_error(state.y_test, y_pred)
                
                # Store metrics
                state.model_metrics[model_name] = {'r2': r2, 'mse': mse}
                structured_log('INFO', f"{model_name} metrics", r2=r2, mse=mse)
                
                # Update best model if this is the first model or has better R2
                if state.best_model is None or r2 > state.model_metrics[state.best_model_name]['r2']:
                    state.best_model = model
                    state.best_model_name = model_name
                    structured_log('INFO', f"New best model: {model_name}", r2=r2)
            
        except Exception as e:
            structured_log('ERROR', f"Error in model training: {str(e)}")
            raise
        return state