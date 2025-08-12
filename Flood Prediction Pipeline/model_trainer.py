from state import FloodPredictionState
from logger import structured_log
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime

class ModelTrainerAgent:
    def __init__(self, config):
        self.config = config

    def train_models(self, state: FloodPredictionState) -> FloodPredictionState:
        """Train multiple models with cross-validation."""
        try:
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(random_state=self.config['random_state']),
                'XGBoost': xgb.XGBRegressor(random_state=self.config['random_state']),
                'LightGBM': lgb.LGBMRegressor(random_state=self.config['random_state'])
            }
            
            for name, model in models.items():
                start_time = datetime.now()
                model.fit(state.X_train, state.y_train)
                predictions = model.predict(state.X_test)
                
                cv_scores = cross_val_score(model, state.X_train, state.y_train, cv=self.config['cv_folds'], scoring='r2')
                
                state.models[name] = {
                    'model': model,
                    'r2': r2_score(state.y_test, predictions),
                    'mse': mean_squared_error(state.y_test, predictions),
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
                structured_log('INFO', f"{name} trained",
                             r2=state.models[name]['r2'],
                             mse=state.models[name]['mse'],
                             cv_r2_mean=state.models[name]['cv_r2_mean'],
                             cv_r2_std=state.models[name]['cv_r2_std'],
                             training_time=state.models[name]['training_time'])
            
        except Exception as e:
            structured_log('ERROR', f"Error training models: {str(e)}")
            raise
        return state