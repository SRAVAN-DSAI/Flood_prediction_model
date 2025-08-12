from state import FloodPredictionState
from logger import structured_log
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class ModelTunerAgent:
    def __init__(self, config):
        self.config = config

    def tune_best_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Perform hyperparameter tuning on the best model."""
        try:
            def model_score(model_info):
                return (model_info['r2'] * 0.7 - model_info['mse'] * 0.2 + (1 / (model_info['training_time'] + 1)) * 0.1)
            
            state.best_model_name = max(state.models, key=lambda x: model_score(state.models[x]))
            structured_log('INFO', f"Selected {state.best_model_name} for tuning", score=model_score(state.models[state.best_model_name]))
            
            if state.best_model_name == 'LinearRegression':
                state.best_model = state.models['LinearRegression']['model']
                structured_log('INFO', "Skipping tuning for LinearRegression")
                return state
            
            param_grid = self.config['model_params'][state.best_model_name]
            model_class = state.models[state.best_model_name]['model'].__class__
            grid_search = GridSearchCV(
                estimator=model_class(random_state=self.config['random_state']),
                param_grid=param_grid,
                cv=self.config['cv_folds'],
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(state.X_train, state.y_train)
            state.best_model = grid_search.best_estimator_
            structured_log('INFO', f"Best parameters for {state.best_model_name}", params=grid_search.best_params_)
            
            final_predictions = state.best_model.predict(state.X_test)
            final_r2 = r2_score(state.y_test, final_predictions)
            final_mse = mean_squared_error(state.y_test, final_predictions)
            structured_log('INFO', f"Tuned {state.best_model_name}", r2=final_r2, mse=final_mse)
            
        except Exception as e:
            structured_log('ERROR', f"Error in hyperparameter tuning: {str(e)}")
            raise
        return state