from state import FloodPredictionState
from logger import structured_log
import joblib
import os

class ModelSaverAgent:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']

    def save_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Save the best model to the output directory."""
        try:
            if state.best_model is None or state.best_model_name is None:
                raise ValueError("No best model available to save")
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            model_path = os.path.join(self.output_dir, f"{state.best_model_name}.joblib")
            joblib.dump(state.best_model, model_path)
            structured_log('INFO', f"Saved best model {state.best_model_name} to {model_path}")
            
        except Exception as e:
            structured_log('ERROR', f"Error saving model: {str(e)}")
            raise
        return state