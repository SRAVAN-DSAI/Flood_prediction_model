from state import FloodPredictionState
from logger import structured_log
import joblib
import os
from datetime import datetime

class ModelSaverAgent:
    def __init__(self, config):
        self.config = config

    def save_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Save the best model and scaler."""
        try:
            if not os.path.exists(self.config['output_dir']):
                os.makedirs(self.config['output_dir'])
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            state.saved_model_path = f"{self.config['output_dir']}/{state.best_model_name}_flood_prediction_{timestamp}.joblib"
            joblib.dump({'model': state.best_model, 'scaler': state.scaler}, state.saved_model_path)
            structured_log('INFO', f"Model and scaler saved to {state.saved_model_path}")
            
        except Exception as e:
            structured_log('ERROR', f"Error saving model: {str(e)}")
            raise
        return state