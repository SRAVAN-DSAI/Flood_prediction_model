from state import FloodPredictionState
from logger import structured_log

class ModelTunerAgent:
    def __init__(self, config):
        self.config = config

    def tune_best_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Select the best model from trained models based on R2 score."""
        try:
            if not state.models or not state.model_metrics:
                raise ValueError("No models or metrics available for tuning")
            
            # Find the model with the highest R2 score
            best_model_name = max(
                state.model_metrics,
                key=lambda x: state.model_metrics[x]['r2']
            )
            
            state.best_model = state.models[best_model_name]
            state.best_model_name = best_model_name
            structured_log('INFO', f"Selected best model: {best_model_name}", r2=state.model_metrics[best_model_name]['r2'])
            
        except Exception as e:
            structured_log('ERROR', f"Error in model tuning: {str(e)}")
            raise
        return state