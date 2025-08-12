from state import FloodPredictionState
from logger import structured_log

class MonitorAgent:
    def monitor_performance(self, state: FloodPredictionState) -> FloodPredictionState:
        """Monitor model performance metrics."""
        try:
            if not state.model_metrics:
                raise ValueError("No model metrics available")
            
            for model_name, metrics in state.model_metrics.items():
                structured_log('INFO', f"Monitoring {model_name}", r2=metrics['r2'], mse=metrics['mse'])
                
            best_r2 = state.model_metrics[state.best_model_name]['r2']
            if best_r2 < 0.5:
                structured_log('WARNING', f"Best model {state.best_model_name} has low R2 score: {best_r2}")
            
        except Exception as e:
            structured_log('ERROR', f"Error in monitoring: {str(e)}")
            raise
        return state