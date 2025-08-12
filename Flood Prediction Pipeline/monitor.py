from state import FloodPredictionState
from logger import structured_log
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

class MonitorAgent:
    def monitor_performance(self, state: FloodPredictionState) -> FloodPredictionState:
        """Monitor model performance."""
        try:
            predictions = state.best_model.predict(state.X_test)
            r2 = r2_score(state.y_test, predictions)
            mse = mean_squared_error(state.y_test, predictions)
            state.monitoring_metrics['timestamps'].append(datetime.now().isoformat())
            state.monitoring_metrics['r2'].append(r2)
            state.monitoring_metrics['mse'].append(mse)
            structured_log('INFO', "Updated monitoring metrics", r2=r2, mse=mse)
            
        except Exception as e:
            structured_log('ERROR', f"Error in monitoring: {str(e)}")
            raise
        return state