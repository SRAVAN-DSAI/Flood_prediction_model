from state import FloodPredictionState
from logger import structured_log
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class VisualizerAgent:
    def __init__(self, config):
        self.config = config

    def visualize_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Generate additional visualizations."""
        try:
            # Correlation heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(state.X.corr(), annot=True, cmap='coolwarm')
            corr_path = f"{self.config['output_dir']}/correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(corr_path)
            plt.close()
            state.visualizations['correlation_heatmap'] = corr_path
            
            # Prediction distribution
            predictions = state.best_model.predict(state.X_test)
            plt.figure(figsize=(10, 6))
            sns.histplot(predictions, kde=True)
            plt.title('Prediction Distribution')
            pred_dist_path = f"{self.config['output_dir']}/prediction_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(pred_dist_path)
            plt.close()
            state.visualizations['prediction_distribution'] = pred_dist_path
            
            structured_log('INFO', "Generated visualizations", visualizations=list(state.visualizations.keys()))
            
        except Exception as e:
            structured_log('ERROR', f"Error in visualization: {str(e)}")
            raise
        return state