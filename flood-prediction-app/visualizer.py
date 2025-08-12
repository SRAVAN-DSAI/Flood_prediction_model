from state import FloodPredictionState
from logger import structured_log
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class VisualizerAgent:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']

    def visualize_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Create visualizations for model performance and feature importance."""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Feature importance plot
            if state.feature_importance:
                feature_importance = pd.DataFrame({
                    'Feature': list(state.feature_importance.keys()),
                    'Importance': list(state.feature_importance.values())
                }).sort_values(by='Importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                with pd.option_context('mode.use_inf_as_na', True):
                    sns.barplot(x='Importance', y='Feature', data=feature_importance)
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
                plt.close()
                structured_log('INFO', "Saved feature importance plot")
            
            # Model metrics plot
            if state.model_metrics:
                metrics_df = pd.DataFrame.from_dict(state.model_metrics, orient='index')
                plt.figure(figsize=(10, 6))
                metrics_df['r2'].plot(kind='bar')
                plt.title('Model R2 Scores')
                plt.ylabel('R2 Score')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'model_metrics.png'))
                plt.close()
                structured_log('INFO', "Saved model metrics plot")
            
        except Exception as e:
            structured_log('ERROR', f"Error in visualization: {str(e)}")
            raise
        return state