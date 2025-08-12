from state import FloodPredictionState
from logger import structured_log
import shap
import matplotlib.pyplot as plt
from datetime import datetime

class ExplainerAgent:
    def __init__(self, config):
        self.config = config

    def explain_model(self, state: FloodPredictionState) -> FloodPredictionState:
        """Generate SHAP explanations."""
        try:
            explainer = shap.TreeExplainer(state.best_model) if state.best_model_name in ['RandomForest', 'XGBoost', 'LightGBM'] else shap.LinearExplainer(state.best_model, state.X_train)
            shap_values = explainer.shap_values(state.X_test)
            
            state.feature_importance = dict(zip(state.X_test.columns, np.abs(shap_values).mean(axis=0)))
            structured_log('INFO', "Generated SHAP feature importance", features=state.feature_importance)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, state.X_test, show=False)
            shap_path = f"{self.config['output_dir']}/shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(shap_path)
            plt.close()
            state.visualizations['shap_summary'] = shap_path
            
        except Exception as e:
            structured_log('ERROR', f"Error in SHAP explanation: {str(e)}")
            raise
        return state