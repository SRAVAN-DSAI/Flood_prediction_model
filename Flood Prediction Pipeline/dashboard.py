from state import FloodPredictionState
from logger import structured_log
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class DashboardAgent:
    def __init__(self, config):
        self.config = config
        self.app = None  # Gradio interface will be set up in setup_dashboard

    def setup_dashboard(self, state: FloodPredictionState) -> FloodPredictionState:
        """Set up and start the Gradio dashboard."""
        try:
            # Convert state data to DataFrame for visualization
            metrics_df = pd.DataFrame.from_dict(state.model_metrics, orient='index')
            
            # Feature importance plot
            feature_importance = pd.DataFrame({
                'Feature': list(state.feature_importance.keys()),
                'Importance': list(state.feature_importance.values())
            }).sort_values(by='Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                title='Feature Importance',
                orientation='h'
            )
            
            # Prediction distribution
            with pd.option_context('mode.use_inf_as_na', True):
                fig_dist = px.histogram(
                    x=state.y_test,
                    nbins=30,
                    title='Prediction Distribution',
                    labels={'x': 'Flood Probability'}
                )
            
            # Correlation heatmap
            with pd.option_context('mode.use_inf_as_na', True):
                corr_matrix = state.X_test.corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='Viridis'
                ))
                fig_corr.update_layout(title='Feature Correlation Heatmap')
            
            # Model performance table
            fig_table = go.Figure(data=[go.Table(
                header=dict(values=['Model', 'R2 Score', 'MSE']),
                cells=dict(values=[
                    metrics_df.index,
                    metrics_df['r2'].round(4),
                    metrics_df['mse'].round(4)
                ])
            )])
            fig_table.update_layout(title='Model Performance Metrics')
            
            # Define prediction function
            def make_prediction(*input_values):
                try:
                    input_data = pd.DataFrame([input_values], columns=state.X_test.columns)
                    prediction = state.best_model.predict(input_data)[0]
                    return f"Predicted Flood Probability: {prediction:.4f}"
                except Exception as e:
                    return f"Error in prediction: {str(e)}"
            
            # Create Gradio interface using Blocks
            with gr.Blocks() as self.app:
                gr.Markdown("# Flood Prediction Dashboard")
                
                # Model performance
                gr.Markdown("## Model Performance")
                gr.Plot(fig_table, label="Model Performance Metrics")
                
                # Feature importance
                gr.Markdown("## Feature Importance")
                gr.Plot(fig_importance, label="Feature Importance")
                
                # Prediction distribution
                gr.Markdown("## Prediction Distribution")
                gr.Plot(fig_dist, label="Prediction Distribution")
                
                # Correlation heatmap
                gr.Markdown("## Feature Correlation")
                gr.Plot(fig_corr, label="Feature Correlation Heatmap")
                
                # Prediction form
                gr.Markdown("## Make a Prediction")
                inputs = [gr.Number(label=col, value=0) for col in state.X_test.columns]
                predict_button = gr.Button("Predict")
                output = gr.Textbox(label="Prediction Result")
                
                # Link button to prediction function
                predict_button.click(
                    fn=make_prediction,
                    inputs=inputs,
                    outputs=output
                )
            
            # Launch Gradio interface
            try:
                self.app.launch(share=True, quiet=True)  # Attempt public URL in Kaggle
                structured_log('INFO', "Gradio dashboard launched with public URL")
            except Exception as e:
                structured_log('INFO', f"Gradio dashboard launch with share=True failed: {str(e)}. Try running locally with share=False.")
                self.app.launch(share=False, quiet=True)  # Fallback to local
                structured_log('INFO', f"Gradio dashboard running locally at http://localhost:7860")
            
        except Exception as e:
            structured_log('ERROR', f"Error setting up dashboard: {str(e)}")
            raise
        return state