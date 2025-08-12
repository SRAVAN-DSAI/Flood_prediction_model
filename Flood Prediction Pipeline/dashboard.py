from state import FloodPredictionState
from logger import structured_log
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pyngrok import ngrok, conf

class DashboardAgent:
    def __init__(self, config):
        self.config = config
        self.app = None  # Gradio interface will be set up in setup_dashboard

    def setup_dashboard(self, state: FloodPredictionState) -> FloodPredictionState:
        """Set up and start the Gradio dashboard with sliders and flood background."""
        try:
            # Ensure state is a FloodPredictionState object
            if isinstance(state, dict):
                state = FloodPredictionState(**state)
            
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
            
            # CSS for flood background
            css = """
            .gradio-container {
                background-image: url('https://images.unsplash.com/photo-1624476869049-7e75db3b41b3');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: white;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
            }
            .gradio-container h1, .gradio-container h2, .gradio-container h3 {
                color: white;
            }
            .gradio-container .markdown {
                background: rgba(0, 0, 0, 0.5);
                padding: 10px;
                border-radius: 5px;
            }
            """
            
            # Create Gradio interface using Blocks
            with gr.Blocks(css=css) as self.app:
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
                
                # Prediction form with sliders
                gr.Markdown("## Make a Prediction")
                inputs = []
                for col in state.X_test.columns:
                    min_val = float(state.X_test[col].min())
                    max_val = float(state.X_test[col].max())
                    default_val = float(state.X_test[col].mean())
                    inputs.append(
                        gr.Slider(
                            minimum=min_val,
                            maximum=max_val,
                            value=default_val,
                            label=col,
                            step=0.1 if max_val - min_val > 10 else 0.01
                        )
                    )
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
                public_url = self.app.launch(share=True, quiet=True)  # Attempt public URL in Kaggle
                structured_log('INFO', f"Gradio dashboard launched with public URL: {public_url}")
            except Exception as e:
                structured_log('WARNING', f"Gradio launch with share=True failed: {str(e)}. Trying ngrok fallback.")
                try:
                    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your ngrok auth token
                    public_url = ngrok.connect(7860).public_url
                    self.app.launch(share=False, quiet=True)
                    structured_log('INFO', f"Gradio dashboard accessible via ngrok at {public_url}")
                except Exception as ngrok_e:
                    structured_log('ERROR', f"Ngrok fallback failed: {str(ngrok_e)}. Run locally at http://localhost:7860")
                    self.app.launch(share=False, quiet=True)
                    structured_log('INFO', "Gradio dashboard running locally at http://localhost:7860")
            
        except Exception as e:
            structured_log('ERROR', f"Error setting up dashboard: {str(e)}")
            raise
        return state