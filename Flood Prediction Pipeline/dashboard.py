from state import FloodPredictionState
from logger import structured_log
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import threading
import os

class DashboardAgent:
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__)

    def setup_dashboard(self, state: FloodPredictionState) -> FloodPredictionState:
        """Set up and start the Dash dashboard."""
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
            
            # Define layout
            self.app.layout = html.Div([
                html.H1('Flood Prediction Dashboard'),
                
                html.Div([
                    html.H3('Model Performance'),
                    dcc.Graph(figure=fig_table)
                ]),
                
                html.Div([
                    html.H3('Feature Importance'),
                    dcc.Graph(figure=fig_importance)
                ]),
                
                html.Div([
                    html.H3('Prediction Distribution'),
                    dcc.Graph(figure=fig_dist)
                ]),
                
                html.Div([
                    html.H3('Feature Correlation'),
                    dcc.Graph(figure=fig_corr)
                ]),
                
                html.Div([
                    html.H3('Make a Prediction'),
                    html.Label('Select Feature Values:'),
                    html.Div([
                        html.Label(col),
                        dcc.Input(id=col, type='number', value=0, step=1)
                    ]) for col in state.X_test.columns
                    ]),
                
                html.Button('Predict', id='predict-button'),
                html.Div(id='prediction-output')
            ])
            
            # Callback for prediction
            @self.app.callback(
                Output('prediction-output', 'children'),
                [Input('predict-button', 'n_clicks')] + [
                    Input(col, 'value') for col in state.X_test.columns
                ]
            )
            def update_prediction(n_clicks, *input_values):
                if n_clicks is None:
                    return "Click the button to predict"
                input_data = pd.DataFrame([input_values], columns=state.X_test.columns)
                try:
                    prediction = state.best_model.predict(input_data)[0]
                    return f"Predicted Flood Probability: {prediction:.4f}"
                except Exception as e:
                    return f"Error in prediction: {str(e)}"
            
            # Start Dash server in a separate thread
            threading.Thread(target=self.app.run, kwargs={'port': self.config['dashboard_port'], 'debug': False}, daemon=True).start()
            structured_log('INFO', f"Dash dashboard started on port {self.config['dashboard_port']}")
            
        except Exception as e:
            structured_log('ERROR', f"Error setting up dashboard: {str(e)}")
            raise
        return state