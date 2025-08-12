from state import FloodPredictionState
from logger import structured_log
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import threading

class DashboardAgent:
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__)
        self.final_state = None

    def setup_dashboard(self, state: FloodPredictionState) -> FloodPredictionState:
        """Set up Dash dashboard."""
        try:
            self.final_state = state
            feature_columns = ['MonsoonIntensity', 'RiverManagement', 'Urbanization', 'ClimateChange',
                              'DamsQuality', 'Siltation', 'AgriculturalPractices', 'Encroachments',
                              'IneffectiveDisasterPreparedness', 'CoastalVulnerability', 'Landslides',
                              'Watersheds', 'PopulationScore', 'WetlandLoss', 'InadequatePlanning',
                              'PoliticalFactors', 'TopographyDrainage', 'Deforestation',
                              'DeterioratingInfrastructure', 'DrainageSystems']
            
            self.app.layout = html.Div([
                html.H1("Flood Prediction Dashboard"),
                html.H2("Model Performance"),
                dcc.Graph(id='model-metrics'),
                html.H2("Feature Importance"),
                dcc.Graph(id='feature-importance'),
                html.H2("Prediction Distribution"),
                dcc.Graph(id='prediction-distribution'),
                html.H2("Correlation Heatmap"),
                dcc.Graph(id='correlation-heatmap'),
                html.H2("Make a Prediction"),
                html.Div([
                    html.Label(f"{col}:"),
                    dcc.Input(id=f'input-{col}', type='number', value=5)
                    for col in feature_columns
                ]),
                html.Button('Predict', id='predict-button'),
                html.Div(id='prediction-output'),
                dcc.Interval(id='monitoring-interval', interval=self.config['monitoring_interval'] * 1000, n_intervals=0)
            ])

            @self.app.callback(
                [Output('model-metrics', 'figure'),
                 Output('feature-importance', 'figure'),
                 Output('prediction-distribution', 'figure'),
                 Output('correlation-heatmap', 'figure'),
                 Output('prediction-output', 'children')],
                [Input('predict-button', 'n_clicks'),
                 Input('monitoring-interval', 'n_intervals')] + 
                [Input(f'input-{col}', 'value') for col in feature_columns]
            )
            def update_dashboard(n_clicks, n_intervals, *input_values):
                metrics_fig = go.Figure()
                if self.final_state and self.final_state.models:
                    metrics_fig.add_trace(go.Bar(
                        x=list(self.final_state.models.keys()),
                        y=[self.final_state.models[m]['r2'] for m in self.final_state.models],
                        name='R2 Score'
                    ))
                    metrics_fig.add_trace(go.Bar(
                        x=list(self.final_state.models.keys()),
                        y=[self.final_state.models[m]['mse'] for m in self.final_state.models],
                        name='MSE'
                    ))
                    metrics_fig.update_layout(barmode='group', title='Model Performance Metrics')

                feature_fig = go.Figure()
                if self.final_state and self.final_state.feature_importance:
                    feature_fig.add_trace(go.Bar(
                        x=list(self.final_state.feature_importance.keys()),
                        y=list(self.final_state.feature_importance.values()),
                        name='Feature Importance'
                    ))
                    feature_fig.update_layout(title='SHAP Feature Importance')

                pred_dist_fig = go.Figure()
                if self.final_state and self.final_state.best_model:
                    predictions = self.final_state.best_model.predict(self.final_state.X_test)
                    pred_dist_fig = px.histogram(x=predictions, nbins=30, title='Prediction Distribution')

                corr_fig = go.Figure()
                if self.final_state and self.final_state.X is not None:
                    corr_matrix = self.final_state.X.corr()
                    corr_fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu'
                    ))
                    corr_fig.update_layout(title='Feature Correlation Heatmap')

                prediction_output = ""
                if n_clicks and self.final_state and self.final_state.best_model:
                    input_data = dict(zip(feature_columns, input_values))
                    input_data['Monsoon_Drainage'] = input_data['MonsoonIntensity'] * input_data['TopographyDrainage']
                    input_data['Urban_Climate'] = input_data['Urbanization'] * input_data['ClimateChange']
                    input_data['LandslideRisk'] = input_data['TopographyDrainage'] + input_data['Deforestation']
                    input_data['InadequateInfrastructure'] = input_data['DeterioratingInfrastructure'] + input_data['DrainageSystems']
                    input_df = pd.DataFrame([input_data], columns=feature_columns)
                    input_df = input_df.drop(columns=['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems'])
                    input_df = pd.DataFrame(
                        self.final_state.scaler.transform(input_df),
                        columns=input_df.columns
                    )
                    prediction = self.final_state.best_model.predict(input_df)[0]
                    prediction_output = f"Predicted Flood Probability: {prediction:.4f}"

                return metrics_fig, feature_fig, pred_dist_fig, corr_fig, prediction_output

            # Start Dash server in a separate thread
            threading.Thread(target=self.app.run_server, kwargs={'port': self.config['dashboard_port'], 'debug': False}, daemon=True).start()
            structured_log('INFO', f"Dash dashboard started on port {self.config['dashboard_port']}")
            
        except Exception as e:
            structured_log('ERROR', f"Error setting up dashboard: {str(e)}")
            raise
        return state