from state import FloodPredictionState
from logger import structured_log
import pandas as pd

class PredictorAgent:
    def predict(self, state: FloodPredictionState, input_data: dict) -> float:
        """Make predictions using the best model."""
        try:
            if state.best_model is None:
                raise ValueError("No trained model available.")
            
            input_data['Monsoon_Drainage'] = input_data['MonsoonIntensity'] * input_data['TopographyDrainage']
            input_data['Urban_Climate'] = input_data['Urbanization'] * input_data['ClimateChange']
            input_df = pd.DataFrame([input_data], columns=state.X.columns)
            input_df = pd.DataFrame(
                state.scaler.transform(input_df),
                columns=input_df.columns
            )
            prediction = state.best_model.predict(input_df)[0]
            structured_log('INFO', f"Generated prediction: {prediction:.4f}")
            return prediction
        except Exception as e:
            structured_log('ERROR', f"Error in prediction: {str(e)}")
            raise

    def make_sample_prediction(self, state: FloodPredictionState) -> FloodPredictionState:
        """Make a sample prediction."""
        try:
            if state.best_model is None:
                raise ValueError("No trained model available.")
            
            sample_data = state.X_test.iloc[[0]]
            state.sample_prediction = state.best_model.predict(sample_data)[0]
            structured_log('INFO', f"Sample prediction: {state.sample_prediction:.4f}")
            
        except Exception as e:
            structured_log('ERROR', f"Error in sample prediction: {str(e)}")
            raise
        return state