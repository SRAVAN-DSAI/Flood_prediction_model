from state import FloodPredictionState
from logger import structured_log
import pandas as pd

class PredictorAgent:
    def make_sample_prediction(self, state: FloodPredictionState) -> FloodPredictionState:
        """Make a sample prediction using the best model."""
        try:
            if state.best_model is None or state.X_test is None:
                raise ValueError("Best model or test data not available")
            
            sample_data = state.X_test.iloc[0:1]
            prediction = state.best_model.predict(sample_data)[0]
            structured_log('INFO', f"Sample prediction for first test instance: {prediction:.4f}")
            
        except Exception as e:
            structured_log('ERROR', f"Error in prediction: {str(e)}")
            raise
        return state

    def predict(self, state: FloodPredictionState, input_data: dict) -> float:
        """Make a prediction for given input data."""
        try:
            input_df = pd.DataFrame([input_data])
            # Apply feature engineering to match training data
            input_df['Monsoon_Drainage'] = input_df['MonsoonIntensity'] * input_df['TopographyDrainage']
            input_df['Urban_Climate'] = input_df['Urbanization'] * input_df['ClimateChange']
            input_df['LandslideRisk'] = input_df['Landslides'] + input_df['TopographyDrainage']
            input_df['InadequateInfrastructure'] = input_df['DeterioratingInfrastructure'] + input_df['DrainageSystems']
            input_df = input_df.drop(columns=['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems'], errors='ignore')
            
            prediction = state.best_model.predict(input_df)[0]
            return prediction
        except Exception as e:
            structured_log('ERROR', f"Error in prediction: {str(e)}")
            raise