from state import FloodPredictionState
from logger import structured_log
import pandas as pd

class PredictorAgent:
    def make_sample_prediction(self, state) -> FloodPredictionState:
        """Make a sample prediction using the best model."""
        try:
            # Handle state as dict or FloodPredictionState
            best_model = state['best_model'] if isinstance(state, dict) else state.best_model
            X_test = state['X_test'] if isinstance(state, dict) else state.X_test
            
            if best_model is None or X_test is None:
                raise ValueError("Best model or test data not available")
            
            sample_data = X_test.iloc[0:1]
            prediction = best_model.predict(sample_data)[0]
            structured_log('INFO', f"Sample prediction for first test instance: {prediction:.4f}")
            
        except Exception as e:
            structured_log('ERROR', f"Error in prediction: {str(e)}")
            raise
        return state

    def predict(self, state, input_data: dict) -> float:
        """Make a prediction for given input data."""
        try:
            # Handle state as dict or FloodPredictionState
            best_model = state['best_model'] if isinstance(state, dict) else state.best_model
            X_test = state['X_test'] if isinstance(state, dict) else state.X_test
            
            input_df = pd.DataFrame([input_data])
            # Apply feature engineering to match training data
            input_df['Monsoon_Drainage'] = input_df['MonsoonIntensity'] * input_df['TopographyDrainage']
            input_df['Urban_Climate'] = input_df['Urbanization'] * input_df['ClimateChange']
            input_df['LandslideRisk'] = input_df['Landslides'] + input_df['TopographyDrainage']
            input_df['InadequateInfrastructure'] = input_df['DeterioratingInfrastructure'] + input_df['DrainageSystems']
            input_df = input_df.drop(columns=['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems'], errors='ignore')
            
            # Ensure input_df has same columns as X_test
            input_df = input_df[X_test.columns]
            
            prediction = best_model.predict(input_df)[0]
            return prediction
        except Exception as e:
            structured_log('ERROR', f"Error in prediction: {str(e)}")
            raise