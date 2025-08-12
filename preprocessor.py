from state import FloodPredictionState
from logger import structured_log
from sklearn.model_selection import train_test_split
import pandas as pd

class PreprocessorAgent:
    def __init__(self, config):
        self.config = config

    def preprocess_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Preprocess the dataset, apply feature engineering, and split into train/test."""
        try:
            if state.df is None:
                raise ValueError("No dataset available for preprocessing")
            
            # Feature engineering
            state.df['Monsoon_Drainage'] = state.df['MonsoonIntensity'] * state.df['TopographyDrainage']
            state.df['Urban_Climate'] = state.df['Urbanization'] * state.df['ClimateChange']
            state.df['LandslideRisk'] = state.df['Landslides'] + state.df['TopographyDrainage']
            state.df['InadequateInfrastructure'] = state.df['DeterioratingInfrastructure'] + state.df['DrainageSystems']
            
            # Drop specified columns
            columns_to_drop = ['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems']
            state.df = state.df.drop(columns=columns_to_drop, errors='ignore')
            structured_log('INFO', f"Dropped columns: {columns_to_drop}")
            
            # Split features and target
            X = state.df.drop(columns=['FloodProbability'])
            y = state.df['FloodProbability']
            
            # Train-test split
            state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
                X, y, test_size=self.config['test_size'], random_state=self.config['random_state']
            )
            structured_log('INFO', f"Train shape: {state.X_train.shape}, Test shape: {state.X_test.shape}")
            
        except Exception as e:
            structured_log('ERROR', f"Error in preprocessing: {str(e)}")
            raise
        return state