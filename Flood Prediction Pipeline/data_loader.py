from state import FloodPredictionState
from logger import structured_log
import pandas as pd
import os
from retry import retry

class DataLoaderAgent:
    @retry(tries=3, delay=2, backoff=2)
    def load_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Load and validate the dataset."""
        try:
            if not os.path.exists(state.data_path):
                raise FileNotFoundError(f"Data file not found at: {state.data_path}")
            
            state.df = pd.read_csv(state.data_path)
            structured_log('INFO', "Dataset loaded successfully", shape=str(state.df.shape))
            
            if state.df.empty:
                raise ValueError("Loaded dataset is empty.")
            if 'FloodProbability' not in state.df.columns:
                raise ValueError("Target column 'FloodProbability' not found in dataset.")
            
            state.X = state.df.drop('FloodProbability', axis=1)
            state.y = state.df['FloodProbability']
            structured_log('INFO', "Features and target prepared")
            
        except Exception as e:
            structured_log('ERROR', f"Error loading data: {str(e)}")
            raise
        return state