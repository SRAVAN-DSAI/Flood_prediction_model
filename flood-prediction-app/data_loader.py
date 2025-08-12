from state import FloodPredictionState
from logger import structured_log
import pandas as pd

class DataLoaderAgent:
    def load_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Load the dataset from the specified path."""
        try:
            structured_log('INFO', f"Loading data from {state.data_path}")
            state.df = pd.read_csv(state.data_path)
            structured_log('INFO', f"Dataset loaded with shape {state.df.shape}")
        except Exception as e:
            structured_log('ERROR', f"Error loading data: {str(e)}")
            raise
        return state