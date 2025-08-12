from state import FloodPredictionState
from logger import structured_log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PreprocessorAgent:
    def __init__(self, config):
        self.config = config

    def preprocess_data(self, state: FloodPredictionState) -> FloodPredictionState:
        """Preprocess data with feature engineering."""
        try:
            # Create a copy of the original feature set
            X_engineered = state.X.copy()
            
            # Feature engineering: Add new features
            X_engineered['Monsoon_Drainage'] = X_engineered['MonsoonIntensity'] * X_engineered['TopographyDrainage']
            X_engineered['Urban_Climate'] = X_engineered['Urbanization'] * X_engineered['ClimateChange']
            X_engineered['LandslideRisk'] = X_engineered['TopographyDrainage'] + X_engineered['Deforestation']
            X_engineered['InadequateInfrastructure'] = X_engineered['DeterioratingInfrastructure'] + X_engineered['DrainageSystems']
            
            # Drop specified columns
            cols_to_drop = ['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems']
            state.X = X_engineered.drop(columns=cols_to_drop)
            
            # Split data
            state.X_train, state.X_test, state.y_train, state.y_test = train_test_split(
                state.X, state.y, test_size=self.config['test_size'], random_state=self.config['random_state']
            )
            
            # Scale features
            state.scaler = StandardScaler()
            state.X_train = pd.DataFrame(
                state.scaler.fit_transform(state.X_train),
                columns=state.X_train.columns,
                index=state.X_train.index
            )
            state.X_test = pd.DataFrame(
                state.scaler.transform(state.X_test),
                columns=state.X_test.columns,
                index=state.X_test.index
            )
            
            structured_log('INFO', f"Data split and scaled: training (n={len(state.X_train)}), test (n={len(state.X_test)})")
            
            if state.X_train.isnull().any().any() or state.y_train.isnull().any():
                raise ValueError("Missing values detected in the dataset.")
            
        except Exception as e:
            structured_log('ERROR', f"Error in preprocessing: {str(e)}")
            raise
        return state