import streamlit as st
import pandas as pd
import joblib

# --- 1. MODEL LOADING ---

# Use st.cache_resource to load the model only once and store it in memory
@st.cache_resource
def load_model(model_path):
    """Loads a saved model from a file."""
    model = joblib.load(model_path)
    return model

# --- 2. USER INTERFACE (SIDEBAR) ---

st.sidebar.header('Adjust Input Features')

def user_input_features():
    """Creates sidebar sliders and returns user inputs as a DataFrame."""
    # Create sliders for each feature. The min, max, and default values are set as examples.
    # You might want to adjust these based on your dataset's statistics (e.g., df.describe()).
    monsoon_intensity = st.sidebar.slider('Monsoon Intensity', 0, 20, 5)
    topography_drainage = st.sidebar.slider('Topography Drainage', 0, 20, 5)
    river_management = st.sidebar.slider('River Management', 0, 20, 5)
    deforestation = st.sidebar.slider('Deforestation', 0, 20, 5)
    urbanization = st.sidebar.slider('Urbanization', 0, 20, 5)
    climate_change = st.sidebar.slider('Climate Change', 0, 20, 5)
    dams_quality = st.sidebar.slider('Dams Quality', 0, 20, 5)
    siltation = st.sidebar.slider('Siltation', 0, 20, 5)
    agricultural_practices = st.sidebar.slider('Agricultural Practices', 0, 20, 5)
    encroachments = st.sidebar.slider('Encroachments', 0, 20, 5)
    effective_disaster_preparedness = st.sidebar.slider('Effective Disaster Preparedness', 0, 20, 5)
    drainage_systems = st.sidebar.slider('Drainage Systems', 0, 20, 5)
    coastal_vulnerability = st.sidebar.slider('Coastal Vulnerability', 0, 20, 5)
    landslides = st.sidebar.slider('Landslides', 0, 20, 5)
    watersheds = st.sidebar.slider('Watersheds', 0, 20, 5)
    deteriorating_infrastructure = st.sidebar.slider('Deteriorating Infrastructure', 0, 20, 5)
    population_score = st.sidebar.slider('Population Score', 0, 20, 5)
    wetland_loss = st.sidebar.slider('Wetland Loss', 0, 20, 5)
    inadequate_planning = st.sidebar.slider('Inadequate Planning', 0, 20, 5)
    political_factors = st.sidebar.slider('Political Factors', 0, 20, 5)

    # Store inputs in a dictionary
    data = {
        'MonsoonIntensity': monsoon_intensity,
        'TopographyDrainage': topography_drainage,
        'RiverManagement': river_management,
        'Deforestation': deforestation,
        'Urbanization': urbanization,
        'ClimateChange': climate_change,
        'DamsQuality': dams_quality,
        'Siltation': siltation,
        'AgriculturalPractices': agricultural_practices,
        'Encroachments': encroachments,
        'EffectiveDisasterPreparedness': effective_disaster_preparedness,
        'DrainageSystems': drainage_systems,
        'CoastalVulnerability': coastal_vulnerability,
        'Landslides': landslides,
        'Watersheds': watersheds,
        'DeterioratingInfrastructure': deteriorating_infrastructure,
        'PopulationScore': population_score,
        'WetlandLoss': wetland_loss,
        'InadequatePlanning': inadequate_planning,
        'PoliticalFactors': political_factors
    }
    
    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# --- 3. MAIN APPLICATION LOGIC ---

# Set the title of the app
st.title('Flood Prediction Web App 🌊')

# Add some introductory text
st.write(
    "This app uses a LightGBM model to predict the probability of a flood. "
    "Adjust the feature values in the sidebar on the left to see the prediction."
)

# Load the pre-trained model
model_path = 'lgbm_flood_prediction_model.joblib'
model = load_model(model_path)

# Get user input from the sidebar
input_df = user_input_features()

# Display the user's selected feature values
st.subheader('Your Input Features')
st.write(input_df)

# Create a button to trigger the prediction
if st.button('Predict Flood Probability'):
    # --- 4. FEATURE ENGINEERING & PREDICTION ---
    
    # IMPORTANT: Apply the same feature engineering steps as in your training script
    input_engineered = input_df.copy()
    input_engineered['LandslideRisk'] = input_engineered['TopographyDrainage'] + input_engineered['Deforestation']
    input_engineered['InadequateInfrastructure'] = input_engineered['DeterioratingInfrastructure'] + input_engineered['DrainageSystems']
    
    # Drop the original columns used for engineering
    cols_to_drop = ['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems']
    input_final = input_engineered.drop(columns=cols_to_drop)


    # Make the prediction
    prediction = model.predict(input_final)
    
    # Display the result
    st.subheader('Prediction Result')
    st.success(f'The predicted flood probability is: {prediction[0]:.2f}')