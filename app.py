import streamlit as st
import pandas as pd
import joblib

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads a saved model from a file."""
    model = joblib.load(model_path)
    return model

# --- 2. USER INTERFACE (SIDEBAR) ---
# This section creates the interactive sliders in the sidebar.
st.sidebar.header('Adjust Input Features')

def user_input_features():
    """Creates sidebar sliders with sensible ranges and returns user inputs as a DataFrame."""
    
    st.sidebar.subheader("Environmental Factors")
    monsoon_intensity = st.sidebar.slider('Monsoon Intensity (1-16)', 1, 16, 8)
    topography_drainage = st.sidebar.slider('Topography & Drainage (1-18)', 1, 18, 9)
    deforestation = st.sidebar.slider('Deforestation Level (0-17)', 0, 17, 8)
    climate_change = st.sidebar.slider('Climate Change Impact (0-17)', 0, 17, 8)
    landslides = st.sidebar.slider('Landslide Risk (0-16)', 0, 16, 8)
    coastal_vulnerability = st.sidebar.slider('Coastal Vulnerability (0-17)', 0, 17, 8)
    
    st.sidebar.subheader("Infrastructural & Management Factors")
    river_management = st.sidebar.slider('River Management Quality (1-16)', 1, 16, 8)
    dams_quality = st.sidebar.slider('Dams Quality (1-16)', 1, 16, 8)
    drainage_systems = st.sidebar.slider('Drainage Systems Quality (1-17)', 1, 17, 9)
    deteriorating_infrastructure = st.sidebar.slider('Infrastructure Condition (1-17)', 1, 17, 9)
    watersheds = st.sidebar.slider('Watershed Health (0-16)', 0, 16, 8)
    siltation = st.sidebar.slider('River Siltation Level (1-16)', 1, 16, 8)

    st.sidebar.subheader("Socio-Economic & Governance Factors")
    urbanization = st.sidebar.slider('Urbanization Level (0-17)', 0, 17, 9)
    agricultural_practices = st.sidebar.slider('Agricultural Practices Impact (0-16)', 0, 16, 8)
    encroachments = st.sidebar.slider('Land Encroachments (0-18)', 0, 18, 9)
    effective_disaster_preparedness = st.sidebar.slider('Disaster Preparedness (1-16)', 1, 16, 8)
    population_score = st.sidebar.slider('Population Density Score (0-18)', 0, 18, 9)
    wetland_loss = st.sidebar.slider('Wetland Loss (0-20)', 0, 20, 10)
    inadequate_planning = st.sidebar.slider('Inadequate Planning (0-16)', 0, 16, 8)
    political_factors = st.sidebar.slider('Political Factors (0-16)', 0, 16, 8)

    # All 20 original features are collected from the user
    data = {
        'MonsoonIntensity': monsoon_intensity, 'TopographyDrainage': topography_drainage, 'RiverManagement': river_management,
        'Deforestation': deforestation, 'Urbanization': urbanization, 'ClimateChange': climate_change, 'DamsQuality': dams_quality,
        'Siltation': siltation, 'AgriculturalPractices': agricultural_practices, 'Encroachments': encroachments,
        'EffectiveDisasterPreparedness': effective_disaster_preparedness, 'DrainageSystems': drainage_systems,
        'CoastalVulnerability': coastal_vulnerability, 'Landslides': landslides, 'Watersheds': watersheds,
        'DeterioratingInfrastructure': deteriorating_infrastructure, 'PopulationScore': population_score,
        'WetlandLoss': wetland_loss, 'InadequatePlanning': inadequate_planning, 'PoliticalFactors': political_factors
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# --- 3. MAIN APPLICATION LOGIC ---

st.title('Flood Prediction Web App 🌊')
st.write(
    "This app uses a pre-trained LightGBM model to predict flood probability. "
    "Adjust the real-world factors in the sidebar to see how they impact the flood risk."
)

model = load_model('lgbm_flood_prediction_model.joblib')
input_df = user_input_features()

st.subheader('Your Input Features')
st.table(input_df.T.rename(columns={0: 'Value'}))

if st.button('Predict Flood Probability'):
    # --- 4. FEATURE ENGINEERING & PREDICTION (MODIFIED FOR 18 FEATURES) ---
    
    # Start with the 20 features from the user
    input_engineered = input_df.copy()
    
    # Add 2 new features, bringing the total to 22
    input_engineered['LandslideRisk'] = input_engineered['TopographyDrainage'] + input_engineered['Deforestation']
    input_engineered['InadequateInfrastructure'] = input_engineered['DeterioratingInfrastructure'] + input_engineered['DrainageSystems']
    
    # Drop the 4 original features, bringing the total to the required 18
    cols_to_drop = ['TopographyDrainage', 'Deforestation', 'DeterioratingInfrastructure', 'DrainageSystems']
    input_final = input_engineered.drop(columns=cols_to_drop)

    # Make the prediction using the final 18-feature DataFrame
    prediction = model.predict(input_final)
    
    st.subheader('Prediction Result')
    probability = prediction[0]
    
    st.progress(probability)
    
    if probability < 0.45:
        st.success(f'Low Flood Risk: {probability:.2%}')
    elif probability < 0.55:
        st.warning(f'Moderate Flood Risk: {probability:.2%}')
    else:
        st.error(f'High Flood Risk: {probability:.2%}')