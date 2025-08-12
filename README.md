# Flood Prediction Model
This is a machine learning application for predicting flood probability, deployed as a Gradio-based dashboard on Hugging Face Spaces. The app provides an interactive interface to explore model performance, visualize key insights, and make predictions using sliders for input features. The dashboard features a flood-themed background for visual context.

## Features
- **Model Performance Metrics**: Displays R2 score and Mean Squared Error (MSE) for trained models (LinearRegression, RandomForest, XGBoost, LightGBM).
- **Feature Importance**: Bar plot showing the importance of each feature in predicting flood probability.
- **Prediction Distribution**: Histogram of predicted flood probabilities.
- **Feature Correlation Heatmap**: Visualizes correlations between input features.
- **Interactive Prediction Form**: Sliders for input features (e.g., MonsoonIntensity, ClimateChange, LandslideRisk) with dynamic ranges based on the dataset.
- **Flood-Themed Background**: Uses a flood-related image (`https://www.spml.co.in/Images/blog/wdt&c-152776632.jpg`) for an immersive interface.

## Usage
1. Open the app at `https://huggingface.co/spaces/sravan837/flood-prediction-app`.
2. Explore visualizations:
   - **Model Performance**: View R2 and MSE metrics in a table.
   - **Feature Importance**: See which features most influence flood predictions.
   - **Prediction Distribution**: Understand the distribution of predicted flood probabilities.
   - **Correlation Heatmap**: Analyze relationships between features.
3. Use the prediction form:
   - Adjust sliders for each feature (e.g., MonsoonIntensity, Urban_Climate).
   - Click "Predict" to get the flood probability.
4. Note: The sliders use dynamic ranges derived from the dataset for accurate predictions.

## Dataset
- **Source**: `data/flood.csv` (21 columns, including `FloodProbability`).
- **Feature Engineering**:
  - **Added Features**: `Monsoon_Drainage`, `Urban_Climate`, `LandslideRisk`, `InadequateInfrastructure`.
  - **Dropped Features**: `TopographyDrainage`, `Deforestation`, `DeterioratingInfrastructure`, `DrainageSystems`.
- **Preprocessing**: Features are scaled and cleaned for model training.

## Models
- **Algorithms**: LinearRegression, RandomForest, XGBoost, LightGBM.
- **Hyperparameters**: Default settings (e.g., `n_estimators=50` for tree-based models).
- **Tuning**: No GridSearchCV; best model selected based on performance metrics.
- **Feature Importance**: Derived from the best model (model-based, no SHAP).

## Setup and Deployment
- **Platform**: Hugging Face Spaces, using Gradio SDK (version 4.44.0).
- **Files**:
  - `app.py`: Main entry point, runs the pipeline and launches the Gradio dashboard.
  - `dashboard.py`: Defines the Gradio interface with sliders and visualizations.
  - `data/flood.csv`: Dataset for training and predictions.
  - Other agent files: `config.py`, `state.py`, `logger.py`, `data_loader.py`, `preprocessor.py`, `model_trainer.py`, `model_tuner.py`, `explainer.py`, `visualizer.py`, `monitor.py`, `model_saver.py`, `predictor.py`.
  - `requirements.txt`: Lists dependencies (pandas, numpy, scikit-learn, xgboost, lightgbm, gradio, etc.).
- **Directory Structure**:
  ```
  flood-prediction-app/
  ├── app.py
  ├── config.py
  ├── data/
  │   └── flood.csv
  ├── dashboard.py
  ├── data_loader.py
  ├── explainer.py
  ├── logger.py
  ├── model_saver.py
  ├── model_trainer.py
  ├── model_tuner.py
  ├── monitor.py
  ├── predictor.py
  ├── preprocessor.py
  ├── state.py
  ├── visualizer.py
  ├── requirements.txt
  ├── README.md
  └── models/  # Created at runtime
  ```

## Local Development
To run locally:
1. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/sravan837/flood-prediction-app
   cd flood-prediction-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Access the dashboard at `http://localhost:7860`.

## Troubleshooting
- **Build Fails**: Check logs in Space settings for missing files or dependencies. Ensure `data/flood.csv` is present and `requirements.txt` includes all packages.
- **Dashboard Issues**: Verify `gradio==4.44.0` and the background image URL (`https://www.spml.co.in/Images/blog/wdt&c-152776632.jpg`). If the image fails, update `dashboard.py` with an alternative URL.
- **Serialization Errors**: Ensure `logger.py` and `explainer.py` handle NumPy types (`int32`, `str_`).

## Contact
For issues or contributions, open an issue on the Space’s repository or contact the owner via Hugging Face.

---
*Deployed on August 12, 2025*
