CONFIG = {
    'data_path': '/kaggle/input/flood-prediction-dataset/flood.csv',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'model_params': {
        'LinearRegression': {},
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        },
        'XGBoost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        },
        'LightGBM': {
            'n_estimators': 100,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.1
        }
    },
    'output_dir': '/kaggle/working/models',
    'dashboard_port': 8050,
    'monitoring_interval': 60  # seconds
}