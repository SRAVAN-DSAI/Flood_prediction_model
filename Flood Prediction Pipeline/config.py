CONFIG = {
    'data_path': '/kaggle/input/flood-prediction-dataset/flood.csv',
    'test_size': 0.2,
    'random_state': 42,
    'model_params': {
        'RandomForest': {},
        'XGBoost': {},
        'LightGBM': {}
    },
    'output_dir': '/kaggle/working/models',
    'dashboard_port': 8050
}