import logging
import json
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_serializable(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj

def structured_log(level: str, message: str, **kwargs):
    log_dict = {'message': message}
    log_dict.update({k: convert_to_serializable(v) for k, v in kwargs.items()})
    log_dict = convert_to_serializable(log_dict)  # Apply to entire dictionary
    if level.upper() == 'INFO':
        logging.info(json.dumps(log_dict))
    elif level.upper() == 'ERROR':
        logging.error(json.dumps(log_dict))
    else:
        logging.debug(json.dumps(log_dict))s