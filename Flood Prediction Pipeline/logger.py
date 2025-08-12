import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def structured_log(level: str, message: str, **kwargs):
    log_dict = {'message': message, **kwargs}
    if level.upper() == 'INFO':
        logging.info(json.dumps(log_dict))
    elif level.upper() == 'ERROR':
        logging.error(json.dumps(log_dict))
    else:
        logging.debug(json.dumps(log_dict))