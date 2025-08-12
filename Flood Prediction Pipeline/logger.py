import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('flood_prediction_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def structured_log(level, message, **kwargs):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'level': level,
        'message': message,
        **kwargs
    }
    logger.log(getattr(logging, level.upper()), json.dumps(log_entry))