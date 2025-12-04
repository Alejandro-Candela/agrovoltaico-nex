import logging
from datetime import datetime
import pytz

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def get_current_time_madrid() -> datetime:
    """Returns current time in Europe/Madrid timezone."""
    madrid_tz = pytz.timezone('Europe/Madrid')
    return datetime.now(madrid_tz)
