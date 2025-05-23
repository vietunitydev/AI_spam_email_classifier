"""
Logging module for the spam classifier.
"""

import logging
import logging.config
from .config import LOGGING_CONFIG, LOGS_DIR
import os


def setup_logging(config=None):
    if config is None:
        config = LOGGING_CONFIG

    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.config.dictConfig(config)
    return logging.getLogger()


logger = setup_logging()