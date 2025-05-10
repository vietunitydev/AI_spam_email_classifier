"""
Logging module for the spam classifier.

This module sets up logging for the application.
"""

import logging
import logging.config
from .config import LOGGING_CONFIG, LOGS_DIR
import os


def setup_logging(config=None):
    """
    Set up logging configuration.

    Parameters
    ----------
    config : dict, optional
        Logging configuration. If None, default configuration is used.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if config is None:
        config = LOGGING_CONFIG

    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Configure logging
    logging.config.dictConfig(config)

    # Return the root logger
    return logging.getLogger()


# Create a default logger instance
logger = setup_logging()