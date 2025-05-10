"""
Configuration module for the spam classifier.

This module defines global configuration settings for the application,
including paths, model parameters, and logging settings.
"""

import os
import yaml
import logging
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'multinomial': {
        'alpha': 1.0,
        'fit_prior': True
    },
    'bernoulli': {
        'alpha': 1.0,
        'fit_prior': True,
        'binarize': 0.0
    },
    'gaussian': {
        'var_smoothing': 1e-9
    },
    'complement': {
        'alpha': 1.0,
        'fit_prior': True,
        'norm': False
    }
}

# Default preprocessing parameters
DEFAULT_PREPROCESSING_PARAMS = {
    'min_df': 5,
    'max_df': 0.8,
    'max_features': 10000,
    'stop_words': 'english',
    'ngram_range': (1, 2),
    'use_stemming': True,
    'use_lemmatization': False,
    'remove_urls': True,
    'remove_numbers': True,
    'remove_punctuation': True
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': os.path.join(LOGS_DIR, 'spam_classifier.log'),
            'mode': 'a',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}


def load_config(config_file=None):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str, optional
        Path to the configuration file. If None, default configuration is used.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    config = {
        'model_params': DEFAULT_MODEL_PARAMS,
        'preprocessing_params': DEFAULT_PREPROCESSING_PARAMS,
        'training_params': DEFAULT_TRAINING_PARAMS,
        'logging_config': LOGGING_CONFIG
    }

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)

            # Update the configuration with user settings
            if 'model_params' in user_config:
                for model, params in user_config['model_params'].items():
                    if model in config['model_params']:
                        config['model_params'][model].update(params)

            if 'preprocessing_params' in user_config:
                config['preprocessing_params'].update(user_config['preprocessing_params'])

            if 'training_params' in user_config:
                config['training_params'].update(user_config['training_params'])

            if 'logging_config' in user_config:
                # This is more complex, so we just replace it if provided
                config['logging_config'] = user_config['logging_config']

    return config