"""
Configuration module for the spam classifier.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

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

DEFAULT_TRAINING_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

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
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}