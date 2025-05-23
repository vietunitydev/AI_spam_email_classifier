"""
Text preprocessing module for the spam classifier.
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.config import DEFAULT_PREPROCESSING_PARAMS, DEFAULT_TRAINING_PARAMS

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


class EmailPreprocessor:
    def __init__(self, config=None):
        self.config = config or DEFAULT_PREPROCESSING_PARAMS.copy()
        self.stemmer = PorterStemmer() if self.config.get('use_stemming', True) else None
        self.lemmatizer = WordNetLemmatizer() if self.config.get('use_lemmatization', False) else None
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None

        logger.info("Initialized EmailPreprocessor")

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()

        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www\S+', '', text)

        if self.config.get('remove_emails', True):
            text = re.sub(r'\S+@\S+', '', text)

        if self.config.get('remove_html', True):
            text = re.sub(r'<.*?>', '', text)

        if self.config.get('remove_numbers', True):
            text = re.sub(r'\d+', '', text)

        if self.config.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)

        min_length = self.config.get('min_token_length', 2)
        filtered_tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) >= min_length
        ]

        if self.stemmer is not None:
            filtered_tokens = [self.stemmer.stem(token) for token in filtered_tokens]

        if self.lemmatizer is not None:
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        return filtered_tokens

    def process_emails(self, email_df, text_column='body'):
        logger.info(f"Processing {len(email_df)} emails")

        processed_df = email_df.copy()
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: ' '.join(self.tokenize_text(x))
        )
        processed_df['token_count'] = processed_df['processed_text'].apply(
            lambda x: len(x.split())
        )

        logger.info("Email processing complete")
        return processed_df

    def vectorize_text(self, processed_df):
        ngram_range = self.config.get('ngram_range', (1, 2))
        max_features = self.config.get('max_features', 10000)
        min_df = self.config.get('min_df', 5)
        max_df = self.config.get('max_df', 0.8)

        logger.info("Vectorizing text using TF-IDF vectorizer")

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df
        )

        X = self.vectorizer.fit_transform(processed_df['processed_text'])
        feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"Vectorization complete: {X.shape[1]} features extracted")
        return X, feature_names

    def prepare_data(self, email_df, text_column='body', config=None):
        config = config or DEFAULT_TRAINING_PARAMS.copy()

        processed_df = self.process_emails(email_df, text_column=text_column)
        X, feature_names = self.vectorize_text(processed_df)
        y = (processed_df['label'] == 'spam').astype(int).values

        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        stratify = processed_df['label'] if config.get('stratify', True) else None

        logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'vectorizer': self.vectorizer,
            'processed_df': processed_df
        }

        logger.info(f"Data preparation complete: {X_train.shape[0]} training samples, "
                    f"{X_test.shape[0]} testing samples")

        return data