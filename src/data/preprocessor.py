"""
Text preprocessing module for the spam classifier.

This module provides functions for preprocessing email text data.
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from ..utils.logger import logger
from ..utils.config import DEFAULT_PREPROCESSING_PARAMS, DEFAULT_TRAINING_PARAMS

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


class EmailPreprocessor:
    """
    Class for preprocessing email data.

    This class provides methods for cleaning, tokenizing, and vectorizing
    email text data for spam classification.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary for preprocessing parameters.
    """

    def __init__(self, config=None):
        """Initialize the EmailPreprocessor."""
        self.config = config or DEFAULT_PREPROCESSING_PARAMS.copy()

        # Initialize tools based on configuration
        self.stemmer = PorterStemmer() if self.config.get('use_stemming', True) else None
        self.lemmatizer = WordNetLemmatizer() if self.config.get('use_lemmatization', False) else None
        self.stop_words = set(stopwords.words('english'))

        # Vectorizer (will be initialized when fitting)
        self.vectorizer = None

        logger.info("Initialized EmailPreprocessor")
        logger.debug(f"Preprocessing configuration: {self.config}")

    def clean_text(self, text):
        """
        Clean and normalize email text.

        Parameters
        ----------
        text : str
            Text to be cleaned.

        Returns
        -------
        str
            Cleaned text.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs if configured
        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses if configured
        if self.config.get('remove_emails', True):
            text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags if configured
        if self.config.get('remove_html', True):
            text = re.sub(r'<.*?>', '', text)

        # Remove numbers if configured
        if self.config.get('remove_numbers', True):
            text = re.sub(r'\d+', '', text)

        # Remove punctuation if configured
        if self.config.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_text(self, text):
        """
        Tokenize text and apply stemming/lemmatization if configured.

        Parameters
        ----------
        text : str
            Text to be tokenized.

        Returns
        -------
        list
            List of tokens.
        """
        # Clean text first
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Remove stopwords and short tokens
        min_length = self.config.get('min_token_length', 2)
        filtered_tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) >= min_length
        ]

        # Apply stemming if configured
        if self.stemmer is not None:
            filtered_tokens = [self.stemmer.stem(token) for token in filtered_tokens]

        # Apply lemmatization if configured
        if self.lemmatizer is not None:
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

        return filtered_tokens

    def process_emails(self, email_df, text_column='body'):
        """
        Process emails by cleaning and tokenizing the text.

        Parameters
        ----------
        email_df : pandas.DataFrame
            DataFrame containing email data.
        text_column : str, optional
            Name of the column containing the email text.

        Returns
        -------
        pandas.DataFrame
            DataFrame with processed emails.
        """
        logger.info(f"Processing {len(email_df)} emails")

        # Create a copy to avoid modifying the original DataFrame
        processed_df = email_df.copy()

        # Create a new column for the processed text
        processed_df['processed_text'] = processed_df[text_column].apply(
            lambda x: ' '.join(self.tokenize_text(x))
        )

        # Create a new column for token count
        processed_df['token_count'] = processed_df['processed_text'].apply(
            lambda x: len(x.split())
        )

        logger.info("Email processing complete")
        return processed_df

    def vectorize_text(self, processed_df, vectorizer_type='tfidf'):
        """
        Vectorize the processed text.

        Parameters
        ----------
        processed_df : pandas.DataFrame
            DataFrame containing processed email data.
        vectorizer_type : str, optional
            Type of vectorizer to use ('count', 'tfidf', or 'binary').

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of features.
        list
            List of feature names.
        """
        # Get configuration for vectorizer
        ngram_range = self.config.get('ngram_range', (1, 2))
        max_features = self.config.get('max_features', 10000)
        min_df = self.config.get('min_df', 5)
        max_df = self.config.get('max_df', 0.8)

        logger.info(f"Vectorizing text using {vectorizer_type} vectorizer")

        # Initialize vectorizer based on type
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df
            )
        elif vectorizer_type == 'binary':
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                binary=True
            )
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df
            )

        # Fit and transform the processed text
        X = self.vectorizer.fit_transform(processed_df['processed_text'])
        feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"Vectorization complete: {X.shape[1]} features extracted")
        return X, feature_names

    def prepare_data(self, email_df, vectorizer_type='tfidf', text_column='body', config=None):
        """
        Prepare data for training and testing.

        Parameters
        ----------
        email_df : pandas.DataFrame
            DataFrame containing email data.
        vectorizer_type : str, optional
            Type of vectorizer to use.
        text_column : str, optional
            Name of the column containing the email text.
        config : dict, optional
            Configuration for train/test split.

        Returns
        -------
        dict
            Dictionary containing the prepared data.
        """
        config = config or DEFAULT_TRAINING_PARAMS.copy()

        # Process emails
        processed_df = self.process_emails(email_df, text_column=text_column)

        # Vectorize text
        X, feature_names = self.vectorize_text(processed_df, vectorizer_type=vectorizer_type)

        # Get labels
        y = (processed_df['label'] == 'spam').astype(int).values

        # Split data
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        stratify = processed_df['label'] if config.get('stratify', True) else None

        logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Return prepared data
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