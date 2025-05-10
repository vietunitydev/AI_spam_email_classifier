"""
Naive Bayes models for the spam classifier.

This module implements various Naive Bayes algorithms for email spam classification.
"""

import numpy as np
import joblib
import os
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from ..utils.logger import logger
from ..utils.config import DEFAULT_MODEL_PARAMS, MODELS_DIR


class NaiveBayesModel:
    """
    Wrapper class for Naive Bayes models for spam classification.

    This class provides a unified interface for various Naive Bayes models,
    including training, prediction, and model persistence.

    Parameters
    ----------
    model_type : str, optional
        Type of Naive Bayes model ('multinomial', 'bernoulli', 'gaussian', or 'complement').
    params : dict, optional
        Model parameters.
    """

    def __init__(self, model_type='multinomial', params=None):
        """Initialize the NaiveBayesModel."""
        self.model_type = model_type.lower()

        # Get default parameters for the selected model type
        default_params = DEFAULT_MODEL_PARAMS.get(self.model_type, {})

        # Update with user-provided parameters if any
        self.params = default_params.copy()
        if params:
            self.params.update(params)

        # Initialize the appropriate model
        self.model = self._create_model()

        logger.info(f"Initialized {self.model_type.capitalize()} Naive Bayes model")
        logger.debug(f"Model parameters: {self.params}")

    def _create_model(self):
        """
        Create the appropriate Naive Bayes model based on model_type.

        Returns
        -------
        object
            Sklearn Naive Bayes model instance.
        """
        if self.model_type == 'multinomial':
            return MultinomialNB(**self.params)
        elif self.model_type == 'bernoulli':
            return BernoulliNB(**self.params)
        elif self.model_type == 'gaussian':
            return GaussianNB(**self.params)
        elif self.model_type == 'complement':
            return ComplementNB(**self.params)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X_train, y_train):
        """
        Train the model on the given data.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.

        Returns
        -------
        self
            The trained model instance.
        """
        logger.info(f"Training {self.model_type.capitalize()} Naive Bayes model")
        self.model.fit(X_train, y_train)
        logger.info("Model training complete")
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like
            Samples to predict.

        Returns
        -------
        array
            Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like
            Samples to predict.

        Returns
        -------
        array
            Class probabilities.
        """
        return self.model.predict_proba(X)

    def save_model(self, filename=None):
        """
        Save the trained model to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the model. If None, a default name is used.

        Returns
        -------
        str
            Path to the saved model file.
        """
        if filename is None:
            filename = f"{self.model_type}_naive_bayes.joblib"

        filepath = os.path.join(MODELS_DIR, filename)

        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save the model
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        NaiveBayesModel
            Loaded model instance.
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            raise FileNotFoundError(f"Model file {filepath} not found")

        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)

    def get_class_probability(self, X):
        """
        Get class probabilities for visualization and analysis.

        Parameters
        ----------
        X : array-like
            Samples to get probabilities for.

        Returns
        -------
        array
            Array of class probabilities.
        """
        return self.predict_proba(X)

    def get_feature_importances(self, feature_names):
        """
        Get the feature importances for the model.

        Parameters
        ----------
        feature_names : array-like
            Names of the features.

        Returns
        -------
        dict
            Dictionary containing the important features for spam and ham.
        """
        if not hasattr(self.model, 'feature_log_prob_'):
            logger.error("Model does not support feature importance extraction")
            return None

        logger.info("Extracting feature importances")

        # For MultinomialNB and BernoulliNB
        if isinstance(self.model, (MultinomialNB, BernoulliNB, ComplementNB)):
            # Get log probabilities
            feature_log_prob = self.model.feature_log_prob_

            # Calculate log odds ratio
            log_odds_ratio = feature_log_prob[1] - feature_log_prob[0]

            # Get top features for spam and ham
            top_spam_indices = np.argsort(log_odds_ratio)[-20:][::-1]
            top_ham_indices = np.argsort(-log_odds_ratio)[-20:][::-1]

            # Create result dictionary
            result = {
                'spam': [(feature_names[i], float(np.exp(log_odds_ratio[i])))
                         for i in top_spam_indices],
                'ham': [(feature_names[i], float(np.exp(-log_odds_ratio[i])))
                        for i in top_ham_indices]
            }

            return result

        return None


class CustomNaiveBayes:
    """
    Custom implementation of Naive Bayes for educational purposes.

    This class implements the Naive Bayes algorithm from scratch, providing
    a clear understanding of the underlying mathematics.

    Parameters
    ----------
    alpha : float, optional
        Laplace smoothing parameter.
    fit_prior : bool, optional
        Whether to learn class prior probabilities.
    """

    def __init__(self, alpha=1.0, fit_prior=True):
        """Initialize the CustomNaiveBayes."""
        self.alpha = alpha  # Laplace smoothing parameter
        self.fit_prior = fit_prior
        self.class_log_prior_ = None  # Log of class prior probabilities
        self.feature_log_prob_ = None  # Log of feature conditional probabilities
        self.classes_ = None  # Class labels
        self.n_features_ = None  # Number of features

        logger.info("Initialized CustomNaiveBayes")
        logger.debug(f"Parameters: alpha={alpha}, fit_prior={fit_prior}")

    def fit(self, X, y):
        """
        Fit the Naive Bayes model according to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self
            The fitted model.
        """
        # Convert X and y to numpy arrays if they're not already
        X = np.asarray(X)
        y = np.asarray(y)

        # Determine the unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape

        # Calculate class priors
        if self.fit_prior:
            class_counts = np.array([np.sum(y == c) for c in self.classes_])
            self.class_log_prior_ = np.log(class_counts / n_samples)
        else:
            self.class_log_prior_ = np.log(np.ones(n_classes) / n_classes)

        # Initialize feature log probabilities
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))

        # Calculate feature log probabilities for each class
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = np.sum(X_c, axis=0)

            # Apply Laplace smoothing
            smoothed_counts = feature_counts + self.alpha
            total_counts = np.sum(X_c) + self.alpha * self.n_features_

            # Calculate log probabilities
            self.feature_log_prob_[i] = np.log(smoothed_counts / total_counts)

        logger.info("Model training complete")
        return self

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        array, shape (n_samples, n_classes)
            Log-probability estimates.
        """
        return self._joint_log_likelihood(X)

    def predict_proba(self, X):
        """
        Return probability estimates for samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        array, shape (n_samples, n_classes)
            Probability estimates.
        """
        log_proba = self.predict_log_proba(X)
        # Exponentiate and normalize to get probabilities
        proba = np.exp(log_proba - np.max(log_proba, axis=1)[:, np.newaxis])
        proba /= np.sum(proba, axis=1)[:, np.newaxis]

        return proba

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        array, shape (n_samples,)
            Predicted class labels.
        """
        # Get log probabilities
        log_proba = self.predict_log_proba(X)

        # Predict the class with the highest log probability
        return self.classes_[np.argmax(log_proba, axis=1)]

    def _joint_log_likelihood(self, X):
        """
        Calculate the joint log likelihood for each class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        array, shape (n_samples, n_classes)
            Joint log likelihood for each sample and class.
        """
        X = np.asarray(X)

        # Check dimensions
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        # Calculate joint log likelihood
        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))

        for i, c in enumerate(self.classes_):
            # Prior probability
            joint_log_likelihood[:, i] = self.class_log_prior_[i]

            # Add log of feature probabilities
            joint_log_likelihood[:, i] += np.sum(X * self.feature_log_prob_[i], axis=1)

        return joint_log_likelihood

    def save_model(self, filename=None):
        """
        Save the trained model to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the model. If None, a default name is used.

        Returns
        -------
        str
            Path to the saved model file.
        """
        if filename is None:
            filename = "custom_naive_bayes.joblib"

        filepath = os.path.join(MODELS_DIR, filename)

        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save the model
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        CustomNaiveBayes
            Loaded model instance.
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            raise FileNotFoundError(f"Model file {filepath} not found")

        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)