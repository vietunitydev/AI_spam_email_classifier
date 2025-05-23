"""
Custom Naive Bayes implementation for spam classification.
"""

import numpy as np
import joblib
import os
from src.utils.logger import logger
from src.utils.config import MODELS_DIR


class CustomNaiveBayes:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.n_features_ = None

        logger.info(f"Initialized CustomNaiveBayes with alpha={alpha}, fit_prior={fit_prior}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape

        if self.fit_prior:
            class_counts = np.array([np.sum(y == c) for c in self.classes_])
            self.class_log_prior_ = np.log(class_counts / n_samples)
        else:
            self.class_log_prior_ = np.log(np.ones(n_classes) / n_classes)

        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = np.sum(X_c, axis=0)

            smoothed_counts = feature_counts + self.alpha
            total_counts = np.sum(X_c) + self.alpha * self.n_features_

            self.feature_log_prob_[i] = np.log(smoothed_counts / total_counts)

        logger.info("Model training complete")
        return self

    def predict_log_proba(self, X):
        return self._joint_log_likelihood(X)

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        proba = np.exp(log_proba - np.max(log_proba, axis=1)[:, np.newaxis])
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        return proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def _joint_log_likelihood(self, X):
        X = np.asarray(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))

        for i, c in enumerate(self.classes_):
            joint_log_likelihood[:, i] = self.class_log_prior_[i]
            joint_log_likelihood[:, i] += np.sum(X * self.feature_log_prob_[i], axis=1)

        return joint_log_likelihood

    def save_model(self, filename=None):
        if filename is None:
            filename = "custom_naive_bayes.joblib"

        filepath = os.path.join(MODELS_DIR, filename)
        os.makedirs(MODELS_DIR, exist_ok=True)

        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    @classmethod
    def load_model(cls, filepath):
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            raise FileNotFoundError(f"Model file {filepath} not found")

        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)