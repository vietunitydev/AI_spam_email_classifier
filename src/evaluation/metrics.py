"""
Evaluation metrics for the spam classifier.

This module provides functions for evaluating the performance of the spam classifier.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from ..utils.logger import logger


class ModelEvaluator:
    """
    Class for evaluating spam classification models.

    This class provides methods for calculating evaluation metrics and
    visualizing model performance.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    feature_names : array-like, optional
        Names of the features.
    """

    def __init__(self, y_true, y_pred, y_prob=None, feature_names=None):
        """Initialize the ModelEvaluator."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.feature_names = feature_names

        logger.info("Initialized ModelEvaluator")

    def calculate_metrics(self):
        """
        Calculate evaluation metrics.

        Returns
        -------
        dict
            Dictionary containing the calculated metrics.
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1': f1_score(self.y_true, self.y_pred),
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred)
        }

        # Calculate ROC AUC if probabilities are available
        if self.y_prob is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr

            # Calculate Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
            metrics['avg_precision'] = average_precision_score(self.y_true, self.y_prob)
            metrics['precision_curve'] = precision
            metrics['recall_curve'] = recall

        # Calculate classification report
        metrics['classification_report'] = classification_report(self.y_true, self.y_pred, output_dict=True)

        logger.info("Calculated evaluation metrics")
        return metrics

    def print_metrics(self, metrics=None):
        """
        Print evaluation metrics.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary containing the calculated metrics. If None, metrics are calculated.
        """
        if metrics is None:
            metrics = self.calculate_metrics()

        print("\n===== EVALUATION METRICS =====")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        if 'avg_precision' in metrics:
            print(f"Average Precision: {metrics['avg_precision']:.4f}")

        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"True Negatives (TN): {cm[0][0]}")
        print(f"False Positives (FP): {cm[0][1]}")
        print(f"False Negatives (FN): {cm[1][0]}")
        print(f"True Positives (TP): {cm[1][1]}")

        print("\nClassification Report:")
        report = metrics['classification_report']
        df_report = pd.DataFrame(report).transpose()
        print(df_report)

    def plot_confusion_matrix(self, metrics=None):
        """
        Plot confusion matrix.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary containing the calculated metrics. If None, metrics are calculated.
        """
        if metrics is None:
            metrics = self.calculate_metrics()

        cm = metrics['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, metrics=None):
        """
        Plot ROC curve.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary containing the calculated metrics. If None, metrics are calculated.
        """
        if metrics is None:
            metrics = self.calculate_metrics()

        if 'roc_auc' not in metrics:
            logger.error("ROC curve cannot be plotted without prediction probabilities")
            return

        fpr = metrics['fpr']
        tpr = metrics['tpr']
        roc_auc = metrics['roc_auc']

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self, metrics=None):
        """
        Plot precision-recall curve.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary containing the calculated metrics. If None, metrics are calculated.
        """
        if metrics is None:
            metrics = self.calculate_metrics()

        if 'avg_precision' not in metrics:
            logger.error("Precision-Recall curve cannot be plotted without prediction probabilities")
            return

        precision = metrics['precision_curve']
        recall = metrics['recall_curve']
        avg_precision = metrics['avg_precision']

        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={avg_precision:.2f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model, top_n=20):
        """
        Plot feature importance.

        Parameters
        ----------
        model : object
            Trained model with get_feature_importances method.
        top_n : int, optional
            Number of top features to display.
        """
        if not hasattr(model, 'get_feature_importances'):
            logger.error("Model does not have get_feature_importances method")
            return

        if self.feature_names is None:
            logger.error("Feature names are required for feature importance visualization")
            return

        # Get feature importances
        try:
            importances = model.get_feature_importances(self.feature_names)
        except Exception as e:
            logger.error(f"Error getting feature importances: {e}")
            return

        if importances is None:
            logger.error("Failed to get feature importances")
            return

        # Plot feature importances for spam
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        spam_features = importances['spam'][:top_n]
        spam_features.reverse()  # For better visualization
        spam_words = [word for word, _ in spam_features]
        spam_values = [value for _, value in spam_features]

        plt.barh(spam_words, spam_values, color='red', alpha=0.7)
        plt.xlabel('Importance (odds ratio)')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Features Indicating Spam')
        plt.grid(True, alpha=0.3)

        # Plot feature importances for ham
        plt.subplot(2, 1, 2)
        ham_features = importances['ham'][:top_n]
        ham_features.reverse()  # For better visualization
        ham_words = [word for word, _ in ham_features]
        ham_values = [value for _, value in ham_features]

        plt.barh(ham_words, ham_values, color='blue', alpha=0.7)
        plt.xlabel('Importance (odds ratio)')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Features Indicating Ham')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_report(self, model=None):
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        model : object, optional
            Trained model for feature importance analysis.

        Returns
        -------
        dict
            Dictionary containing the evaluation report.
        """
        # Calculate metrics
        metrics = self.calculate_metrics()

        # Print metrics
        self.print_metrics(metrics)

        # Plot visualizations
        self.plot_confusion_matrix(metrics)

        if self.y_prob is not None:
            self.plot_roc_curve(metrics)
            self.plot_precision_recall_curve(metrics)

        # Plot feature importance if model is provided
        if model is not None and self.feature_names is not None:
            self.plot_feature_importance(model)

        # Return metrics
        return metrics