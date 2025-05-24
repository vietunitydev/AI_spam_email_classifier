"""
Model evaluation module for the spam classifier.
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
from src.utils.logger import logger


class ModelEvaluator:
    """
    Class for evaluating spam classification models.
    """

    def __init__(self, y_true, y_pred, y_prob=None, feature_names=None):
        """
        Initialize the ModelEvaluator.

        Parameters
        ----------
        y_true : array-like
            True labels (0 for ham, 1 for spam)
        y_pred : array-like
            Predicted labels (0 for ham, 1 for spam)
        y_prob : array-like, optional
            Predicted probabilities for spam class
        feature_names : array-like, optional
            Names of the features
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.feature_names = feature_names

        logger.info("Initialized ModelEvaluator")

    def calculate_basic_metrics(self):
        """
        Calculate basic classification metrics.

        Returns
        -------
        dict
            Dictionary containing basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred),
            'recall': recall_score(self.y_true, self.y_pred),
            'f1_score': f1_score(self.y_true, self.y_pred),
            'specificity': self._calculate_specificity(),
        }

        logger.info("Calculated basic metrics")
        return metrics

    def calculate_advanced_metrics(self):
        """
        Calculate advanced metrics including ROC and PR curves.

        Returns
        -------
        dict
            Dictionary containing advanced metrics
        """
        if self.y_prob is None:
            logger.warning("No probabilities provided, skipping advanced metrics")
            return {}

        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(self.y_true, self.y_prob)
        avg_precision = average_precision_score(self.y_true, self.y_prob)

        metrics = {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'precision_curve': precision,
            'recall_curve': recall,
            'pr_thresholds': pr_thresholds
        }

        logger.info("Calculated advanced metrics")
        return metrics

    def get_confusion_matrix(self):
        """
        Get confusion matrix and related statistics.

        Returns
        -------
        dict
            Dictionary containing confusion matrix info
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        tn, fp, fn, tp = cm.ravel()

        return {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'total_samples': len(self.y_true),
            'spam_samples': np.sum(self.y_true == 1),
            'ham_samples': np.sum(self.y_true == 0)
        }

    def get_classification_report(self):
        """
        Get detailed classification report.

        Returns
        -------
        dict
            Classification report as dictionary
        """
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=['Ham', 'Spam'],
            output_dict=True
        )

    def _calculate_specificity(self):
        """Calculate specificity (True Negative Rate)."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def print_evaluation_summary(self):
        """
        Print comprehensive evaluation summary.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)

        # Basic metrics
        basic_metrics = self.calculate_basic_metrics()
        print(f"\nBASIC METRICS:")
        print(f"  Accuracy:    {basic_metrics['accuracy']:.4f}")
        print(f"  Precision:   {basic_metrics['precision']:.4f}")
        print(f"  Recall:      {basic_metrics['recall']:.4f}")
        print(f"  F1-Score:    {basic_metrics['f1_score']:.4f}")
        print(f"  Specificity: {basic_metrics['specificity']:.4f}")

        # Advanced metrics
        if self.y_prob is not None:
            advanced_metrics = self.calculate_advanced_metrics()
            print(f"\nADVANCED METRICS:")
            print(f"  ROC AUC:           {advanced_metrics['roc_auc']:.4f}")
            print(f"  Average Precision: {advanced_metrics['avg_precision']:.4f}")

        # Confusion matrix
        cm_info = self.get_confusion_matrix()
        print(f"\nCONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                Ham   Spam")
        print(f"  Actual Ham    {cm_info['true_negatives']:4d}   {cm_info['false_positives']:4d}")
        print(f"         Spam   {cm_info['false_negatives']:4d}   {cm_info['true_positives']:4d}")

        print(f"\nSAMPLE DISTRIBUTION:")
        print(f"  Total samples: {cm_info['total_samples']}")
        print(
            f"  Ham samples:   {cm_info['ham_samples']} ({cm_info['ham_samples'] / cm_info['total_samples'] * 100:.1f}%)")
        print(
            f"  Spam samples:  {cm_info['spam_samples']} ({cm_info['spam_samples'] / cm_info['total_samples'] * 100:.1f}%)")

        # Classification report
        report = self.get_classification_report()
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 50)
        for class_name in ['Ham', 'Spam']:
            metrics = report[class_name]
            print(f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1-score']:<10.4f} {metrics['support']:<8.0f}")

        # Macro and weighted averages
        print("-" * 50)
        for avg_type in ['macro avg', 'weighted avg']:
            metrics = report[avg_type]
            print(f"{avg_type:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1-score']:<10.4f} {metrics['support']:<8.0f}")

    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix heatmap.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    cbar_kws={'label': 'Count'})

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j] / total * 100:.1f}%)',
                         ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot ROC curve.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.y_prob is None:
            print("Cannot plot ROC curve without prediction probabilities")
            return

        advanced_metrics = self.calculate_advanced_metrics()
        fpr = advanced_metrics['fpr']
        tpr = advanced_metrics['tpr']
        roc_auc = advanced_metrics['roc_auc']

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot Precision-Recall curve.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.y_prob is None:
            print("Cannot plot PR curve without prediction probabilities")
            return

        advanced_metrics = self.calculate_advanced_metrics()
        precision = advanced_metrics['precision_curve']
        recall = advanced_metrics['recall_curve']
        avg_precision = advanced_metrics['avg_precision']

        plt.figure(figsize=figsize)
        plt.step(recall, precision, color='b', alpha=0.8, where='post',
                 label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')

        # Baseline (random classifier)
        spam_ratio = np.mean(self.y_true)
        plt.axhline(y=spam_ratio, color='navy', linestyle='--',
                    label=f'Random Classifier (AP = {spam_ratio:.3f})')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")

        plt.show()

    def plot_metrics_comparison(self, figsize=(10, 6), save_path=None):
        """
        Plot comparison of different metrics.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        basic_metrics = self.calculate_basic_metrics()

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        metrics_values = [
            basic_metrics['accuracy'],
            basic_metrics['precision'],
            basic_metrics['recall'],
            basic_metrics['f1_score'],
            basic_metrics['specificity']
        ]

        plt.figure(figsize=figsize)
        bars = plt.bar(metrics_names, metrics_values,
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'],
                       alpha=0.8, edgecolor='black', linewidth=1)

        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")

        plt.show()

    def analyze_misclassifications(self, sample_texts=None, top_n=10):
        """
        Analyze misclassified samples.

        Parameters
        ----------
        sample_texts : array-like, optional
            Original text samples for analysis
        top_n : int
            Number of top misclassifications to show

        Returns
        -------
        dict
            Analysis of misclassifications
        """
        # Find misclassified indices
        misclassified_mask = self.y_true != self.y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return {}

        # Separate false positives and false negatives
        false_positives = []  # Ham predicted as Spam
        false_negatives = []  # Spam predicted as Ham

        for idx in misclassified_indices:
            if self.y_true[idx] == 0 and self.y_pred[idx] == 1:
                false_positives.append(idx)
            elif self.y_true[idx] == 1 and self.y_pred[idx] == 0:
                false_negatives.append(idx)

        print(f"\nMISCLASSIFICATION ANALYSIS:")
        print(f"Total misclassifications: {len(misclassified_indices)}")
        print(f"False Positives (Ham → Spam): {len(false_positives)}")
        print(f"False Negatives (Spam → Ham): {len(false_negatives)}")

        analysis = {
            'total_misclassified': len(misclassified_indices),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'fp_indices': false_positives[:top_n],
            'fn_indices': false_negatives[:top_n]
        }

        # If probabilities are available, show confidence for misclassifications
        if self.y_prob is not None:
            print(f"\nTop {min(top_n, len(false_positives))} False Positives (with confidence):")
            fp_probs = [(idx, self.y_prob[idx]) for idx in false_positives]
            fp_probs.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence

            for i, (idx, prob) in enumerate(fp_probs[:top_n]):
                print(f"  {i + 1}. Index {idx}: Confidence {prob * 100:.1f}%")
                if sample_texts is not None and idx < len(sample_texts):
                    text_preview = str(sample_texts[idx])[:100] + "..." if len(str(sample_texts[idx])) > 100 else str(
                        sample_texts[idx])
                    print(f"      Text: {text_preview}")

            print(f"\nTop {min(top_n, len(false_negatives))} False Negatives (with confidence):")
            fn_probs = [(idx, self.y_prob[idx]) for idx in false_negatives]
            fn_probs.sort(key=lambda x: x[1])  # Sort by confidence (ascending for FN)

            for i, (idx, prob) in enumerate(fn_probs[:top_n]):
                print(f"  {i + 1}. Index {idx}: Confidence {prob * 100:.1f}%")
                if sample_texts is not None and idx < len(sample_texts):
                    text_preview = str(sample_texts[idx])[:100] + "..." if len(str(sample_texts[idx])) > 100 else str(
                        sample_texts[idx])
                    print(f"      Text: {text_preview}")

        return analysis

    def generate_comprehensive_report(self, model=None, sample_texts=None, save_plots=True):
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        model : object, optional
            The trained model for additional analysis
        sample_texts : array-like, optional
            Original text samples
        save_plots : bool
            Whether to save plots to files

        Returns
        -------
        dict
            Complete evaluation results
        """
        print("Generating comprehensive evaluation report...")

        # Print summary
        self.print_evaluation_summary()

        # Plot visualizations
        if save_plots:
            self.plot_confusion_matrix(save_path="confusion_matrix.png")
            self.plot_metrics_comparison(save_path="metrics_comparison.png")

            if self.y_prob is not None:
                self.plot_roc_curve(save_path="roc_curve.png")
                self.plot_precision_recall_curve(save_path="pr_curve.png")
        else:
            self.plot_confusion_matrix()
            self.plot_metrics_comparison()

            if self.y_prob is not None:
                self.plot_roc_curve()
                self.plot_precision_recall_curve()

        # Analyze misclassifications
        misclass_analysis = self.analyze_misclassifications(sample_texts)

        # Compile all results
        results = {
            'basic_metrics': self.calculate_basic_metrics(),
            'confusion_matrix_info': self.get_confusion_matrix(),
            'classification_report': self.get_classification_report(),
            'misclassification_analysis': misclass_analysis
        }

        if self.y_prob is not None:
            results['advanced_metrics'] = self.calculate_advanced_metrics()

        print("\nComprehensive evaluation report completed!")
        return results