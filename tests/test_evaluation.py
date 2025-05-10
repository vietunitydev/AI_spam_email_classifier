"""
Tests for the evaluation module.
"""

import unittest
import numpy as np
from src.evaluation.metrics import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class."""

    def setUp(self):
        """Set up test environment."""
        # Create a simple dataset for testing
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
        self.y_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9, 0.4, 0.8, 0.7])
        self.feature_names = [f"feature_{i}" for i in range(10)]

        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            self.y_true, self.y_pred, self.y_prob, self.feature_names
        )

    def test_init(self):
        """Test initialization."""
        self.assertTrue(np.array_equal(self.evaluator.y_true, self.y_true))
        self.assertTrue(np.array_equal(self.evaluator.y_pred, self.y_pred))
        self.assertTrue(np.array_equal(self.evaluator.y_prob, self.y_prob))
        self.assertEqual(self.evaluator.feature_names, self.feature_names)

    def test_calculate_metrics(self):
        """Test calculate_metrics method."""
        metrics = self.evaluator.calculate_metrics()

        # Check if all required metrics are calculated
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix',
            'roc_auc', 'fpr', 'tpr', 'avg_precision', 'precision_curve',
            'recall_curve', 'classification_report'
        ]

        for metric in required_metrics:
            self.assertIn(metric, metrics)

        # Check if metrics are of the correct type
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['precision'], float)
        self.assertIsInstance(metrics['recall'], float)
        self.assertIsInstance(metrics['f1'], float)
        self.assertIsInstance(metrics['roc_auc'], float)
        self.assertIsInstance(metrics['avg_precision'], float)

        # Check if confusion matrix is correct
        cm = metrics['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm[0, 0], 3)  # True Negatives
        self.assertEqual(cm[0, 1], 2)  # False Positives
        self.assertEqual(cm[1, 0], 1)  # False Negatives
        self.assertEqual(cm[1, 1], 4)  # True Positives

        # Check if accuracy is correctly calculated
        self.assertAlmostEqual(metrics['accuracy'], 0.7)

        # Check if precision is correctly calculated
        self.assertAlmostEqual(metrics['precision'], 0.6666666666666666)

        # Check if recall is correctly calculated
        self.assertAlmostEqual(metrics['recall'], 0.8)

        # Check if F1 is correctly calculated
        self.assertAlmostEqual(metrics['f1'], 0.7272727272727273)

    def test_print_metrics(self):
        """Test print_metrics method."""
        # This is a visual test, just check if it runs without errors
        try:
            self.evaluator.print_metrics()
            passed = True
        except Exception as e:
            passed = False
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()