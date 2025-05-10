"""
Tests for the models module.
"""

import unittest
import os
import numpy as np
import tempfile
from src.models.naive_bayes import NaiveBayesModel, CustomNaiveBayes

class TestNaiveBayesModel(unittest.TestCase):
    """Test NaiveBayesModel class."""

    def setUp(self):
        """Set up test environment."""
        # Create a simple dataset for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)

        # Initialize models
        self.model_types = ['multinomial', 'bernoulli', 'complement']
        self.models = {
            model_type: NaiveBayesModel(model_type=model_type)
            for model_type in self.model_types
        }

        # Initialize custom model
        self.custom_model = CustomNaiveBayes()

    def test_init(self):
        """Test initialization."""
        for model_type, model in self.models.items():
            self.assertEqual(model.model_type, model_type)

    def test_fit_predict(self):
        """Test fit and predict methods."""
        for model_type, model in self.models.items():
            # Train model
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test)

            # Check prediction shape
            self.assertEqual(y_pred.shape, self.y_test.shape)

            # Check if all predictions are 0 or 1
            self.assertTrue(np.all(np.logical_or(y_pred == 0, y_pred == 1)))

    def test_predict_proba(self):
        """Test predict_proba method."""
        for model_type, model in self.models.items():
            # Train model
            model.fit(self.X_train, self.y_train)

            # Predict probabilities
            y_prob = model.predict_proba(self.X_test)

            # Check shape
            self.assertEqual(y_prob.shape, (self.X_test.shape[0], 2))

            # Check if probabilities sum to 1
            self.assertTrue(np.allclose(np.sum(y_prob, axis=1), 1.0))

            # Check if all probabilities are between 0 and 1
            self.assertTrue(np.all(y_prob >= 0))
            self.assertTrue(np.all(y_prob <= 1))

    def test_save_load_model(self):
        """Test save_model and load_model methods."""
        for model_type, model in self.models.items():
            # Train model
            model.fit(self.X_train, self.y_train)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp:
                temp_path = temp.name

            try:
                # Save model
                model.save_model(temp_path)

                # Check if file exists
                self.assertTrue(os.path.exists(temp_path))

                # Load model
                loaded_model = NaiveBayesModel.load_model(temp_path)

                # Check if loaded model is a NaiveBayesModel
                self.assertIsInstance(loaded_model, NaiveBayesModel)

                # Check if model type is preserved
                self.assertEqual(loaded_model.model_type, model_type)

                # Check if predictions are the same
                y_pred_original = model.predict(self.X_test)
                y_pred_loaded = loaded_model.predict(self.X_test)
                self.assertTrue(np.array_equal(y_pred_original, y_pred_loaded))
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_custom_naive_bayes(self):
        """Test CustomNaiveBayes class."""
        # Train model
        self.custom_model.fit(self.X_train, self.y_train)

        # Predict
        y_pred = self.custom_model.predict(self.X_test)

        # Check prediction shape
        self.assertEqual(y_pred.shape, self.y_test.shape)

        # Check if all predictions are 0 or 1
        self.assertTrue(np.all(np.logical_or(y_pred == 0, y_pred == 1)))

        # Predict probabilities
        y_prob = self.custom_model.predict_proba(self.X_test)

        # Check shape
        self.assertEqual(y_prob.shape, (self.X_test.shape[0], 2))

        # Check if probabilities sum to 1
        self.assertTrue(np.allclose(np.sum(y_prob, axis=1), 1.0))

        # Check if all probabilities are between 0 and 1
        self.assertTrue(np.all(y_prob >= 0))
        self.assertTrue(np.all(y_prob <= 1))

        # Test save and load
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp:
            temp_path = temp.name

        try:
            # Save model
            self.custom_model.save_model(temp_path)

            # Check if file exists
            self.assertTrue(os.path.exists(temp_path))

            # Load model
            loaded_model = CustomNaiveBayes.load_model(temp_path)

            # Check if loaded model is a CustomNaiveBayes
            self.assertIsInstance(loaded_model, CustomNaiveBayes)

            # Check if predictions are the same
            y_pred_original = self.custom_model.predict(self.X_test)
            y_pred_loaded = loaded_model.predict(self.X_test)
            self.assertTrue(np.array_equal(y_pred_original, y_pred_loaded))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()