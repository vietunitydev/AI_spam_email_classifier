"""
Tests for the data module.
"""

import unittest
import os
import pandas as pd
from src.data.data_loader import EmailDataLoader
from src.data.preprocessor import EmailPreprocessor
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class TestDataLoader(unittest.TestCase):
    """Test EmailDataLoader class."""

    def setUp(self):
        """Set up test environment."""
        self.dataset_name = 'test_dataset'
        self.raw_dir = os.path.join(RAW_DATA_DIR, self.dataset_name)
        self.processed_dir = os.path.join(PROCESSED_DATA_DIR, self.dataset_name)

        # Create test directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'spam'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'ham'), exist_ok=True)

        # Create test data loader
        self.data_loader = EmailDataLoader(
            self.dataset_name,
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir
        )

    def tearDown(self):
        """Clean up test environment."""
        # Remove test directories
        if os.path.exists(self.raw_dir):
            for root, dirs, files in os.walk(self.raw_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.raw_dir)

        if os.path.exists(self.processed_dir):
            for root, dirs, files in os.walk(self.processed_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.processed_dir)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.data_loader.dataset_name, self.dataset_name)
        self.assertEqual(self.data_loader.raw_dir, self.raw_dir)
        self.assertEqual(self.data_loader.processed_dir, self.processed_dir)
        self.assertEqual(self.data_loader.spam_dir, os.path.join(self.processed_dir, 'spam'))
        self.assertEqual(self.data_loader.ham_dir, os.path.join(self.processed_dir, 'ham'))

    def test_save_to_csv(self):
        """Test save_to_csv method."""
        # Create a test DataFrame
        data = {
            'label': ['spam', 'ham', 'spam'],
            'subject': ['Test subject 1', 'Test subject 2', 'Test subject 3'],
            'body': ['Test body 1', 'Test body 2', 'Test body 3']
        }
        email_df = pd.DataFrame(data)

        # Save to CSV
        filepath = self.data_loader.save_to_csv(email_df, 'test_data.csv')

        # Check if file exists
        self.assertTrue(os.path.exists(filepath))

        # Check if file content is correct
        loaded_df = pd.read_csv(filepath)
        self.assertEqual(len(loaded_df), len(email_df))
        self.assertListEqual(list(loaded_df['label']), list(email_df['label']))
        self.assertListEqual(list(loaded_df['subject']), list(email_df['subject']))
        self.assertListEqual(list(loaded_df['body']), list(email_df['body']))


class TestPreprocessor(unittest.TestCase):
    """Test EmailPreprocessor class."""

    def setUp(self):
        """Set up test environment."""
        self.preprocessor = EmailPreprocessor()

        # Create test data
        self.text_samples = [
            "Hello, this is a test email with some numbers 123 and URL http://example.com",
            "Another test email with EMAIL@example.com and special characters !@#$%^&*()",
            "<html>This email has HTML tags</html> and some more text"
        ]

    def test_clean_text(self):
        """Test clean_text method."""
        for text in self.text_samples:
            cleaned_text = self.preprocessor.clean_text(text)

            # Check if text is converted to lowercase
            self.assertEqual(cleaned_text, cleaned_text.lower())

            # Check if URLs are removed
            self.assertNotIn('http', cleaned_text)

            # Check if email addresses are removed
            self.assertNotIn('@', cleaned_text)

            # Check if HTML tags are removed
            self.assertNotIn('<html>', cleaned_text)
            self.assertNotIn('</html>', cleaned_text)

            # Check if numbers are removed
            self.assertNotIn('123', cleaned_text)

            # Check if punctuation is removed
            for punct in '!@#$%^&*()':
                self.assertNotIn(punct, cleaned_text)

    def test_tokenize_text(self):
        """Test tokenize_text method."""
        for text in self.text_samples:
            tokens = self.preprocessor.tokenize_text(text)

            # Check if tokens is a list
            self.assertIsInstance(tokens, list)

            # Check if all tokens are strings
            for token in tokens:
                self.assertIsInstance(token, str)

            # Check if all tokens are lowercase
            for token in tokens:
                self.assertEqual(token, token.lower())

            # Check if all tokens have minimum length
            min_length = self.preprocessor.config.get('min_token_length', 2)
            for token in tokens:
                self.assertGreaterEqual(len(token), min_length)

    def test_process_emails(self):
        """Test process_emails method."""
        # Create a test DataFrame
        data = {
            'label': ['spam', 'ham', 'spam'],
            'body': self.text_samples
        }
        email_df = pd.DataFrame(data)

        # Process emails
        processed_df = self.preprocessor.process_emails(email_df)

        # Check if processed_text column is created
        self.assertIn('processed_text', processed_df.columns)

        # Check if token_count column is created
        self.assertIn('token_count', processed_df.columns)

        # Check if all processed texts are strings
        for text in processed_df['processed_text']:
            self.assertIsInstance(text, str)

        # Check if all token counts are integers
        for count in processed_df['token_count']:
            self.assertIsInstance(count, int)


if __name__ == '__main__':
    unittest.main()