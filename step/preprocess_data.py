"""
Data preprocessing script for the spam classifier.
"""

import os
import pickle
import sys
from src.data.preprocessor import EmailPreprocessor


def preprocess_data():
    print("\n" + "=" * 50)
    print("PREPROCESSING DATA")
    print("=" * 50 + "\n")

    input_file = 'email_data.pkl'
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please run load_data.py first.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        email_df = pickle.load(f)

    print(f"Loaded {len(email_df)} emails.")

    print("Preprocessing data using TF-IDF vectorization")
    preprocessor = EmailPreprocessor()
    data = preprocessor.prepare_data(email_df)

    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Features: {data['X_train'].shape[1]}")

    print("\nSample features:")
    for i, feature in enumerate(data['feature_names'][:10]):
        print(f"  {i + 1}. {feature}")

    output_file = 'preprocessed_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved preprocessed data to {output_file}")
    return data


if __name__ == "__main__":
    preprocess_data()