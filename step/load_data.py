"""
Data loading script for the spam classifier.
"""

import os
import pickle
import sys
from src.data.data_loader import EmailDataLoader


def load_data():
    print("\n" + "=" * 50)
    print("LOADING AND PROCESSING DATA")
    print("=" * 50 + "\n")

    dataset_name = 'enron'
    data_loader = EmailDataLoader(dataset_name)

    print("Processing dataset...")
    email_df = data_loader.process_dataset(limit=None)

    if email_df is None or len(email_df) == 0:
        print("Cannot load data. Please check configuration.")
        sys.exit(1)

    print(f"Loaded {len(email_df)} emails ({email_df['label'].value_counts()['spam']} spam, {email_df['label'].value_counts()['ham']} ham)")

    output_file = 'email_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(email_df, f)

    print(f"Saved data to {output_file}")
    return email_df


if __name__ == "__main__":
    load_data()