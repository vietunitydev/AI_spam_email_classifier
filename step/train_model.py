"""
Model training script for the spam classifier.
"""

import os
import pickle
import sys
import time
from src.models.custom_naive_bayes import CustomNaiveBayes


def train_model():
    print("\n" + "=" * 50)
    print("TRAINING CUSTOM NAIVE BAYES MODEL")
    print("=" * 50 + "\n")

    alpha = 1.0
    fit_prior = True

    input_file = 'preprocessed_data.pkl'
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please run preprocess_data.py first.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded preprocessed data ({data['X_train'].shape[0]} training samples).")
    print(f"Features: {data['X_train'].shape[1]}")

    print(f"Initializing Custom Naive Bayes with alpha={alpha}, fit_prior={fit_prior}")
    model = CustomNaiveBayes(alpha=alpha, fit_prior=fit_prior)

    X_train = data['X_train']
    X_test = data['X_test']

    if hasattr(X_train, 'toarray'):
        print("Converting sparse matrix to dense array...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    print(f"Training data shape: {X_train.shape}")

    print("Training Custom Naive Bayes model...")
    start_time = time.time()
    model.fit(X_train, data['y_train'])
    train_time = time.time() - start_time

    print(f"Model trained in {train_time:.2f} seconds")

    print(f"Classes: {len(model.classes_)}")
    print(f"Class labels: {model.classes_}")
    print(f"Class log priors: {model.class_log_prior_}")

    print("\nTesting prediction speed...")
    sample_size = min(100, X_test.shape[0])
    start_time = time.time()
    predictions = model.predict(X_test[:sample_size])
    predict_time = time.time() - start_time

    print(f"Average prediction time: {(predict_time / sample_size) * 1000:.2f} ms/email")

    print(f"\nSample predictions (first 10):")
    sample_predictions = model.predict(X_test[:10])
    sample_probabilities = model.predict_proba(X_test[:10])

    for i in range(min(10, len(sample_predictions))):
        pred_label = "SPAM" if sample_predictions[i] == 1 else "HAM"
        spam_prob = sample_probabilities[i][1] * 100
        print(f"  Sample {i+1}: {pred_label} (Spam probability: {spam_prob:.2f}%)")

    model_file = "custom_naive_bayes_model.joblib"
    model_path = model.save_model(model_file)
    print(f"Model saved to: {model_path}")

    data['X_train'] = X_train
    data['X_test'] = X_test

    output_file = 'trained_model_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'model_path': model_path,
            'model_type': 'custom',
            'model_params': {'alpha': alpha, 'fit_prior': fit_prior},
            'data': data
        }, f)

    print(f"Saved model information to {output_file}")
    return model, data


if __name__ == "__main__":
    train_model()