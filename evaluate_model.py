"""
Model evaluation script for the spam classifier.
"""

import os
import pickle
import sys
import time
import numpy as np
from src.models.custom_naive_bayes import CustomNaiveBayes
from src.evaluation.model_evaluator import ModelEvaluator


def evaluate_model():
    """
    Evaluate the trained Custom Naive Bayes model.
    """
    print("\n" + "=" * 50)
    print("EVALUATING CUSTOM NAIVE BAYES MODEL")
    print("=" * 50 + "\n")

    # Load model and data
    input_file = 'trained_model_data.pkl'
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Please run train_model.py first.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        model_data = pickle.load(f)

    model_path = model_data['model_path']
    model_type = model_data['model_type']
    model_params = model_data.get('model_params', {})
    data = model_data['data']

    print(f"Loaded model information ({model_type}) and test data.")
    print(f"Model parameters: {model_params}")

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = CustomNaiveBayes.load_model(model_path)

    # Display model information
    print(f"Model loaded:")
    print(f"  - Classes: {len(model.classes_)}")
    print(f"  - Class labels: {model.classes_}")
    print(f"  - Features: {model.n_features_}")
    print(f"  - Alpha (Laplace smoothing): {model.alpha}")

    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    y_test = data['y_test']
    X_test = data['X_test']

    print(f"Test set size: {X_test.shape}")

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    # Calculate probabilities
    start_time = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of spam class
    proba_time = time.time() - start_time

    print(f"Prediction time: {predict_time:.4f} seconds")
    print(f"Probability calculation time: {proba_time:.4f} seconds")

    # Create evaluator
    evaluator = ModelEvaluator(y_test, y_pred, y_prob, data['feature_names'])

    # Generate comprehensive evaluation report
    print("\nGenerating comprehensive evaluation report...")

    # Get sample texts for misclassification analysis (if available)
    sample_texts = None
    if 'processed_df' in data:
        # Get test indices to map back to original texts
        test_indices = data.get('test_indices', None)
        if test_indices is not None:
            sample_texts = data['processed_df'].iloc[test_indices]['body'].values
        else:
            # If no test indices available, use processed text
            sample_texts = data['processed_df']['processed_text'].values[-len(y_test):]

    # Generate comprehensive report
    evaluation_results = evaluator.generate_comprehensive_report(
        model=model,
        sample_texts=sample_texts,
        save_plots=True
    )

    # Additional analysis specific to Custom Naive Bayes
    print("\nCUSTOM NAIVE BAYES SPECIFIC ANALYSIS:")
    analyze_feature_importance(model, data['feature_names'])

    # Performance statistics
    print(f"\nPERFORMANCE STATISTICS:")
    print(f"Total prediction time: {predict_time:.4f} seconds")
    print(f"Average prediction time: {(predict_time / len(y_test)) * 1000:.4f} ms/email")
    print(f"Predictions per second: {len(y_test) / predict_time:.0f}")

    # Probability distribution analysis
    spam_probs = y_prob[y_test == 1]  # Probabilities for actual spam emails
    ham_probs = y_prob[y_test == 0]   # Probabilities for actual ham emails

    print(f"\nPROBABILITY DISTRIBUTION ANALYSIS:")
    print(f"Actual SPAM emails - Average probability: {np.mean(spam_probs):.4f}")
    print(f"                   - Std deviation: {np.std(spam_probs):.4f}")
    print(f"                   - Min probability: {np.min(spam_probs):.4f}")
    print(f"                   - Max probability: {np.max(spam_probs):.4f}")

    print(f"Actual HAM emails  - Average probability: {np.mean(ham_probs):.4f}")
    print(f"                   - Std deviation: {np.std(ham_probs):.4f}")
    print(f"                   - Min probability: {np.min(ham_probs):.4f}")
    print(f"                   - Max probability: {np.max(ham_probs):.4f}")

    # Save evaluation results
    output_file = 'evaluation_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'evaluation_results': evaluation_results,
            'model_type': model_type,
            'model_path': model_path,
            'model_params': model_params,
            'performance_stats': {
                'prediction_time': predict_time,
                'probability_time': proba_time,
                'avg_prediction_time_ms': (predict_time / len(y_test)) * 1000,
                'predictions_per_second': len(y_test) / predict_time
            },
            'probability_analysis': {
                'spam_probs_stats': {
                    'mean': np.mean(spam_probs),
                    'std': np.std(spam_probs),
                    'min': np.min(spam_probs),
                    'max': np.max(spam_probs)
                },
                'ham_probs_stats': {
                    'mean': np.mean(ham_probs),
                    'std': np.std(ham_probs),
                    'min': np.min(ham_probs),
                    'max': np.max(ham_probs)
                }
            }
        }, f)

    print(f"\nEvaluation results saved to {output_file}")

    return evaluation_results, model


def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Analyze feature importance for Custom Naive Bayes.

    Parameters
    ----------
    model : CustomNaiveBayes
        Trained model
    feature_names : array-like
        Feature names
    top_n : int
        Number of top features to analyze
    """
    if not hasattr(model, 'feature_log_prob_'):
        print("Model does not support feature importance analysis.")
        return

    print(f"\nFEATURE IMPORTANCE ANALYSIS:")

    # Calculate log odds ratio
    log_odds_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]

    # Find most important features for spam
    top_spam_indices = np.argsort(log_odds_ratio)[-top_n:][::-1]

    # Find most important features for ham
    top_ham_indices = np.argsort(log_odds_ratio)[:top_n]

    print(f"\nTop {top_n} words most indicative of SPAM:")
    for i, idx in enumerate(top_spam_indices):
        word = feature_names[idx]
        score = log_odds_ratio[idx]
        odds_ratio = np.exp(score)
        print(f"  {i+1:2d}. {word:20s} (log odds: {score:8.4f}, odds ratio: {odds_ratio:8.2f})")

    print(f"\nTop {top_n} words most indicative of HAM:")
    for i, idx in enumerate(top_ham_indices):
        word = feature_names[idx]
        score = log_odds_ratio[idx]
        odds_ratio = np.exp(-score)  # Inverse for ham
        print(f"  {i+1:2d}. {word:20s} (log odds: {score:8.4f}, odds ratio: {odds_ratio:8.2f})")

    # Plot feature importance
    plot_feature_importance(feature_names, log_odds_ratio, top_n)


def plot_feature_importance(feature_names, log_odds_ratio, top_n=15):
    """
    Plot feature importance for both spam and ham.

    Parameters
    ----------
    feature_names : array-like
        Feature names
    log_odds_ratio : array-like
        Log odds ratios
    top_n : int
        Number of features to plot
    """
    import matplotlib.pyplot as plt

    # Top spam features
    top_spam_indices = np.argsort(log_odds_ratio)[-top_n:]
    spam_words = [feature_names[i] for i in top_spam_indices]
    spam_scores = [log_odds_ratio[i] for i in top_spam_indices]

    # Top ham features
    top_ham_indices = np.argsort(log_odds_ratio)[:top_n]
    ham_words = [feature_names[i] for i in top_ham_indices]
    ham_scores = [abs(log_odds_ratio[i]) for i in top_ham_indices]  # Absolute value for visualization

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Spam features plot
    ax1.barh(range(len(spam_words)), spam_scores, color='red', alpha=0.7)
    ax1.set_yticks(range(len(spam_words)))
    ax1.set_yticklabels(spam_words)
    ax1.set_xlabel('Log Odds Ratio')
    ax1.set_title(f'Top {top_n} Features Indicating SPAM')
    ax1.grid(True, alpha=0.3)

    # Ham features plot
    ax2.barh(range(len(ham_words)), ham_scores, color='green', alpha=0.7)
    ax2.set_yticks(range(len(ham_words)))
    ax2.set_yticklabels(ham_words)
    ax2.set_xlabel('|Log Odds Ratio|')
    ax2.set_title(f'Top {top_n} Features Indicating HAM')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Feature importance plot saved to feature_importance.png")


if __name__ == "__main__":
    evaluate_model()