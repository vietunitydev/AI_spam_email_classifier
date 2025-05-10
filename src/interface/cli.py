"""
Command-line interface for the spam classifier.

This module provides a command-line interface for training and using the spam classifier.
"""

import os
import argparse
import pandas as pd
from ..data.data_loader import EmailDataLoader
from ..data.preprocessor import EmailPreprocessor
from ..models.naive_bayes import NaiveBayesModel, CustomNaiveBayes
from ..evaluation.metrics import ModelEvaluator
from ..utils.logger import logger
from ..utils.config import load_config, MODELS_DIR, PROCESSED_DATA_DIR


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Spam Classifier CLI')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Parser for 'download' command
    download_parser = subparsers.add_parser('download', help='Download and process a dataset')
    download_parser.add_argument('--dataset', choices=['spamassassin', 'enron'], default='spamassassin',
                                 help='Dataset to download')
    download_parser.add_argument('--limit', type=int, default=None,
                                 help='Limit the number of emails to process')

    # Parser for 'train' command
    train_parser = subparsers.add_parser('train', help='Train a Naive Bayes model')
    train_parser.add_argument('--dataset', choices=['spamassassin', 'enron'], default='spamassassin',
                              help='Dataset to use for training')
    train_parser.add_argument('--model-type', choices=['multinomial', 'bernoulli', 'gaussian', 'complement', 'custom'],
                              default='multinomial', help='Type of Naive Bayes model')
    train_parser.add_argument('--vectorizer', choices=['count', 'tfidf', 'binary'],
                              default='tfidf', help='Type of vectorizer')
    train_parser.add_argument('--config', type=str, default=None,
                              help='Path to configuration file')
    train_parser.add_argument('--output', type=str, default=None,
                              help='Name of the output model file')

    # Parser for 'evaluate' command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                 help='Path to the trained model')
    evaluate_parser.add_argument('--dataset', choices=['spamassassin', 'enron'], default='spamassassin',
                                 help='Dataset to use for evaluation')

    # Parser for 'classify' command
    classify_parser = subparsers.add_parser('classify', help='Classify a single email')
    classify_parser.add_argument('--model', type=str, required=True,
                                 help='Path to the trained model')
    classify_parser.add_argument('--email', type=str, required=True,
                                 help='Path to the email file to classify')

    return parser.parse_args()


def download_command(args):
    """
    Execute the 'download' command.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info(f"Downloading and processing {args.dataset} dataset")

    # Create data loader
    data_loader = EmailDataLoader(args.dataset)

    # Download and process dataset
    email_df = data_loader.process_dataset(limit=args.limit)

    if email_df is not None:
        logger.info(f"Successfully processed {len(email_df)} emails")
    else:
        logger.error("Failed to process dataset")


def train_command(args):
    """
    Execute the 'train' command.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info(f"Training {args.model_type} Naive Bayes model on {args.dataset} dataset")

    # Load configuration
    config = load_config(args.config)

    # Create data loader
    data_loader = EmailDataLoader(args.dataset)

    # Check if dataset is already processed
    csv_path = os.path.join(PROCESSED_DATA_DIR, args.dataset, 'email_data.csv')
    if os.path.exists(csv_path):
        logger.info(f"Loading processed data from {csv_path}")
        email_df = data_loader.load_from_csv()
    else:
        logger.info("Processed data not found, processing dataset")
        email_df = data_loader.process_dataset()

    if email_df is None:
        logger.error("Failed to load or process dataset")
        return

    # Create preprocessor
    preprocessor = EmailPreprocessor(config=config.get('preprocessing_params'))

    # Prepare data
    data = preprocessor.prepare_data(
        email_df, vectorizer_type=args.vectorizer, config=config.get('training_params')
    )

    # Create and train model
    if args.model_type == 'custom':
        model = CustomNaiveBayes(
            alpha=config.get('model_params', {}).get('multinomial', {}).get('alpha', 1.0)
        )
    else:
        model = NaiveBayesModel(
            model_type=args.model_type,
            params=config.get('model_params', {}).get(args.model_type, {})
        )

    # Fit model
    model.fit(data['X_train'], data['y_train'])

    # Evaluate model
    y_pred = model.predict(data['X_test'])
    y_prob = model.predict_proba(data['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None

    evaluator = ModelEvaluator(
        data['y_test'], y_pred, y_prob, feature_names=data['feature_names']
    )

    metrics = evaluator.calculate_metrics()
    evaluator.print_metrics(metrics)

    # Save model
    output_file = args.output or f"{args.model_type}_{args.dataset}_model.joblib"
    model_path = model.save_model(output_file)

    logger.info(f"Model saved to {model_path}")


def evaluate_command(args):
    """
    Execute the 'evaluate' command.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info(f"Evaluating model from {args.model}")

    # Check if model file exists
    if not os.path.exists(args.model):
        model_path = os.path.join(MODELS_DIR, args.model)
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {args.model}")
            return
        args.model = model_path

    # Load model
    try:
        if 'custom' in args.model:
            model = CustomNaiveBayes.load_model(args.model)
        else:
            model = NaiveBayesModel.load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Create data loader
    data_loader = EmailDataLoader(args.dataset)

    # Check if dataset is already processed
    csv_path = os.path.join(PROCESSED_DATA_DIR, args.dataset, 'email_data.csv')
    if os.path.exists(csv_path):
        logger.info(f"Loading processed data from {csv_path}")
        email_df = data_loader.load_from_csv()
    else:
        logger.info("Processed data not found, processing dataset")
        email_df = data_loader.process_dataset()

    if email_df is None:
        logger.error("Failed to load or process dataset")
        return

    # Create preprocessor
    preprocessor = EmailPreprocessor()

    # Prepare data
    data = preprocessor.prepare_data(email_df)

    # Evaluate model
    y_pred = model.predict(data['X_test'])
    y_prob = model.predict_proba(data['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None

    evaluator = ModelEvaluator(
        data['y_test'], y_pred, y_prob, feature_names=data['feature_names']
    )

    # Generate and print evaluation report
    evaluator.generate_report(model)


def classify_command(args):
    """
    Execute the 'classify' command.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    logger.info(f"Classifying email from {args.email}")

    # Check if model file exists
    if not os.path.exists(args.model):
        model_path = os.path.join(MODELS_DIR, args.model)
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {args.model}")
            return
        args.model = model_path

    # Check if email file exists
    if not os.path.exists(args.email):
        logger.error(f"Email file not found: {args.email}")
        return

    # Load model
    try:
        if 'custom' in args.model:
            model = CustomNaiveBayes.load_model(args.model)
        else:
            model = NaiveBayesModel.load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Read email
    try:
        with open(args.email, 'r', encoding='utf-8', errors='ignore') as f:
            email_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read email: {e}")
        return

    # Create preprocessor
    preprocessor = EmailPreprocessor()

    # Preprocess email
    email_df = pd.DataFrame({
        'body': [email_text],
        'label': ['unknown']
    })

    processed_df = preprocessor.process_emails(email_df)

    # Vectorize email
    if isinstance(model, CustomNaiveBayes):
        # For custom model, we need to use the same vectorizer that was used for training
        logger.error("Custom model requires vectorizer from training, cannot classify directly")
        return
    else:
        vectorizer = model.model._vectorizer if hasattr(model.model, '_vectorizer') else None
        if vectorizer is None:
            logger.error("Model does not have a vectorizer, cannot classify directly")
            return

        X = vectorizer.transform([processed_df['processed_text'].iloc[0]])

    # Classify email
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None

    # Print result
    print("\n===== EMAIL CLASSIFICATION RESULT =====")
    print(f"Classification: {'SPAM' if prediction == 1 else 'HAM'}")
    if probability is not None:
        print(f"Probability of being spam: {probability:.4f}")

    # Print email content
    print("\nEmail Content (first 300 characters):")
    print(email_text[:300] + "..." if len(email_text) > 300 else email_text)


def main():
    """
    Main function for the CLI.
    """
    args = parse_args()

    if args.command == 'download':
        download_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'classify':
        classify_command(args)
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == '__main__':
    main()