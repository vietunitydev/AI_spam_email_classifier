# Email Spam Classifier using Naive Bayes

A comprehensive email spam classification system built with Python. This project provides tools for downloading and processing email datasets, training and evaluating Naive Bayes models, and classifying new emails as spam or ham (non-spam).

## Features

- **Data Processing** : Download and preprocess standard email spam datasets (SpamAssassin, Enron)
- **Text Processing** : Tokenization, stopwords removal, stemming/lemmatization
- **Feature Extraction**: Convert email text to numerical features (Bag of Words, TF-IDF)
- **Multiple Models**: Various Naive Bayes implementations (Multinomial, Bernoulli, Gaussian, Complement)
- **Custom Implementation**: Educational implementation of Naive Bayes from scratch
- **Evaluation Tools**: Comprehensive model evaluation with various metrics and visualizations
- **Command-line Interface**: Train models and classify emails from the command line
- **Web Interface**: User-friendly web interface for email classification

## Installation
### Requirements

- Python 3.8 or higher

### Install from source

1. Clone this repository:
    ```bash
    git clone https://github.com/vietunitydev/AI_spam_email_classifier.git
    cd AI_spam_email_classifier
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package and its dependencies:
    ```bash
    pip install -e .
- For development, include additional tools:
    ```bash
    pip install -e ".[dev]"

## RUN QUICKLY
1. Down load dataset and train model
    ```bash
    python3 run_full_step.py # On Windows : python run_full_step.py
2. Run app 
    ```bash
   python3 step/app_ui.py # On Windows : python step/app_ui.py

## How to run sequence code
1. ```bash
   python3 step/fix_nltk.py # On Windows : python step/fix_nltk.py
2. ```bash
   python3 step/step1.py # On Windows : python step/step1.py
3. ```bash
   python3 step/step2.py # On Windows : python step/step2.py
4. ```bash
   python3 step/step3.py # On Windows : python step/step3.py
5. ```bash
   python3 step/step4.py # On Windows : python step/step4.py
6. ```bash
   python3 step/step5.py # On Windows : python step/step5.py
7. ```bash
   python3 step/app_ui.py # On Windows : python step/app_ui.py

# Documentation
## Data Module
### EmailDataLoader
Class for loading and processing email datasets.
- __init__(dataset_name, raw_dir=None, processed_dir=None): Initialize the data loader.
- download_dataset(): Download the dataset from the internet.
- extract_dataset(): Extract the downloaded dataset.
- organize_files(): Organize the extracted files into spam and ham directories.
- parse_emails(limit=None): Parse emails to extract relevant information.
- save_to_csv(email_df, filename='email_data.csv'): Save the parsed email data to a CSV file.
- load_from_csv(filename='email_data.csv'): Load the parsed email data from a CSV file.
- process_dataset(limit=None, save_to_csv=True): Process the complete dataset.

### EmailPreprocessor
Class for preprocessing email text data.

- __init__(config=None): Initialize the preprocessor.
- clean_text(text): Clean and normalize email text.
- tokenize_text(text): Tokenize text and apply stemming/lemmatization if configured.
- process_emails(email_df, text_column='body'): Process emails by cleaning and tokenizing the text.
- vectorize_text(processed_df, vectorizer_type='tfidf'): Vectorize the processed text.
- prepare_data(email_df, vectorizer_type='tfidf', text_column='body', config=None): Prepare data for training and testing.

## Models Module
### NaiveBayesModel
Wrapper class for Naive Bayes models for spam classification.

- __init__(model_type='multinomial', params=None): Initialize the model.
- fit(X_train, y_train): Train the model on the given data.
- predict(X): Predict class labels for samples in X.
- predict_proba(X): Predict class probabilities for samples in X.
- save_model(filename=None): Save the trained model to a file.
- load_model(filepath): Load a trained model from a file.
- get_feature_importances(feature_names): Get the feature importances for the model.

### CustomNaiveBayes
Custom implementation of Naive Bayes for educational purposes.

- __init__(alpha=1.0, fit_prior=True): Initialize the model.
- fit(X, y): Fit the model according to the training data.
- predict_log_proba(X): Return log-probability estimates for samples in X.
- predict_proba(X): Return probability estimates for samples in X.
- predict(X): Perform classification on samples in X.
- save_model(filename=None): Save the trained model to a file.
- load_model(filepath): Load a trained model from a file.

## Evaluation Module
### ModelEvaluator
Class for evaluating spam classification models.

- __init__(y_true, y_pred, y_prob=None, feature_names=None): Initialize the evaluator.
- calculate_metrics(): Calculate evaluation metrics.
- print_metrics(metrics=None): Print evaluation metrics.
- plot_confusion_matrix(metrics=None): Plot confusion matrix.
- plot_roc_curve(metrics=None): Plot ROC curve.
- plot_precision_recall_curve(metrics=None): Plot precision-recall curve.
- plot_feature_importance(model, top_n=20): Plot feature importance.
- generate_report(model=None): Generate a comprehensive evaluation report.

## Interface Module
### Command-line Interface (CLI)

- download: Download and process a dataset.
- train: Train a Naive Bayes model.
- evaluate: Evaluate a trained model.
- classify: Classify a single email.

## Web Interface

- Homepage: Select a model and input email text or upload an email file.
- Classification: Display classification result (spam or ham) and probability.

## Project Structure
- spam_classifier/
- │
- ├── data/                  # Data directory
- │   ├── raw/               # Raw data
- │   └── processed/         # Processed data
- │
- ├── models/                # Trained models
- │
- ├── logs/                  # Log files
- │
- ├── src/                   # Source code
- │   ├── data/              # Data processing
- │   ├── models/            # Model implementations
- │   ├── evaluation/        # Evaluation tools
- │   ├── utils/             # Utilities
- │   └── interface/         # User interfaces
- │
- ├── tests/                 # Unit tests
- │
- ├── main.py                # Main entry point
- ├── requirements.txt       # Dependencies
- ├── setup.py               # Package setup
- └── README.md              # This file

## License
MIT License
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## Acknowledgments

- The SpamAssassin Public Corpus: https://spamassassin.apache.org/old/publiccorpus/
- The Enron Spam Dataset: http://www.aueb.gr/users/ion/data/enron-spam/