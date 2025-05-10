"""
Web interface for the spam classifier.

This module provides a web interface for using the spam classifier.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from ..data.preprocessor import EmailPreprocessor
from ..models.naive_bayes import NaiveBayesModel, CustomNaiveBayes
from ..utils.logger import logger
from ..utils.config import MODELS_DIR

app = Flask(__name__, template_folder='templates')
Bootstrap(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'eml'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
preprocessor = None
vectorizer = None


def allowed_file(filename):
    """
    Check if the file extension is allowed.

    Parameters
    ----------
    filename : str
        Name of the file.

    Returns
    -------
    bool
        True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model(model_path):
    """
    Load a trained model from a file.

    Parameters
    ----------
    model_path : str
        Path to the model file.

    Returns
    -------
    tuple
        Model instance and vectorizer.
    """
    global model, preprocessor, vectorizer

    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            full_path = os.path.join(MODELS_DIR, model_path)
            if not os.path.exists(full_path):
                return None, None
            model_path = full_path

        # Load model
        model_obj = joblib.load(model_path)

        # Initialize preprocessor
        preprocessor = EmailPreprocessor()

        # Get vectorizer from model if available
        if hasattr(model_obj, 'vectorizer'):
            vectorizer = model_obj.vectorizer
        elif hasattr(model_obj, 'model') and hasattr(model_obj.model, '_vectorizer'):
            vectorizer = model_obj.model._vectorizer
        else:
            vectorizer = None

        return model_obj, vectorizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


@app.route('/')
def home():
    """Render the home page."""
    # Get list of available models
    models = []
    if os.path.exists(MODELS_DIR):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]

    return render_template('index.html', models=models)


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify an email from text or file upload.

    Returns
    -------
    dict
        Classification result.
    """
    global model, preprocessor, vectorizer

    # Load model if specified
    model_path = request.form.get('model')
    if model_path:
        model, vectorizer = load_model(model_path)
        if model is None:
            return jsonify({
                'error': f"Failed to load model: {model_path}"
            })

    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': "No model loaded. Please select a model."
        })

    # Get email text
    email_text = request.form.get('email_text')

    # If email_text is empty, check for file upload
    if not email_text:
        if 'email_file' not in request.files:
            return jsonify({
                'error': "No email text or file provided."
            })

        file = request.files['email_file']
        if file.filename == '':
            return jsonify({
                'error': "No file selected."
            })

        if not allowed_file(file.filename):
            return jsonify({
                'error': f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
            })

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read file
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                email_text = f.read()
        except Exception as e:
            return jsonify({
                'error': f"Failed to read file: {e}"
            })

    # Preprocess email
    try:
        email_df = pd.DataFrame({
            'body': [email_text],
            'label': ['unknown']
        })

        processed_df = preprocessor.process_emails(email_df)
        processed_text = processed_df['processed_text'].iloc[0]

        # Vectorize email
        if vectorizer is None:
            return jsonify({
                'error': "Model does not have a vectorizer, cannot classify directly."
            })

        X = vectorizer.transform([processed_text])

        # Classify email
        prediction = model.predict(X)[0]
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]

        return jsonify({
            'is_spam': bool(prediction == 1),
            'probability': float(probability) if probability is not None else None,
            'processed_text': processed_text
        })
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'error': f"Classification failed: {e}"
        })


def create_app():
    """
    Create and configure the Flask application.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    return app


if __name__ == '__main__':
    app.run(debug=True)