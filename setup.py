"""
Setup script for the spam classifier package.
"""

from setuptools import setup, find_packages

setup(
    name='spam_classifier',
    version='1.0.0',
    description='A spam email classifier using Custom Naive Bayes',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.22.0',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'scikit-learn>=1.0.2',
        'joblib>=1.1.0',
        'nltk>=3.7',
        'tqdm>=4.62.3',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
    ],
)