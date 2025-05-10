"""
Setup script for the spam classifier package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spam_classifier',
    version='1.0.0',
    description='A comprehensive email spam classifier using Naive Bayes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/spam_classifier',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.22.0',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'scikit-learn>=1.0.2',
        'joblib>=1.1.0',
        'nltk>=3.7',
        'tqdm>=4.62.3',
        'matplotlib>=3.5.1',
        'seaborn>=0.11.2',
        'wordcloud>=1.8.1',
        'flask>=2.0.2',
        'flask-bootstrap>=3.3.7.1',
        'werkzeug>=2.0.2',
        'pyyaml>=6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.1.0',
            'flake8>=4.0.1',
            'isort>=5.10.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'spam_classifier=src.interface.cli:main',
            'spam_web=src.interface.web:create_app',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Filters',
    ],
)