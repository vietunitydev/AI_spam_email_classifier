"""
Data loading module for the spam classifier.

This module provides functions for loading email data from various sources.
"""

import os
import re
import email
import tarfile
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from email.parser import Parser
from email.header import decode_header
from src.utils.logger import logger
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class EmailDataLoader:
    """
    Class for loading and processing email datasets.

    This class provides methods for downloading, extracting, and organizing
    email datasets for spam classification.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('spamassassin' or 'enron').
    raw_dir : str, optional
        Directory for storing raw data.
    processed_dir : str, optional
        Directory for storing processed data.
    """

    def __init__(self, dataset_name, raw_dir=None, processed_dir=None):
        """Initialize the EmailDataLoader."""
        self.dataset_name = dataset_name.lower()
        self.raw_dir = raw_dir or os.path.join(RAW_DATA_DIR, self.dataset_name)
        self.processed_dir = processed_dir or os.path.join(PROCESSED_DATA_DIR, self.dataset_name)
        self.spam_dir = os.path.join(self.processed_dir, 'spam')
        self.ham_dir = os.path.join(self.processed_dir, 'ham')

        self.dataset_urls = {
            'spamassassin': [
                # 2002 collections
                'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2',

                # 2003 collections
                'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2',
                'https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2',

                # 2005 collection
                'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
            ],
            'enron': [
            # Đầy đủ bộ dữ liệu Enron (6 tập)
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz',
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz',
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron3.tar.gz',
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron4.tar.gz',
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron5.tar.gz',
            'http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron6.tar.gz'
        ]
        }

        # Directory structure for each dataset
        self.dataset_structures = {
            'spamassassin': {
                'spam_folders': [
                    'spam',  # từ 20021010_spam
                    'spam_2',  # từ 20030228_spam_2
                    '20030228_spam',  # từ 20030228_spam
                    '20050311_spam_2'  # từ 20050311_spam_2
                ],
                'ham_folders': [
                    'easy_ham',  # từ 20021010_easy_ham
                    'hard_ham',  # từ 20021010_hard_ham
                    '20030228_easy_ham',  # từ 20030228_easy_ham
                    '20030228_easy_ham_2',  # từ 20030228_easy_ham_2
                    '20030228_hard_ham'  # từ 20030228_hard_ham
                ]
            },
            'enron': {
                'spam_folders': [
                    'enron1/spam',
                    'enron2/spam',
                    'enron3/spam',
                    'enron4/spam',
                    'enron5/spam',
                    'enron6/spam'
                ],
                'ham_folders': [
                    'enron1/ham',
                    'enron2/ham',
                    'enron3/ham',
                    'enron4/ham',
                    'enron5/ham',
                    'enron6/ham'
                ]
            }
        }

        # Create all necessary directories
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.spam_dir, exist_ok=True)
        os.makedirs(self.ham_dir, exist_ok=True)

        logger.info(f"Initialized EmailDataLoader for {dataset_name} dataset with full corpus")

    def download_dataset(self):
        """
        Download the dataset from the internet.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self.dataset_name not in self.dataset_urls:
            logger.error(f"Dataset '{self.dataset_name}' not supported")
            return False

        logger.info(f"Downloading {self.dataset_name} dataset")

        # Download each file
        for url in self.dataset_urls[self.dataset_name]:
            filename = url.split('/')[-1]
            filepath = os.path.join(self.raw_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                logger.info(f"File {filename} already exists, skipping download")
                continue

            logger.info(f"Downloading {filename} from {url}")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
                return False

        logger.info("Dataset download complete")
        return True

    def extract_dataset(self):
        """
        Extract the downloaded dataset.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        logger.info(f"Extracting {self.dataset_name} dataset")

        # Extract each file in the raw directory
        for filename in os.listdir(self.raw_dir):
            filepath = os.path.join(self.raw_dir, filename)

            try:
                if filename.endswith('.tar.bz2'):
                    logger.info(f"Extracting {filename}")
                    with tarfile.open(filepath, 'r:bz2') as tar:
                        tar.extractall(path=self.raw_dir)
                    logger.info(f"Successfully extracted {filename}")

                elif filename.endswith('.tar.gz'):
                    logger.info(f"Extracting {filename}")
                    with tarfile.open(filepath, 'r:gz') as tar:
                        tar.extractall(path=self.raw_dir)
                    logger.info(f"Successfully extracted {filename}")

                elif filename.endswith('.zip'):
                    logger.info(f"Extracting {filename}")
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(path=self.raw_dir)
                    logger.info(f"Successfully extracted {filename}")
            except Exception as e:
                logger.error(f"Error extracting {filename}: {e}")
                return False

        logger.info("Dataset extraction complete")
        return True

    def organize_files(self):
        """
        Organize the extracted files into spam and ham directories.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        structure = self.dataset_structures.get(self.dataset_name)
        if not structure:
            logger.error(f"No structure defined for dataset '{self.dataset_name}'")
            return False

        logger.info(f"Organizing files for {self.dataset_name} dataset")

        # Move files from spam folders
        for folder in structure['spam_folders']:
            folder_path = os.path.join(self.raw_dir, folder)
            if os.path.exists(folder_path):
                self._copy_email_files(folder_path, self.spam_dir, 'spam')

        # Move files from ham folders
        for folder in structure['ham_folders']:
            folder_path = os.path.join(self.raw_dir, folder)
            if os.path.exists(folder_path):
                self._copy_email_files(folder_path, self.ham_dir, 'ham')

        logger.info("File organization complete")
        return True

    def _copy_email_files(self, source_dir, target_dir, label):
        """
        Copy email files from source directory to target directory.

        Parameters
        ----------
        source_dir : str
            Source directory.
        target_dir : str
            Target directory.
        label : str
            Label for the files ('spam' or 'ham').
        """
        # Count existing files in target directory
        existing_files = len(os.listdir(target_dir))

        # Get all files in source directory
        source_files = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Skip non-email files
                if file.startswith('.') or file in ['cmds']:
                    continue
                source_files.append(os.path.join(root, file))

        logger.info(f"Copying {len(source_files)} {label} files")

        # Copy each file with progress bar
        for i, filepath in tqdm(enumerate(source_files), total=len(source_files)):
            try:
                # Read file content
                with open(filepath, 'rb') as f:
                    content = f.read()

                # Save to target directory with unique name
                new_filename = f"{label}_{existing_files + i + 1}.txt"
                new_filepath = os.path.join(target_dir, new_filename)

                with open(new_filepath, 'wb') as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Error copying {filepath}: {e}")

    def parse_emails(self, limit=None):
        """
        Parse emails to extract relevant information.

        Parameters
        ----------
        limit : int, optional
            Maximum number of emails to parse from each class.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parsed email information.
        """
        spam_files = os.listdir(self.spam_dir)
        ham_files = os.listdir(self.ham_dir)

        # Apply limit if specified
        if limit is not None:
            spam_files = spam_files[:limit]
            ham_files = ham_files[:limit]

        logger.info(f"Parsing {len(spam_files)} spam and {len(ham_files)} ham emails")

        # Data for DataFrame
        data = []

        # Parse spam emails
        for filename in tqdm(spam_files, desc="Parsing spam emails"):
            filepath = os.path.join(self.spam_dir, filename)
            email_info = self._extract_email_info(filepath, 'spam')
            if email_info:
                data.append(email_info)

        # Parse ham emails
        for filename in tqdm(ham_files, desc="Parsing ham emails"):
            filepath = os.path.join(self.ham_dir, filename)
            email_info = self._extract_email_info(filepath, 'ham')
            if email_info:
                data.append(email_info)

        # Create DataFrame
        email_df = pd.DataFrame(data)
        logger.info(f"Parsed a total of {len(email_df)} emails")

        return email_df

    def _extract_email_info(self, filepath, label):
        """
        Extract information from an email file.

        Parameters
        ----------
        filepath : str
            Path to the email file.
        label : str
            Label of the email ('spam' or 'ham').

        Returns
        -------
        dict
            Dictionary containing the extracted information.
        """
        try:
            # Read email
            with open(filepath, 'rb') as f:
                raw_email = f.read()

            # Handle encoding errors
            try:
                # Parse email content
                msg = email.message_from_bytes(raw_email)
            except:
                # Try as string if bytes fails
                msg = email.message_from_string(raw_email.decode('latin1', errors='ignore'))

            # Extract subject
            subject = self._decode_header(msg.get('Subject', ''))

            # Extract sender
            from_addr = self._decode_header(msg.get('From', ''))

            # Extract recipient
            to_addr = self._decode_header(msg.get('To', ''))

            # Extract date
            date = self._decode_header(msg.get('Date', ''))

            # Extract body
            body = self._get_email_body(msg)

            # Return information
            return {
                'filepath': filepath,
                'label': label,
                'subject': subject,
                'from': from_addr,
                'to': to_addr,
                'date': date,
                'body': body,
                'length': len(body) if body else 0
            }

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None

    def _decode_header(self, header):
        """
        Decode email header.

        Parameters
        ----------
        header : str
            Email header to decode.

        Returns
        -------
        str
            Decoded header.
        """
        if not header:
            return ""

        try:
            # Decode email header
            decoded_parts = []
            for part, encoding in decode_header(header):
                if isinstance(part, bytes):
                    # If encoding is not specified, assume utf-8
                    if encoding is None:
                        encoding = 'utf-8'
                    try:
                        decoded_parts.append(part.decode(encoding, errors='replace'))
                    except:
                        decoded_parts.append(part.decode('latin1', errors='replace'))
                else:
                    decoded_parts.append(part)

            return ''.join(decoded_parts)
        except Exception as e:
            # Return original string if error
            logger.error(f"Error decoding header: {e}")
            return header

    def _get_email_body(self, msg):
        """
        Extract the body of an email.

        Parameters
        ----------
        msg : email.message.Message
            Parsed email message.

        Returns
        -------
        str
            Email body text.
        """
        if msg.is_multipart():
            # If email has multiple parts
            body_parts = []
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))

                # Get text content
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    try:
                        body = part.get_payload(decode=True)
                        charset = part.get_content_charset()
                        if charset:
                            body = body.decode(charset, errors='replace')
                        else:
                            body = body.decode('utf-8', errors='replace')
                        body_parts.append(body)
                    except Exception as e:
                        logger.error(f"Error decoding email part: {e}")
                        body_parts.append(f"[Error decoding content: {e}]")

            return '\n'.join(body_parts)
        else:
            # If email has only one part
            body = msg.get_payload(decode=True)
            if body:
                charset = msg.get_content_charset()
                if charset:
                    try:
                        return body.decode(charset, errors='replace')
                    except:
                        return body.decode('utf-8', errors='replace')
                else:
                    return body.decode('utf-8', errors='replace')
            return ""

    def save_to_csv(self, email_df, filename='email_data.csv'):
        """
        Save the parsed email data to a CSV file.

        Parameters
        ----------
        email_df : pandas.DataFrame
            DataFrame containing the parsed email data.
        filename : str, optional
            Name of the CSV file.

        Returns
        -------
        str
            Path to the saved CSV file.
        """
        # Create a copy to avoid modifying the original DataFrame
        df_to_save = email_df.copy()

        # Remove filepath column if it exists
        if 'filepath' in df_to_save.columns:
            df_to_save = df_to_save.drop('filepath', axis=1)

        # Save the DataFrame with proper escape characters and quoting
        filepath = os.path.join(self.processed_dir, filename)

        # Thêm các tham số này để xử lý ký tự đặc biệt
        df_to_save.to_csv(
            filepath,
            index=False,
            escapechar='\\',  # Sử dụng ký tự backslash để escape
            quoting=1,  # Quoting các trường chứa ký tự đặc biệt
            doublequote=True,  # Sử dụng hai dấu nháy để escape dấu nháy
            quotechar='"'  # Sử dụng dấu nháy kép cho quoting
        )

        logger.info(f"Saved processed data to {filepath}")

        return filepath

    def load_from_csv(self, filename='email_data.csv'):
        """
        Load the parsed email data from a CSV file.

        Parameters
        ----------
        filename : str, optional
            Name of the CSV file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parsed email data.
        """
        filepath = os.path.join(self.processed_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"CSV file {filepath} not found")
            return None

        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)

    def process_dataset(self, limit=None, save_to_csv=True):
        """
        Process the complete dataset.

        Parameters
        ----------
        limit : int, optional
            Maximum number of emails to parse from each class.
        save_to_csv : bool, optional
            Whether to save the processed data to a CSV file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the processed email data.
        """
        # Check if processed data already exists
        csv_path = os.path.join(self.processed_dir, 'email_data.csv')
        if os.path.exists(csv_path):
            logger.info(f"Processed data already exists at {csv_path}")
            return pd.read_csv(csv_path)

        # Download dataset
        if not self.download_dataset():
            logger.error("Dataset download failed")
            return None

        # Extract dataset
        if not self.extract_dataset():
            logger.error("Dataset extraction failed")
            return None

        # Organize files
        if not self.organize_files():
            logger.error("File organization failed")
            return None

        # Parse emails
        email_df = self.parse_emails(limit=limit)

        # Save to CSV if requested
        if save_to_csv:
            self.save_to_csv(email_df)

        return email_df