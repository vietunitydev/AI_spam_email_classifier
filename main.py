"""
Main entry point for the spam classifier application.

This module provides a command-line interface for training and using the spam classifier.
"""

import sys
from src.interface.cli import main as cli_main

if __name__ == '__main__':
    # Call the CLI main function
    sys.exit(cli_main())