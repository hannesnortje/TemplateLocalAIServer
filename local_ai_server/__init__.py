"""Local AI Server - A Flask-based server for running local language models."""

__version__ = "0.1.0"
__author__ = "Local AI Server Contributors"
__license__ = "MIT"

import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('local_ai_server.log')
    ]
)

# Suppress verbose logs from libraries
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
