"""
Logging configuration for the ML recommendation engine.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from .constants import BASE_DIR

# Create logs directory
LOGS_DIR = BASE_DIR / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
def setup_logging(name=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        name: Logger name (defaults to root logger if None)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Create file handler
    log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(LOGS_DIR / log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logging('ml_engine')
