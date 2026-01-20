"""
Utility Functions Module

This module provides various utility functions used throughout the project.
"""

import os
import json
import time
from datetime import datetime
from functools import wraps


def create_directory(dir_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        dir_path (str): Path to the directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")


def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config


def get_timestamp():
    """
    Get current timestamp as a formatted string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timer(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to be timed
        
    Returns:
        wrapper: Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def log_message(message, log_file=None):
    """
    Log a message with timestamp.
    
    Args:
        message (str): Message to log
        log_file (str): Optional log file path
    """
    timestamp = get_timestamp()
    log_entry = f"[{timestamp}] {message}"
    
    print(log_entry)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # TODO: Implement metric calculation
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    return metrics
