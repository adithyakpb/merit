"""
Utility functions for test set generation.

This module provides utility functions for validation, similarity calculation,
and parallel processing.
"""

import os
import threading
import concurrent.futures
import numpy as np
from typing import Any, List, Callable, Dict
from functools import partial
import logging

from ..core.logging import get_logger
from ..core.models import Document

logger = get_logger(__name__)

# Thread-local storage for client instances
_thread_local = threading.local()

def get_thread_safe_client(original_client):
    """
    Get a thread-safe client instance.
    
    Args:
        original_client: The original client to make thread-safe
        
    Returns:
        A thread-safe client instance
    """
    if not hasattr(_thread_local, 'client'):
        # Clone the client for this thread
        # Note: This assumes the client can be cloned or is thread-safe
        _thread_local.client = original_client
    return _thread_local.client

def parallel_map(func: Callable, items: List, max_workers: int = None, 
                 use_processes: bool = False, **kwargs) -> List:
    """
    Execute a function on a list of items in parallel.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of workers (default: CPU count * 2)
        use_processes: Whether to use processes instead of threads
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2 or 4)
    
    # Create a partial function with the kwargs
    if kwargs:
        func = partial(func, **kwargs)
    
    # Choose the executor based on the task type
    executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")
                results.append((idx, None))
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]

def is_valid_input(text: str) -> bool:
    """
    Check if a string is a valid input.
    
    Args:
        text: The string to check.
        
    Returns:
        bool: True if the string is a valid input, False otherwise.
    """
    # Check if text is a string
    if not isinstance(text, str):
        return False
    
    # Minimum length check
    if len(text) < 10:
        return False
    
    # Check for JSON markers
    if text.startswith(("```", "[", "{")) or text in ["[", "{", "```", "```json"]:
        return False
    
    # Check that it contains actual words (at least 3)
    words = [w for w in text.split() if len(w) > 1]
    if len(words) < 3:
        return False
    
    return True

