"""
MERIT Universal Logging System

This module provides a comprehensive logging system for the MERIT package.
It ensures all logs include source information, timezone, and supports loop progress tracking.
All logs are directed to the same file, regardless of which module they come from.
"""

import os
import sys
import logging
import datetime
import threading
from logging.handlers import RotatingFileHandler

# Thread-local storage for loop context
_thread_local = threading.local()

class MERITLogger(logging.Logger):
    """
    Enhanced logger that includes source information, timezone, and loop progress tracking.
    """
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the MERITLogger with the given name and level."""
        super().__init__(name, level)
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        """
        Override _log to add loop progress information if available.
        """
        # Check if we're in a loop context
        if hasattr(_thread_local, 'loop_context'):
            context = _thread_local.loop_context
            if context:
                current, total = context
                msg = f"[{current}/{total}] {msg}"
        
        # Call the parent class's _log method
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
    
    def log_loop(self, iterable, message=None, level=logging.INFO):
        """
        Log iterations through an iterable with progress tracking.
        
        Args:
            iterable: The iterable to loop through
            message: Optional message prefix for logs
            level: Logging level to use
            
        Returns:
            Generator that yields items from the iterable with logging
        """
        items = list(iterable)  # Convert to list to get length
        total = len(items)
        
        if message:
            self.log(level, f"{message} - Starting loop with {total} items")
        
        for i, item in enumerate(items):
            # Set loop context for any logs within the loop body
            if not hasattr(_thread_local, 'loop_context'):
                _thread_local.loop_context = None
            
            _thread_local.loop_context = (i+1, total)
            
            try:
                yield item
            except Exception as e:
                self.error(f"Error processing item {i+1}/{total}: {str(e)}", exc_info=True)
                raise
            finally:
                _thread_local.loop_context = None
        
        if message:
            self.log(level, f"{message} - Completed all {total} items")


# Setup function
def setup_logging(
    log_file: str = "merit.log",
    log_level: str = "INFO",
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Set up the MERIT logging system.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_file_size: Maximum size of log file before rotation (bytes)
        backup_count: Number of backup log files to keep
    """
    # Register our custom logger class
    logging.setLoggerClass(MERITLogger)
    
    # Create a custom log record factory to add timezone information
    original_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        record.timezone = datetime.datetime.now().astimezone().strftime('%z')
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    # Create formatter with timezone and source information
    formatter = logging.Formatter(
        '%(asctime)s %(timezone)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler with the same formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set up global exception handler
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        # Skip KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        logger = logging.getLogger("merit.uncaught")
        logger.error(
            "Uncaught exception:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    # Set the exception hook
    sys.excepthook = global_exception_handler
    
    # Log configuration details
    root_logger.info(f"MERIT logging configured: file={log_file}, level={log_level}")


# Helper function to get a logger
def get_logger(name: str) -> MERITLogger:
    """
    Get a MERITLogger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        MERITLogger: A configured MERITLogger instance
    """
    return logging.getLogger(name)
